"""Couple a Qucs-S external netlist to the one-zone global plasma model."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from plasma_circuit.ngspice import (
    SimulationMetrics,
    SimulationResult,
    _cycle_l2,
    _fmt,
    _phasor,
    _window,
)
from plasma_circuit.physics import (
    PlasmaParameters,
    compute_plasma_parameters,
    density_from_power_balance,
    electron_temperature_from_particle_balance,
)


VECTOR_SPECS = (
    ("v_source", "v({source_node})"),
    ("v_feed", "v({interface_node})"),
    ("v_plasma", "v(plasma)"),
    ("v_sheath_powered", "v(bulk1,plasma)"),
    ("v_sheath_grounded", "v(bulk2)"),
    ("i_source_device", "i(Vsense_generator_qucs)"),
    ("i_plasma_port", "i(Vsense_surface_wafer)"),
    ("i_bulk", "i(Vsense_bulk)"),
)


@dataclass(frozen=True)
class QucsDensityIteration:
    iteration: int
    density_input_m3: float
    density_target_m3: float
    density_output_m3: float
    balance_relative_residual: float
    absorbed_power_w: float
    mean_sheath_powered_v: float
    mean_sheath_grounded_v: float
    input_impedance_real_ohm: float
    input_impedance_imag_ohm: float
    cycle_l2_voltage: float
    cycle_l2_current: float


@dataclass(frozen=True)
class QucsOneZoneCoupledResult:
    electron_temperature_ev: float
    electron_density_m3: float
    converged: bool
    history: tuple[QucsDensityIteration, ...]
    final_simulation: SimulationResult
    final_density_target_m3: float
    final_balance_relative_residual: float


def load_qucs_one_zone_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a Qucs-S one-zone coupling configuration."""
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "ngspice_path",
        "frequency_hz",
        "source_amplitude_v",
        "source_resistance_ohm",
        "pressure_pa",
        "gas_temperature_k",
        "powered_area_m2",
        "grounded_area_m2",
        "bulk_length_m",
        "regularization",
        "qucs_netlist",
        "transient",
        "coupling",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing Qucs one-zone configuration fields: {missing}")
    ngspice_path = Path(config["ngspice_path"])
    if not ngspice_path.is_file():
        raise FileNotFoundError(f"ngspice executable not found: {ngspice_path}")
    qucs = config["qucs_netlist"]
    for name in (
        "path",
        "interface_node",
        "source_node",
        "source_device",
        "series_loss_resistance_ohm",
    ):
        if name not in qucs:
            raise ValueError(f"missing qucs_netlist field: {name}")
    raw_path = Path(qucs["path"])
    resolved = raw_path if raw_path.is_absolute() else config_path.parent / raw_path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Qucs-S netlist not found: {resolved}")
    qucs["resolved_path"] = str(resolved)
    if int(config["transient"]["saved_cycles"]) < 3:
        raise ValueError("at least three saved RF cycles are required")
    return config


def _device_names(netlist: str) -> set[str]:
    names: set[str] = set()
    for line in netlist.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("*", ".")):
            continue
        names.add(stripped.split()[0].lower())
    return names


def normalize_qucs_netlist(raw_netlist: str, qucs: Mapping[str, Any]) -> str:
    """Remove Qucs-S analysis control while preserving the external circuit body."""
    output = ["* Normalized Qucs-S external circuit body"]
    in_control = False
    dropped_includes = {
        str(name).lower() for name in qucs.get("drop_include_basenames", [])
    }
    for raw_line in raw_netlist.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        stripped = raw_line.strip()
        lower = stripped.lower()
        if lower == ".control":
            if in_control:
                raise ValueError("nested .control section in Qucs-S netlist")
            in_control = True
            continue
        if in_control:
            if lower == ".endc":
                in_control = False
            continue
        if lower in {".end", ".endc"}:
            continue
        if lower.startswith((".tran", ".ac", ".dc", ".op")):
            continue
        if lower.startswith((".include", ".inc")) and any(
            name in lower for name in dropped_includes
        ):
            continue
        if lower.startswith("* qucs"):
            continue
        if stripped:
            output.append(stripped)
    if in_control:
        raise ValueError("unterminated .control section in Qucs-S netlist")
    normalized = "\n".join(output) + "\n"
    if not re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(str(qucs['interface_node']))}(?![A-Za-z0-9_])",
        normalized,
        flags=re.IGNORECASE,
    ):
        raise ValueError(f"interface node {qucs['interface_node']!r} not found")
    names = _device_names(normalized)
    missing = [
        name for name in qucs.get("required_devices", []) if str(name).lower() not in names
    ]
    if missing:
        raise ValueError(f"required Qucs-S devices not found: {missing}")
    forbidden = [
        name for name in qucs.get("forbidden_devices", []) if str(name).lower() in names
    ]
    if forbidden:
        raise ValueError(f"forbidden Qucs-S devices remain in netlist: {forbidden}")
    source_device = str(qucs["source_device"]).lower()
    source_lines = [
        line for line in normalized.splitlines() if line.lower().startswith(source_device + " ")
    ]
    if len(source_lines) != 1 or "sin(" not in source_lines[0].lower():
        raise ValueError("Qucs-S source must be one transient SIN voltage source")
    return normalized


def apply_component_overrides(
    normalized_netlist: str, overrides: Mapping[str, float]
) -> str:
    """Replace scalar R/L/C values while preserving the exported topology."""
    if not overrides:
        return normalized_netlist
    pending = {str(name).lower(): float(value) for name, value in overrides.items()}
    if any(not np.isfinite(value) or value <= 0.0 for value in pending.values()):
        raise ValueError("Qucs-S component overrides must be finite and positive")
    output: list[str] = []
    replaced: set[str] = set()
    for line in normalized_netlist.splitlines():
        fields = line.split()
        if fields and fields[0].lower() in pending:
            device = fields[0]
            if device[0].lower() not in {"r", "l", "c"} or len(fields) < 4:
                raise ValueError(
                    f"component override supports scalar R/L/C devices only: {device}"
                )
            fields[3] = _fmt(pending[device.lower()])
            line = " ".join(fields)
            replaced.add(device.lower())
        output.append(line)
    missing = sorted(set(pending) - replaced)
    if missing:
        raise ValueError(f"Qucs-S component override devices not found: {missing}")
    return "\n".join(output) + "\n"


def apply_series_inductor_quality_factor(
    normalized_netlist: str,
    device_name: str,
    frequency_hz: float,
    quality_factor: float,
) -> str:
    """Add a series ESR to one exported inductor using R=omega*L/Q."""
    if not np.isfinite(quality_factor) or quality_factor <= 0.0:
        raise ValueError("series inductor quality factor must be finite and positive")
    target = device_name.lower()
    output: list[str] = []
    replaced = False
    for line in normalized_netlist.splitlines():
        fields = line.split()
        if fields and fields[0].lower() == target:
            if fields[0][0].lower() != "l" or len(fields) != 4:
                raise ValueError(
                    f"quality-factor model requires a scalar inductor: {device_name}"
                )
            inductance_h = float(fields[3])
            resistance_ohm = (
                2.0 * np.pi * float(frequency_hz) * inductance_h / quality_factor
            )
            internal_node = f"qucs_{target}_esr_internal"
            output.append(f"{fields[0]} {fields[1]} {internal_node} {fields[3]}")
            output.append(
                f"Rloss_{fields[0]} {internal_node} {fields[2]} {_fmt(resistance_ohm)}"
            )
            replaced = True
        else:
            output.append(line)
    if not replaced:
        raise ValueError(f"series inductor for quality-factor model not found: {device_name}")
    return "\n".join(output) + "\n"


def _prepare_external_body(
    raw_netlist: str, config: Mapping[str, Any]
) -> str:
    normalized = normalize_qucs_netlist(raw_netlist, config["qucs_netlist"])
    normalized = apply_component_overrides(
        normalized,
        config["qucs_netlist"].get("component_overrides", {}),
    )
    matching = config.get("matching", {})
    quality_factor = matching.get("series_inductor_quality_factor")
    if quality_factor is not None:
        normalized = apply_series_inductor_quality_factor(
            normalized,
            str(matching.get("series_inductor_device", "L1")),
            float(config["frequency_hz"]),
            float(quality_factor),
        )
    return normalized


def _vector_specs(config: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    qucs = config["qucs_netlist"]
    substitutions = {
        "source_node": str(qucs["source_node"]),
        "interface_node": str(qucs["interface_node"]),
        "source_device": str(qucs["source_device"]),
    }
    return tuple((name, expression.format(**substitutions)) for name, expression in VECTOR_SPECS)


def series_inductor_esr_ohm(config: Mapping[str, Any]) -> float:
    """Return the optional Q-derived loss inserted in series with the Qucs inductor."""
    matching = config.get("matching", {})
    quality_factor = matching.get("series_inductor_quality_factor")
    if quality_factor is None:
        return 0.0
    device = str(matching.get("series_inductor_device", "L1"))
    overrides = config["qucs_netlist"].get("component_overrides", {})
    value = next(
        (
            float(component_value)
            for component_name, component_value in overrides.items()
            if str(component_name).lower() == device.lower()
        ),
        None,
    )
    if value is None:
        raise ValueError(
            "Q-derived series loss requires an explicit component override for "
            f"{device}"
        )
    return float(
        2.0
        * np.pi
        * float(config["frequency_hz"])
        * value
        / float(quality_factor)
    )


def _replace_source_with_ramped_source(
    normalized: str, config: Mapping[str, Any]
) -> str:
    """Replace the exported SIN source only in the coupled case to tame startup."""
    qucs = config["qucs_netlist"]
    source_device = str(qucs["source_device"])
    source_node = str(qucs["source_node"])
    output: list[str] = []
    replaced = False
    for line in normalized.splitlines():
        fields = line.split()
        if fields and fields[0].lower() == source_device.lower():
            if len(fields) < 3 or fields[1].lower() != source_node.lower() or fields[2] != "0":
                raise ValueError(
                    "Qucs-S source terminals do not match configured source_node and ground"
                )
            frequency = float(config["frequency_hz"])
            ramp_time = float(config.get("source_ramp_cycles", 10.0)) / frequency
            phase_rad = np.deg2rad(float(config.get("source_phase_deg", 0.0)))
            output.extend(
                (
                    "* Qucs-S SIN source replaced by an equal steady-state source with startup ramp.",
                    "Bsource_qucs qucs_source 0 "
                    f"V='{_fmt(float(config['source_amplitude_v']))}*"
                    f"sin(2*pi*freq*time+{_fmt(float(phase_rad))})*"
                    f"tanh(time/{_fmt(ramp_time)})'",
                    f"Vsense_generator_qucs qucs_source {source_node} 0",
                )
            )
            replaced = True
        else:
            output.append(line)
    if not replaced:
        raise ValueError(f"Qucs-S source {source_device!r} was not replaced")
    return "\n".join(output) + "\n"


def render_qucs_one_zone_netlist(
    config: Mapping[str, Any], plasma: PlasmaParameters, raw_netlist: str
) -> str:
    """Assemble the normalized Qucs-S RLC and nonlinear one-zone plasma circuit."""
    transient = config["transient"]
    regularization = config["regularization"]
    frequency = float(config["frequency_hz"])
    period = 1.0 / frequency
    total_time = int(transient["cycles"]) * period
    saved_start = total_time - int(transient["saved_cycles"]) * period
    step = period / int(transient["samples_per_cycle"])
    vectors = " ".join(expression for _, expression in _vector_specs(config))
    normalized_body = _prepare_external_body(raw_netlist, config)
    body = _replace_source_with_ramped_source(normalized_body, config)
    eq_v = plasma.equilibrium_sheath_voltage_v
    interface_node = str(config["qucs_netlist"]["interface_node"])
    shunt_capacitance = float(
        config.get("matching", {}).get("shunt_capacitance_f", 0.0)
    )
    shunt_inductance = float(
        config.get("matching", {}).get("shunt_inductance_h", 0.0)
    )
    if not np.isfinite(shunt_capacitance) or shunt_capacitance < 0.0:
        raise ValueError("matching shunt capacitance must be finite and non-negative")
    if not np.isfinite(shunt_inductance) or shunt_inductance < 0.0:
        raise ValueError("matching shunt inductance must be finite and non-negative")
    if shunt_capacitance > 0.0 and shunt_inductance > 0.0:
        raise ValueError("select either shunt capacitance or shunt inductance, not both")
    if shunt_capacitance > 0.0:
        shunt_element = (
            f"Cmatch_shunt_qucs {interface_node} 0 {_fmt(shunt_capacitance)}\n"
        )
    elif shunt_inductance > 0.0:
        shunt_element = (
            f"Lmatch_shunt_qucs {interface_node} 0 {_fmt(shunt_inductance)}\n"
        )
    else:
        shunt_element = ""
    return f"""Qucs-S RLC external circuit coupled to one-zone global plasma model
.param freq={_fmt(frequency)}
.param te_ev={_fmt(plasma.electron_temperature_ev)}
.param veps={_fmt(float(regularization['electron_voltage_v']))}
.param vcap={_fmt(float(regularization['capacitance_voltage_v']))}
.param cscale={_fmt(float(regularization['capacitance_scale']))}
.param ksh1={_fmt(plasma.sheath_k_powered)}
.param ksh2={_fmt(plasma.sheath_k_grounded)}
.param iesat1={_fmt(plasma.electron_saturation_powered_a)}
.param iesat2={_fmt(plasma.electron_saturation_grounded_a)}
.func spos(x,d) {{(x > 0) ? 0.5*(x+sqrt(x*x+d*d)) : 0.5*d*d/(sqrt(x*x+d*d)-x)}}

{body}
{shunt_element}* Python-inserted sensor: positive current flows from Qucs-S into plasma.
Vsense_surface_wafer {interface_node} plasma 0

* Powered wafer sheath.
Csh1 plasma bulk1 C='cscale*sqrt(ksh1/sqrt(v(bulk1,plasma)*v(bulk1,plasma)+vcap*vcap))'
Belectron1 plasma bulk1 I='iesat1*exp(-spos(v(bulk1,plasma),veps)/te_ev)'
Iion1 bulk1 plasma DC {_fmt(plasma.ion_current_powered_a)}

* Quasineutral bulk electron inertia and collisions.
Vsense_bulk bulk1 bulk_sense 0
Lplasma bulk_sense bulk_mid {_fmt(plasma.bulk_inductance_h)}
Rplasma bulk_mid bulk2 {_fmt(plasma.bulk_resistance_ohm)}

* Grounded chamber sheath.
Csh2 bulk2 0 C='cscale*sqrt(ksh2/sqrt(v(bulk2)*v(bulk2)+vcap*vcap))'
Belectron2 0 bulk2 I='iesat2*exp(-spos(v(bulk2),veps)/te_ev)'
Iion2 bulk2 0 DC {_fmt(plasma.ion_current_grounded_a)}

.ic v(plasma)=0 v(bulk1)={_fmt(eq_v)} v(bulk2)={_fmt(eq_v)} v(bulk_mid)={_fmt(eq_v)} v(bulk_sense)={_fmt(eq_v)}
.options method=gear maxord=2 reltol={_fmt(float(transient['reltol']))} abstol={_fmt(float(transient['abstol']))} vntol={_fmt(float(transient['vntol']))}

.control
set noaskquit
set wr_singlescale
set wr_vecnames
option numdgt=15
tran {_fmt(step)} {_fmt(total_time)} {_fmt(saved_start)} {_fmt(step)} uic
wrdata waveforms.dat {vectors}
quit
.endc
.end
"""


def _parse_waveforms(
    path: Path, vector_specs: tuple[tuple[str, str], ...]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"Qucs one-zone waveform output is empty: {path}")
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    expected_columns = 1 + len(vector_specs)
    if data.shape[1] != expected_columns:
        raise RuntimeError(
            f"unexpected Qucs one-zone wrdata columns {data.shape[1]}; "
            f"expected {expected_columns}"
        )
    return data[:, 0], {
        name: data[:, index + 1] for index, (name, _) in enumerate(vector_specs)
    }


def analyze_qucs_one_zone_waveforms(
    config: Mapping[str, Any], time_s: np.ndarray, waveforms: Mapping[str, np.ndarray]
) -> SimulationMetrics:
    """Calculate observables using the Qucs-S source and the Python plasma boundary."""
    frequency = float(config["frequency_hz"])
    analysis_cycles = min(20, int(config["transient"]["saved_cycles"]) - 2)
    mask = _window(time_s, analysis_cycles, frequency)
    t = time_s[mask]
    w = {name: values[mask] for name, values in waveforms.items()}
    duration = t[-1] - t[0]

    def mean(values: np.ndarray) -> float:
        return float(np.trapezoid(values, t) / duration)

    # The zero-volt source points from the ramped generator into the Qucs-S circuit.
    i_generator_wave = w["i_source_device"]
    i_plasma_wave = w["i_plasma_port"]
    p_abs = mean(w["v_plasma"] * i_plasma_wave)
    p_source = mean(w["v_source"] * i_generator_wave)
    series_resistance = float(config["qucs_netlist"]["series_loss_resistance_ohm"])
    current_squared_mean = mean(i_generator_wave**2)
    p_series = current_squared_mean * series_resistance
    p_inductor = current_squared_mean * series_inductor_esr_ohm(config)
    accounted_power = p_series + p_inductor + p_abs
    power_error = abs(p_source - accounted_power) / max(abs(p_source), 1.0e-30)
    source_phasor = _phasor(t, w["v_source"], frequency)
    generator_phasor = _phasor(t, i_generator_wave, frequency)
    feed_phasor = _phasor(t, w["v_feed"], frequency)
    plasma_phasor = _phasor(t, i_plasma_wave, frequency)
    z_input = source_phasor / generator_phasor
    z_load = feed_phasor / plasma_phasor
    plasma_harmonics = tuple(
        abs(_phasor(t, i_plasma_wave, harmonic * frequency))
        for harmonic in range(1, 13)
    )
    generator_harmonics = tuple(
        abs(_phasor(t, i_generator_wave, harmonic * frequency))
        for harmonic in range(1, 13)
    )
    thd = float(
        np.sqrt(np.sum(np.square(plasma_harmonics[1:])))
        / max(plasma_harmonics[0], 1.0e-30)
    )
    return SimulationMetrics(
        absorbed_power_w=float(p_abs),
        source_delivered_power_w=float(p_source),
        source_resistor_loss_w=float(p_series),
        match_loss_w=float(p_inductor),
        stray_loss_w=0.0,
        mean_sheath_powered_v=mean(w["v_sheath_powered"]),
        mean_sheath_grounded_v=mean(w["v_sheath_grounded"]),
        input_impedance_real_ohm=float(z_input.real),
        input_impedance_imag_ohm=float(z_input.imag),
        load_impedance_real_ohm=float(z_load.real),
        load_impedance_imag_ohm=float(z_load.imag),
        generator_current_amplitude_a=abs(generator_phasor),
        plasma_current_amplitude_a=plasma_harmonics[0],
        load_current_amplitude_a=abs(plasma_phasor),
        stray_current_amplitude_a=0.0,
        plasma_voltage_amplitude_v=abs(_phasor(t, w["v_plasma"], frequency)),
        plasma_voltage_offset_v=mean(w["v_plasma"]),
        plasma_current_thd=thd,
        cycle_l2_voltage=_cycle_l2(time_s, waveforms["v_plasma"], frequency),
        cycle_l2_current=_cycle_l2(time_s, waveforms["i_plasma_port"], frequency),
        power_balance_relative_error=float(power_error),
        harmonic_amplitudes_plasma_a=plasma_harmonics,
        harmonic_amplitudes_generator_a=generator_harmonics,
    )


def run_qucs_one_zone_ngspice(
    config: Mapping[str, Any], plasma: PlasmaParameters, case_directory: Path
) -> SimulationResult:
    """Normalize one Qucs-S RLC export, attach plasma, run ngspice, and analyze it."""
    case_directory.mkdir(parents=True, exist_ok=True)
    source_path = Path(config["qucs_netlist"]["resolved_path"])
    raw_netlist = source_path.read_text(encoding="utf-8", errors="replace")
    normalized = _prepare_external_body(raw_netlist, config)
    case_text = render_qucs_one_zone_netlist(config, plasma, raw_netlist)
    (case_directory / "qucs_source.cir").write_text(raw_netlist, encoding="utf-8")
    (case_directory / "external_normalized.inc").write_text(
        normalized, encoding="ascii", newline="\n"
    )
    (case_directory / "case.cir").write_text(case_text, encoding="ascii", newline="\n")
    command = [str(config["ngspice_path"]), "-n", "-o", "ngspice.log", "case.cir"]
    completed = subprocess.run(
        command,
        cwd=case_directory,
        capture_output=True,
        text=True,
        timeout=float(config["transient"]["timeout_s"]),
        check=False,
    )
    (case_directory / "process.json").write_text(
        json.dumps(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "qucs_source_path": str(source_path),
                "qucs_source_sha256": hashlib.sha256(raw_netlist.encode("utf-8")).hexdigest(),
                "plasma": asdict(plasma),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log_path = case_directory / "ngspice.log"
    if completed.returncode != 0:
        log = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"ngspice Qucs one-zone case failed ({completed.returncode}):\n{log[-4000:]}"
        )
    waveform_path = case_directory / "waveforms.dat"
    if not waveform_path.is_file():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"ngspice Qucs one-zone case produced no waveforms:\n{log[-4000:]}")
    specs = _vector_specs(config)
    time_s, waveforms = _parse_waveforms(waveform_path, specs)
    metrics = analyze_qucs_one_zone_waveforms(config, time_s, waveforms)
    return SimulationResult(metrics, time_s, waveforms, case_directory)


def solve_qucs_one_zone_coupled(
    config: Mapping[str, Any], output_directory: Path
) -> QucsOneZoneCoupledResult:
    """Iterate Qucs-S circuit absorption and one-zone global power balance."""
    coupling = config["coupling"]
    density = float(coupling["initial_density_m3"])
    relaxation = float(coupling["relaxation"])
    tolerance = float(coupling["relative_tolerance"])
    cycle_tolerance = float(coupling["cycle_l2_tolerance"])
    required_consecutive = int(coupling.get("consecutive_converged", 1))
    maximum_iterations = int(coupling["max_iterations"])
    temperature = electron_temperature_from_particle_balance(config)
    lower_log_density: float | None = None
    upper_log_density: float | None = None
    consecutive = 0
    history: list[QucsDensityIteration] = []
    final_simulation: SimulationResult | None = None
    final_target = float("nan")
    final_residual = float("inf")

    for iteration in range(maximum_iterations):
        plasma = compute_plasma_parameters(config, density, temperature)
        simulation = run_qucs_one_zone_ngspice(
            config, plasma, output_directory / f"iteration_{iteration:02d}"
        )
        metrics = simulation.metrics
        target = density_from_power_balance(
            config,
            temperature,
            metrics.absorbed_power_w,
            metrics.mean_sheath_powered_v,
            metrics.mean_sheath_grounded_v,
        )
        log_density = float(np.log(density))
        log_residual = float(np.log(target) - log_density)
        relative_residual = abs(float(np.expm1(log_residual)))
        if log_residual > 0.0:
            lower_log_density = log_density
        else:
            upper_log_density = log_density
        if lower_log_density is not None and upper_log_density is not None:
            next_log_density = 0.5 * (lower_log_density + upper_log_density)
        else:
            next_log_density = log_density + relaxation * log_residual
        density_next = float(np.exp(next_log_density))
        history.append(
            QucsDensityIteration(
                iteration=iteration,
                density_input_m3=density,
                density_target_m3=target,
                density_output_m3=density_next,
                balance_relative_residual=relative_residual,
                absorbed_power_w=metrics.absorbed_power_w,
                mean_sheath_powered_v=metrics.mean_sheath_powered_v,
                mean_sheath_grounded_v=metrics.mean_sheath_grounded_v,
                input_impedance_real_ohm=metrics.input_impedance_real_ohm,
                input_impedance_imag_ohm=metrics.input_impedance_imag_ohm,
                cycle_l2_voltage=metrics.cycle_l2_voltage,
                cycle_l2_current=metrics.cycle_l2_current,
            )
        )
        final_simulation = simulation
        final_target = target
        final_residual = relative_residual
        rf_converged = max(metrics.cycle_l2_voltage, metrics.cycle_l2_current) <= cycle_tolerance
        fixed_point_ok = relative_residual <= tolerance and rf_converged
        consecutive = consecutive + 1 if fixed_point_ok else 0
        if consecutive >= required_consecutive:
            return QucsOneZoneCoupledResult(
                electron_temperature_ev=temperature,
                electron_density_m3=density,
                converged=True,
                history=tuple(history),
                final_simulation=simulation,
                final_density_target_m3=target,
                final_balance_relative_residual=relative_residual,
            )
        density = density_next

    if final_simulation is None:
        raise RuntimeError("Qucs one-zone solver performed no iterations")
    return QucsOneZoneCoupledResult(
        electron_temperature_ev=temperature,
        electron_density_m3=history[-1].density_input_m3,
        converged=False,
        history=tuple(history),
        final_simulation=final_simulation,
        final_density_target_m3=final_target,
        final_balance_relative_residual=final_residual,
    )


def qucs_one_zone_result_to_dict(result: QucsOneZoneCoupledResult) -> dict[str, Any]:
    return {
        "converged": result.converged,
        "electron_temperature_ev": result.electron_temperature_ev,
        "electron_density_m3": result.electron_density_m3,
        "final_density_target_m3": result.final_density_target_m3,
        "final_balance_relative_residual": result.final_balance_relative_residual,
        "history": [asdict(row) for row in result.history],
        "metrics": asdict(result.final_simulation.metrics),
    }
