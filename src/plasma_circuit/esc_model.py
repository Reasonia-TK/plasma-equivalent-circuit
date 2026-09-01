"""Two-surface electrostatic-chuck plasma equivalent-circuit model."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from plasma_circuit.ngspice import _cycle_l2, _phasor, _window
from plasma_circuit.physics import (
    ATOMIC_MASS_UNIT_KG,
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    VACUUM_PERMITTIVITY_F_M,
    argon_rate_coefficients,
    density_from_power_balance_surfaces,
    electron_temperature_from_particle_balance_geometry,
    neutral_density,
)


SURFACE_NAMES = ("wafer", "focus_ring")
VECTOR_SPECS = (
    ("v_source_wafer", "v(w_src)"),
    ("v_source_focus", "v(f_src)"),
    ("v_input_wafer", "v(w_series_in)"),
    ("v_input_focus", "v(f_series_in)"),
    ("v_electrode_wafer", "v(w_electrode)"),
    ("v_electrode_focus", "v(f_electrode)"),
    ("v_surface_wafer", "v(wafer)"),
    ("v_surface_focus", "v(focus)"),
    ("v_sheath_wafer", "v(w_bulk,wafer)"),
    ("v_sheath_focus", "v(f_bulk,focus)"),
    ("v_sheath_ground", "v(plasma_bulk)"),
    ("v_dielectric_wafer", "v(w_electrode,w_feed)"),
    ("v_dielectric_focus", "v(f_electrode,f_feed)"),
    ("i_source_wafer", "i(Vsense_generator_wafer)"),
    ("i_source_focus", "i(Vsense_generator_focus)"),
    ("i_dielectric_wafer", "i(Vsense_dielectric_wafer)"),
    ("i_dielectric_focus", "i(Vsense_dielectric_focus)"),
    ("i_surface_wafer", "i(Vsense_surface_wafer)"),
    ("i_surface_focus", "i(Vsense_surface_focus)"),
    ("i_bulk_wafer", "i(Vsense_bulk_wafer)"),
    ("i_bulk_focus", "i(Vsense_bulk_focus)"),
)


@dataclass(frozen=True)
class EscSurfaceParameters:
    """Density-dependent parameters for one plasma-facing surface."""

    area_m2: float
    ion_current_a: float
    electron_saturation_current_a: float
    sheath_k: float
    bulk_inductance_h: float
    bulk_resistance_ohm: float


@dataclass(frozen=True)
class EscPlasmaParameters:
    """Common plasma state and the three surface branches."""

    electron_temperature_ev: float
    electron_density_m3: float
    neutral_density_m3: float
    mean_electron_speed_m_s: float
    bohm_speed_m_s: float
    momentum_collision_frequency_hz: float
    effective_collision_frequency_hz: float
    equilibrium_sheath_voltage_v: float
    wafer: EscSurfaceParameters
    focus_ring: EscSurfaceParameters
    ground: EscSurfaceParameters


@dataclass(frozen=True)
class EscMetrics:
    """Electrical observables for one frozen-density ESC circuit simulation."""

    absorbed_power_total_w: float
    absorbed_power_wafer_port_w: float
    absorbed_power_focus_port_w: float
    source_delivered_total_w: float
    source_delivered_wafer_w: float
    source_delivered_focus_w: float
    source_resistor_loss_total_w: float
    source_resistor_loss_wafer_w: float
    source_resistor_loss_focus_w: float
    dielectric_loss_total_w: float
    dielectric_loss_wafer_w: float
    dielectric_loss_focus_w: float
    mean_sheath_wafer_v: float
    mean_sheath_focus_v: float
    mean_sheath_ground_v: float
    input_impedance_wafer_real_ohm: float
    input_impedance_wafer_imag_ohm: float
    input_impedance_focus_real_ohm: float
    input_impedance_focus_imag_ohm: float
    electrode_voltage_wafer_amplitude_v: float
    electrode_voltage_focus_amplitude_v: float
    surface_voltage_wafer_amplitude_v: float
    surface_voltage_focus_amplitude_v: float
    surface_voltage_wafer_offset_v: float
    surface_voltage_focus_offset_v: float
    source_current_wafer_amplitude_a: float
    source_current_focus_amplitude_a: float
    surface_current_wafer_amplitude_a: float
    surface_current_focus_amplitude_a: float
    surface_current_wafer_thd: float
    surface_current_focus_thd: float
    cycle_l2_voltage_wafer: float
    cycle_l2_voltage_focus: float
    cycle_l2_current_wafer: float
    cycle_l2_current_focus: float
    power_balance_relative_error: float
    harmonic_amplitudes_wafer_a: tuple[float, ...]
    harmonic_amplitudes_focus_a: tuple[float, ...]


@dataclass(frozen=True)
class EscSimulationResult:
    metrics: EscMetrics
    time_s: np.ndarray
    waveforms: Mapping[str, np.ndarray]
    case_directory: Path


@dataclass(frozen=True)
class EscDensityIteration:
    iteration: int
    density_input_m3: float
    density_target_m3: float
    density_output_m3: float
    relative_change: float
    absorbed_power_w: float
    cycle_l2_max: float


@dataclass(frozen=True)
class EscCoupledResult:
    electron_temperature_ev: float
    electron_density_m3: float
    converged: bool
    history: tuple[EscDensityIteration, ...]
    final_simulation: EscSimulationResult


def _fmt(value: float) -> str:
    return f"{value:.16e}"


def load_esc_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the two-surface ESC model configuration."""
    config_path = Path(path)
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "ngspice_path",
        "frequency_hz",
        "pressure_pa",
        "gas_temperature_k",
        "plasma_volume_m3",
        "grounded_area_m2",
        "bulk_path_length_m",
        "surfaces",
        "regularization",
        "transient",
        "coupling",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing ESC configuration fields: {missing}")
    if set(config["surfaces"]) != set(SURFACE_NAMES):
        raise ValueError(f"ESC surfaces must be exactly {SURFACE_NAMES}")
    positive_top_level = (
        "frequency_hz",
        "pressure_pa",
        "gas_temperature_k",
        "plasma_volume_m3",
        "grounded_area_m2",
        "bulk_path_length_m",
    )
    if any(float(config[field]) <= 0.0 for field in positive_top_level):
        raise ValueError("ESC frequency, gas, and geometry values must be positive")
    if float(config.get("source_ramp_cycles", 10.0)) <= 0.0:
        raise ValueError("ESC source ramp cycles must be positive")
    for name in SURFACE_NAMES:
        surface = config["surfaces"][name]
        surface_required = {
            "area_m2",
            "source_amplitude_v",
            "source_phase_deg",
            "source_resistance_ohm",
            "series_inductance_h",
            "coupling_capacitance_f",
            "dielectric_loss_tangent",
            "leakage_resistance_ohm",
        }
        surface_missing = sorted(surface_required - surface.keys())
        if surface_missing:
            raise ValueError(f"Missing {name} fields: {surface_missing}")
        positive_fields = (
            "area_m2",
            "source_resistance_ohm",
            "coupling_capacitance_f",
            "leakage_resistance_ohm",
        )
        if any(float(surface[field]) <= 0.0 for field in positive_fields):
            raise ValueError(f"{name} positive-valued fields must be positive")
        if float(surface["series_inductance_h"]) < 0.0:
            raise ValueError(f"{name} series inductance cannot be negative")
        if float(surface["dielectric_loss_tangent"]) < 0.0:
            raise ValueError(f"{name} dielectric loss tangent cannot be negative")
    if not Path(config["ngspice_path"]).is_file():
        raise FileNotFoundError(f"ngspice executable not found: {config['ngspice_path']}")
    return config


def _surface_parameters(
    area_m2: float,
    bulk_path_length_m: float,
    electron_density_m3: float,
    mean_electron_speed_m_s: float,
    bohm_speed_m_s: float,
    effective_collision_frequency_hz: float,
) -> EscSurfaceParameters:
    current_scale = ELEMENTARY_CHARGE_C * electron_density_m3 * area_m2
    if bulk_path_length_m > 0.0:
        inductance = (
            bulk_path_length_m
            * ELECTRON_MASS_KG
            / (ELEMENTARY_CHARGE_C**2 * electron_density_m3 * area_m2)
        )
        resistance = effective_collision_frequency_hz * inductance
    else:
        inductance = 0.0
        resistance = 0.0
    return EscSurfaceParameters(
        area_m2=area_m2,
        ion_current_a=float(current_scale * bohm_speed_m_s),
        electron_saturation_current_a=float(current_scale * mean_electron_speed_m_s),
        sheath_k=float(
            2.0
            * ELEMENTARY_CHARGE_C
            * electron_density_m3
            * VACUUM_PERMITTIVITY_F_M
            * area_m2**2
        ),
        bulk_inductance_h=float(inductance),
        bulk_resistance_ohm=float(resistance),
    )


def compute_esc_plasma_parameters(
    config: Mapping[str, Any],
    electron_density_m3: float,
    electron_temperature_ev: float,
) -> EscPlasmaParameters:
    """Calculate density-dependent branches for wafer, focus ring, and ground."""
    if electron_density_m3 <= 0.0 or not np.isfinite(electron_density_m3):
        raise ValueError("electron density must be finite and positive")
    argon_mass = float(config.get("argon_mass_amu", 39.948)) * ATOMIC_MASS_UNIT_KG
    n_gas = neutral_density(float(config["pressure_pa"]), float(config["gas_temperature_k"]))
    rates = argon_rate_coefficients(electron_temperature_ev)
    mean_speed = float(
        np.sqrt(
            8.0
            * ELEMENTARY_CHARGE_C
            * electron_temperature_ev
            / (np.pi * ELECTRON_MASS_KG)
        )
    )
    bohm_speed = float(np.sqrt(ELEMENTARY_CHARGE_C * electron_temperature_ev / argon_mass))
    momentum_frequency = float(n_gas * rates["elastic"])
    bulk_length = float(config["bulk_path_length_m"])
    effective_frequency = momentum_frequency + mean_speed / bulk_length
    equilibrium_voltage = float(electron_temperature_ev * np.log(mean_speed / bohm_speed))

    def branch(name: str, path_length_m: float = bulk_length) -> EscSurfaceParameters:
        area = (
            float(config["grounded_area_m2"])
            if name == "ground"
            else float(config["surfaces"][name]["area_m2"])
        )
        return _surface_parameters(
            area,
            path_length_m,
            electron_density_m3,
            mean_speed,
            bohm_speed,
            effective_frequency,
        )

    return EscPlasmaParameters(
        electron_temperature_ev=electron_temperature_ev,
        electron_density_m3=electron_density_m3,
        neutral_density_m3=n_gas,
        mean_electron_speed_m_s=mean_speed,
        bohm_speed_m_s=bohm_speed,
        momentum_collision_frequency_hz=momentum_frequency,
        effective_collision_frequency_hz=float(effective_frequency),
        equilibrium_sheath_voltage_v=equilibrium_voltage,
        wafer=branch("wafer"),
        focus_ring=branch("focus_ring"),
        ground=branch("ground", path_length_m=0.0),
    )


def _source_and_dielectric_lines(
    label: str,
    prefix: str,
    surface_node: str,
    surface: Mapping[str, Any],
    frequency_hz: float,
    ramp_cycles: float,
) -> str:
    capacitance = float(surface["coupling_capacitance_f"])
    loss_tangent = float(surface["dielectric_loss_tangent"])
    esr = max(loss_tangent / (2.0 * np.pi * frequency_hz * capacitance), 1.0e-9)
    inductance = float(surface["series_inductance_h"])
    series_element = (
        f"Lseries_{label} {prefix}_series_in {prefix}_electrode {_fmt(inductance)}"
        if inductance > 0.0
        else f"Vwire_{label} {prefix}_series_in {prefix}_electrode 0"
    )
    ramp_time_s = ramp_cycles / frequency_hz
    phase_rad = np.deg2rad(float(surface["source_phase_deg"]))
    return f"""Bsource_{label} {prefix}_src 0 V='{_fmt(float(surface['source_amplitude_v']))}*sin(2*pi*freq*time+{_fmt(float(phase_rad))})*tanh(time/{_fmt(ramp_time_s)})'
Rsource_{label} {prefix}_src {prefix}_generator_sense {_fmt(float(surface['source_resistance_ohm']))}
Vsense_generator_{label} {prefix}_generator_sense {prefix}_series_in 0
{series_element}
Cesc_{label} {prefix}_electrode {prefix}_cap {_fmt(capacitance)}
Vsense_dielectric_{label} {prefix}_cap {prefix}_loss 0
Resr_{label} {prefix}_loss {prefix}_feed {_fmt(esr)}
Rleak_{label} {prefix}_electrode {prefix}_feed {_fmt(float(surface['leakage_resistance_ohm']))}
Vsense_surface_{label} {prefix}_feed {surface_node} 0"""


def render_esc_netlist(
    config: Mapping[str, Any], plasma: EscPlasmaParameters
) -> str:
    """Render the dual capacitively coupled wafer/focus-ring circuit."""
    transient = config["transient"]
    regularization = config["regularization"]
    frequency = float(config["frequency_hz"])
    period = 1.0 / frequency
    total_time = int(transient["cycles"]) * period
    saved_start = total_time - int(transient["saved_cycles"]) * period
    step = period / int(transient["samples_per_cycle"])
    vectors = " ".join(spec for _, spec in VECTOR_SPECS)
    equilibrium = plasma.equilibrium_sheath_voltage_v
    wafer_external = _source_and_dielectric_lines(
        "wafer",
        "w",
        "wafer",
        config["surfaces"]["wafer"],
        frequency,
        float(config.get("source_ramp_cycles", 10.0)),
    )
    focus_external = _source_and_dielectric_lines(
        "focus",
        "f",
        "focus",
        config["surfaces"]["focus_ring"],
        frequency,
        float(config.get("source_ramp_cycles", 10.0)),
    )

    return f"""Dual-surface electrostatic chuck plasma equivalent circuit
* Wafer and focus-ring surfaces couple to independent buried electrodes.
.param freq={_fmt(frequency)}
.param te_ev={_fmt(plasma.electron_temperature_ev)}
.param veps={_fmt(float(regularization['electron_voltage_v']))}
.param vcap={_fmt(float(regularization['capacitance_voltage_v']))}
.param cscale={_fmt(float(regularization['capacitance_scale']))}
.param kshw={_fmt(plasma.wafer.sheath_k)}
.param kshf={_fmt(plasma.focus_ring.sheath_k)}
.param kshg={_fmt(plasma.ground.sheath_k)}
.param iesatw={_fmt(plasma.wafer.electron_saturation_current_a)}
.param iesatf={_fmt(plasma.focus_ring.electron_saturation_current_a)}
.param iesatg={_fmt(plasma.ground.electron_saturation_current_a)}
.func spos(x,d) {{(x > 0) ? 0.5*(x+sqrt(x*x+d*d)) : 0.5*d*d/(sqrt(x*x+d*d)-x)}}

* External electrode and dielectric coupling branches.
{wafer_external}
{focus_external}

* Wafer-facing sheath and its share of the plasma bulk.
Csh_wafer wafer w_bulk C='cscale*sqrt(kshw/sqrt(v(w_bulk,wafer)*v(w_bulk,wafer)+vcap*vcap))'
Belectron_wafer wafer w_bulk I='iesatw*exp(-spos(v(w_bulk,wafer),veps)/te_ev)'
Iion_wafer w_bulk wafer DC {_fmt(plasma.wafer.ion_current_a)}
Vsense_bulk_wafer w_bulk w_bulk_sense 0
Lbulk_wafer w_bulk_sense w_bulk_mid {_fmt(plasma.wafer.bulk_inductance_h)}
Rbulk_wafer w_bulk_mid plasma_bulk {_fmt(plasma.wafer.bulk_resistance_ohm)}

* Focus-ring-facing sheath and bulk branch.
Csh_focus focus f_bulk C='cscale*sqrt(kshf/sqrt(v(f_bulk,focus)*v(f_bulk,focus)+vcap*vcap))'
Belectron_focus focus f_bulk I='iesatf*exp(-spos(v(f_bulk,focus),veps)/te_ev)'
Iion_focus f_bulk focus DC {_fmt(plasma.focus_ring.ion_current_a)}
Vsense_bulk_focus f_bulk f_bulk_sense 0
Lbulk_focus f_bulk_sense f_bulk_mid {_fmt(plasma.focus_ring.bulk_inductance_h)}
Rbulk_focus f_bulk_mid plasma_bulk {_fmt(plasma.focus_ring.bulk_resistance_ohm)}

* Shared grounded chamber sheath.
Csh_ground plasma_bulk 0 C='cscale*sqrt(kshg/sqrt(v(plasma_bulk)*v(plasma_bulk)+vcap*vcap))'
Belectron_ground 0 plasma_bulk I='iesatg*exp(-spos(v(plasma_bulk),veps)/te_ev)'
Iion_ground plasma_bulk 0 DC {_fmt(plasma.ground.ion_current_a)}

.ic v(wafer)=0 v(focus)=0 v(w_bulk)={_fmt(equilibrium)} v(f_bulk)={_fmt(equilibrium)} v(plasma_bulk)={_fmt(equilibrium)} v(w_bulk_sense)={_fmt(equilibrium)} v(w_bulk_mid)={_fmt(equilibrium)} v(f_bulk_sense)={_fmt(equilibrium)} v(f_bulk_mid)={_fmt(equilibrium)}
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


def _parse_waveforms(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"ngspice ESC waveform output is empty: {path}")
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    expected_columns = 1 + len(VECTOR_SPECS)
    if data.shape[1] != expected_columns:
        raise RuntimeError(
            f"unexpected ESC wrdata columns {data.shape[1]}; expected {expected_columns}"
        )
    time_s = data[:, 0]
    waveforms = {
        name: data[:, index + 1]
        for index, (name, _) in enumerate(VECTOR_SPECS)
    }
    return time_s, waveforms


def _dielectric_esr(surface: Mapping[str, Any], frequency_hz: float) -> float:
    capacitance = float(surface["coupling_capacitance_f"])
    return max(
        float(surface["dielectric_loss_tangent"])
        / (2.0 * np.pi * frequency_hz * capacitance),
        1.0e-9,
    )


def analyze_esc_waveforms(
    config: Mapping[str, Any],
    time_s: np.ndarray,
    waveforms: Mapping[str, np.ndarray],
) -> EscMetrics:
    """Calculate two-port powers, impedances, biases, harmonics, and residuals."""
    frequency = float(config["frequency_hz"])
    analysis_cycles = min(20, int(config["transient"]["saved_cycles"]) - 2)
    mask = _window(time_s, analysis_cycles, frequency)
    t = time_s[mask]
    w = {name: values[mask] for name, values in waveforms.items()}
    duration = t[-1] - t[0]

    def mean(values: np.ndarray) -> float:
        return float(np.trapezoid(values, t) / duration)

    p_wafer = mean(w["v_surface_wafer"] * w["i_surface_wafer"])
    p_focus = mean(w["v_surface_focus"] * w["i_surface_focus"])
    p_absorbed = p_wafer + p_focus
    p_source_wafer = mean(w["v_source_wafer"] * w["i_source_wafer"])
    p_source_focus = mean(w["v_source_focus"] * w["i_source_focus"])
    wafer_config = config["surfaces"]["wafer"]
    focus_config = config["surfaces"]["focus_ring"]
    p_source_resistor_wafer = mean(w["i_source_wafer"] ** 2) * float(
        wafer_config["source_resistance_ohm"]
    )
    p_source_resistor_focus = mean(w["i_source_focus"] ** 2) * float(
        focus_config["source_resistance_ohm"]
    )
    p_dielectric_wafer = mean(w["i_dielectric_wafer"] ** 2) * _dielectric_esr(
        wafer_config, frequency
    ) + mean(w["v_dielectric_wafer"] ** 2) / float(
        wafer_config["leakage_resistance_ohm"]
    )
    p_dielectric_focus = mean(w["i_dielectric_focus"] ** 2) * _dielectric_esr(
        focus_config, frequency
    ) + mean(w["v_dielectric_focus"] ** 2) / float(
        focus_config["leakage_resistance_ohm"]
    )
    source_total = p_source_wafer + p_source_focus
    accounted = (
        p_source_resistor_wafer
        + p_source_resistor_focus
        + p_dielectric_wafer
        + p_dielectric_focus
        + p_absorbed
    )
    power_error = abs(source_total - accounted) / max(abs(source_total), 1.0e-30)

    input_voltage_wafer = _phasor(t, w["v_input_wafer"], frequency)
    input_voltage_focus = _phasor(t, w["v_input_focus"], frequency)
    source_current_wafer = _phasor(t, w["i_source_wafer"], frequency)
    source_current_focus = _phasor(t, w["i_source_focus"], frequency)
    impedance_wafer = input_voltage_wafer / source_current_wafer
    impedance_focus = input_voltage_focus / source_current_focus
    harmonics_wafer = tuple(
        abs(_phasor(t, w["i_surface_wafer"], harmonic * frequency))
        for harmonic in range(1, 13)
    )
    harmonics_focus = tuple(
        abs(_phasor(t, w["i_surface_focus"], harmonic * frequency))
        for harmonic in range(1, 13)
    )
    thd_wafer = float(
        np.sqrt(np.sum(np.square(harmonics_wafer[1:])))
        / max(harmonics_wafer[0], 1.0e-30)
    )
    thd_focus = float(
        np.sqrt(np.sum(np.square(harmonics_focus[1:])))
        / max(harmonics_focus[0], 1.0e-30)
    )

    return EscMetrics(
        absorbed_power_total_w=float(p_absorbed),
        absorbed_power_wafer_port_w=float(p_wafer),
        absorbed_power_focus_port_w=float(p_focus),
        source_delivered_total_w=float(source_total),
        source_delivered_wafer_w=float(p_source_wafer),
        source_delivered_focus_w=float(p_source_focus),
        source_resistor_loss_total_w=float(
            p_source_resistor_wafer + p_source_resistor_focus
        ),
        source_resistor_loss_wafer_w=float(p_source_resistor_wafer),
        source_resistor_loss_focus_w=float(p_source_resistor_focus),
        dielectric_loss_total_w=float(p_dielectric_wafer + p_dielectric_focus),
        dielectric_loss_wafer_w=float(p_dielectric_wafer),
        dielectric_loss_focus_w=float(p_dielectric_focus),
        mean_sheath_wafer_v=mean(w["v_sheath_wafer"]),
        mean_sheath_focus_v=mean(w["v_sheath_focus"]),
        mean_sheath_ground_v=mean(w["v_sheath_ground"]),
        input_impedance_wafer_real_ohm=float(impedance_wafer.real),
        input_impedance_wafer_imag_ohm=float(impedance_wafer.imag),
        input_impedance_focus_real_ohm=float(impedance_focus.real),
        input_impedance_focus_imag_ohm=float(impedance_focus.imag),
        electrode_voltage_wafer_amplitude_v=abs(
            _phasor(t, w["v_electrode_wafer"], frequency)
        ),
        electrode_voltage_focus_amplitude_v=abs(
            _phasor(t, w["v_electrode_focus"], frequency)
        ),
        surface_voltage_wafer_amplitude_v=abs(
            _phasor(t, w["v_surface_wafer"], frequency)
        ),
        surface_voltage_focus_amplitude_v=abs(
            _phasor(t, w["v_surface_focus"], frequency)
        ),
        surface_voltage_wafer_offset_v=mean(w["v_surface_wafer"]),
        surface_voltage_focus_offset_v=mean(w["v_surface_focus"]),
        source_current_wafer_amplitude_a=abs(source_current_wafer),
        source_current_focus_amplitude_a=abs(source_current_focus),
        surface_current_wafer_amplitude_a=harmonics_wafer[0],
        surface_current_focus_amplitude_a=harmonics_focus[0],
        surface_current_wafer_thd=thd_wafer,
        surface_current_focus_thd=thd_focus,
        cycle_l2_voltage_wafer=_cycle_l2(
            time_s, waveforms["v_surface_wafer"], frequency
        ),
        cycle_l2_voltage_focus=_cycle_l2(
            time_s, waveforms["v_surface_focus"], frequency
        ),
        cycle_l2_current_wafer=_cycle_l2(
            time_s, waveforms["i_surface_wafer"], frequency
        ),
        cycle_l2_current_focus=_cycle_l2(
            time_s, waveforms["i_surface_focus"], frequency
        ),
        power_balance_relative_error=float(power_error),
        harmonic_amplitudes_wafer_a=harmonics_wafer,
        harmonic_amplitudes_focus_a=harmonics_focus,
    )


def run_esc_ngspice(
    config: Mapping[str, Any],
    plasma: EscPlasmaParameters,
    case_directory: Path,
) -> EscSimulationResult:
    """Run one frozen-density ESC netlist and analyze the saved RF cycles."""
    case_directory.mkdir(parents=True, exist_ok=True)
    netlist_path = case_directory / "case.cir"
    netlist_path.write_text(
        render_esc_netlist(config, plasma), encoding="ascii", newline="\n"
    )
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
        raise RuntimeError(f"ngspice ESC case failed ({completed.returncode}):\n{log[-4000:]}")
    waveform_path = case_directory / "waveforms.dat"
    if not waveform_path.is_file():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"ngspice ESC case produced no waveforms:\n{log[-4000:]}")
    time_s, waveforms = _parse_waveforms(waveform_path)
    metrics = analyze_esc_waveforms(config, time_s, waveforms)
    return EscSimulationResult(metrics, time_s, waveforms, case_directory)


def solve_esc_model(
    config: Mapping[str, Any],
    output_directory: Path,
    initial_density_m3: float | None = None,
) -> EscCoupledResult:
    """Couple the dual-surface circuit to particle and global power balances."""
    coupling = config["coupling"]
    density = float(
        coupling["initial_density_m3"]
        if initial_density_m3 is None
        else initial_density_m3
    )
    relaxation = float(coupling["relaxation"])
    tolerance = float(coupling["relative_tolerance"])
    required_consecutive = int(coupling.get("consecutive_converged", 1))
    cycle_tolerance = float(coupling.get("cycle_l2_tolerance", np.inf))
    max_iterations = int(coupling["max_iterations"])
    wafer_area = float(config["surfaces"]["wafer"]["area_m2"])
    focus_area = float(config["surfaces"]["focus_ring"]["area_m2"])
    ground_area = float(config["grounded_area_m2"])
    temperature = electron_temperature_from_particle_balance_geometry(
        config,
        plasma_volume_m3=float(config["plasma_volume_m3"]),
        loss_area_m2=wafer_area + focus_area + ground_area,
    )
    history: list[EscDensityIteration] = []
    final_simulation: EscSimulationResult | None = None
    lower_log_density: float | None = None
    upper_log_density: float | None = None
    consecutive_converged = 0

    for iteration in range(max_iterations):
        plasma = compute_esc_plasma_parameters(config, density, temperature)
        simulation = run_esc_ngspice(
            config,
            plasma,
            output_directory / f"density_{iteration:02d}",
        )
        metrics = simulation.metrics
        target = density_from_power_balance_surfaces(
            config,
            temperature,
            metrics.absorbed_power_total_w,
            surface_areas_m2=(wafer_area, focus_area, ground_area),
            mean_sheath_voltages_v=(
                metrics.mean_sheath_wafer_v,
                metrics.mean_sheath_focus_v,
                metrics.mean_sheath_ground_v,
            ),
            plasma_volume_m3=float(config["plasma_volume_m3"]),
        )
        log_density = float(np.log(density))
        log_residual = float(np.log(target) - log_density)
        if log_residual > 0.0:
            lower_log_density = log_density
        else:
            upper_log_density = log_density
        if lower_log_density is not None and upper_log_density is not None:
            next_log_density = 0.5 * (lower_log_density + upper_log_density)
        else:
            next_log_density = log_density + relaxation * log_residual
        density_next = float(np.exp(next_log_density))
        relative_change = abs(density_next - density) / density
        cycle_l2_max = max(
            metrics.cycle_l2_voltage_wafer,
            metrics.cycle_l2_voltage_focus,
            metrics.cycle_l2_current_wafer,
            metrics.cycle_l2_current_focus,
        )
        history.append(
            EscDensityIteration(
                iteration=iteration,
                density_input_m3=density,
                density_target_m3=target,
                density_output_m3=density_next,
                relative_change=relative_change,
                absorbed_power_w=metrics.absorbed_power_total_w,
                cycle_l2_max=cycle_l2_max,
            )
        )
        final_simulation = simulation
        density_converged = abs(np.expm1(log_residual)) < tolerance
        rf_converged = cycle_l2_max < cycle_tolerance
        consecutive_converged = (
            consecutive_converged + 1
            if density_converged and rf_converged
            else 0
        )
        if consecutive_converged >= required_consecutive:
            return EscCoupledResult(
                electron_temperature_ev=temperature,
                electron_density_m3=density,
                converged=True,
                history=tuple(history),
                final_simulation=simulation,
            )
        density = density_next

    assert final_simulation is not None
    return EscCoupledResult(
        electron_temperature_ev=temperature,
        electron_density_m3=history[-1].density_input_m3,
        converged=False,
        history=tuple(history),
        final_simulation=final_simulation,
    )


def esc_result_to_dict(result: EscCoupledResult) -> dict[str, Any]:
    """Convert the coupled result to a JSON-compatible dictionary."""
    return {
        "model": "dual-surface-electrostatic-chuck-global-model",
        "electron_temperature_ev": result.electron_temperature_ev,
        "electron_density_m3": result.electron_density_m3,
        "converged": result.converged,
        "history": [asdict(row) for row in result.history],
        "metrics": asdict(result.final_simulation.metrics),
    }
