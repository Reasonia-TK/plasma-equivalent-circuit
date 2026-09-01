"""ngspice netlist generation, execution, and waveform analysis."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from plasma_circuit.physics import PlasmaParameters


VECTOR_SPECS = (
    ("v_source", "v(src)"),
    ("v_input", "v(match)"),
    ("v_load", "v(load_sense)"),
    ("v_plasma", "v(plasma)"),
    ("v_sheath_powered", "v(bulk1,plasma)"),
    ("v_sheath_grounded", "v(bulk2)"),
    ("i_generator", "i(Vsense_generator)"),
    ("i_match_series", "i(Vsense_load)"),
    ("i_match_shunt", "i(Vsense_match_shunt)"),
    ("i_stray", "i(Vsense_stray)"),
    ("i_plasma_port", "i(Vsense_plasma)"),
    ("i_bulk", "i(Vsense_bulk)"),
)


@dataclass(frozen=True)
class SimulationMetrics:
    absorbed_power_w: float
    source_delivered_power_w: float
    source_resistor_loss_w: float
    match_loss_w: float
    stray_loss_w: float
    mean_sheath_powered_v: float
    mean_sheath_grounded_v: float
    input_impedance_real_ohm: float
    input_impedance_imag_ohm: float
    load_impedance_real_ohm: float
    load_impedance_imag_ohm: float
    generator_current_amplitude_a: float
    plasma_current_amplitude_a: float
    load_current_amplitude_a: float
    stray_current_amplitude_a: float
    plasma_voltage_amplitude_v: float
    plasma_voltage_offset_v: float
    plasma_current_thd: float
    cycle_l2_voltage: float
    cycle_l2_current: float
    power_balance_relative_error: float
    harmonic_amplitudes_plasma_a: tuple[float, ...]
    harmonic_amplitudes_generator_a: tuple[float, ...]


@dataclass(frozen=True)
class SimulationResult:
    metrics: SimulationMetrics
    time_s: np.ndarray
    waveforms: Mapping[str, np.ndarray]
    case_directory: Path


def _fmt(value: float) -> str:
    return f"{value:.16e}"


def render_netlist(
    config: Mapping[str, Any],
    plasma: PlasmaParameters,
    match_c1_f: float | None = None,
    match_c2_f: float | None = None,
) -> str:
    """Render the paper's Figure 1 circuit as an ngspice netlist."""
    transient = config["transient"]
    regularization = config["regularization"]
    frequency = float(config["frequency_hz"])
    period = 1.0 / frequency
    total_time = int(transient["cycles"]) * period
    saved_start = total_time - int(transient["saved_cycles"]) * period
    step = period / int(transient["samples_per_cycle"])
    c1 = float(config["match_c1_f"] if match_c1_f is None else match_c1_f)
    c2 = float(config["match_c2_f"] if match_c2_f is None else match_c2_f)
    vectors = " ".join(spec for _, spec in VECTOR_SPECS)
    eq_v = plasma.equilibrium_sheath_voltage_v
    capacitance_scale = float(regularization.get("capacitance_scale", 1.0))

    source_phase = float(config.get("source_phase_deg", 90.0))

    return f"""Schmidt et al. 2018 CCP and matching network reproduction
* Generated file. SI units are used throughout.
.param freq={_fmt(frequency)}
.param te_ev={_fmt(plasma.electron_temperature_ev)}
.param veps={_fmt(float(regularization['electron_voltage_v']))}
.param vcap={_fmt(float(regularization['capacitance_voltage_v']))}
.param cscale={_fmt(capacitance_scale)}
.param ksh1={_fmt(plasma.sheath_k_powered)}
.param ksh2={_fmt(plasma.sheath_k_grounded)}
.param iesat1={_fmt(plasma.electron_saturation_powered_a)}
.param iesat2={_fmt(plasma.electron_saturation_grounded_a)}
.func spos(x,d) {{(x > 0) ? 0.5*(x+sqrt(x*x+d*d)) : 0.5*d*d/(sqrt(x*x+d*d)-x)}}

* The paper specifies V0*cos(omega*t); ngspice SIN uses a phase in degrees.
Vrf src 0 SIN(0 {_fmt(float(config['source_amplitude_v']))} {{freq}} 0 0 {_fmt(source_phase)})
Rrf src generator_sense {_fmt(float(config['source_resistance_ohm']))}
Vsense_generator generator_sense match 0
Vsense_match_shunt match match_shunt 0
Cmatch1 match_shunt 0 {_fmt(c1)}
Cmatch2 match series_c {_fmt(c2)}
Lmatch series_c load_sense {_fmt(float(config['match_l_h']))}
Vsense_load load_sense match_loss_in 0
* Figure 1 places Rm before the plasma/stray split; it therefore carries I_L.
Rmatch match_loss_in load {_fmt(float(config['match_loss_ohm']))}
Vsense_plasma load plasma 0
Vsense_stray load stray_top 0
Cstray stray_top stray_mid {_fmt(float(config['stray_c_f']))}
Rstray stray_mid 0 {_fmt(float(config['stray_loss_ohm']))}

* Powered-electrode sheath: conventional electron and ion currents oppose.
Csh1 plasma bulk1 C='cscale*sqrt(ksh1/sqrt(v(bulk1,plasma)*v(bulk1,plasma)+vcap*vcap))'
Belectron1 plasma bulk1 I='iesat1*exp(-spos(v(bulk1,plasma),veps)/te_ev)'
Iion1 bulk1 plasma DC {_fmt(plasma.ion_current_powered_a)}

Vsense_bulk bulk1 bulk_sense 0
Lplasma bulk_sense bulk_mid {_fmt(plasma.bulk_inductance_h)}
Rplasma bulk_mid bulk2 {_fmt(plasma.bulk_resistance_ohm)}

* Grounded-electrode sheath. Vs2 is v(bulk2).
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


def _parse_wrdata(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"ngspice waveform output is empty: {path}")
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    expected_columns = 1 + len(VECTOR_SPECS)
    if data.shape[1] != expected_columns:
        header = lines[0]
        raise RuntimeError(
            f"unexpected wrdata column count {data.shape[1]} (expected {expected_columns}); header={header!r}"
        )
    time_s = data[:, 0]
    waveforms = {name: data[:, index + 1] for index, (name, _) in enumerate(VECTOR_SPECS)}
    return time_s, waveforms


def _window(time_s: np.ndarray, cycles: int, frequency_hz: float) -> np.ndarray:
    start = time_s[-1] - cycles / frequency_hz
    return time_s >= start - 1e-15


def _phasor(time_s: np.ndarray, values: np.ndarray, frequency_hz: float) -> complex:
    duration = time_s[-1] - time_s[0]
    if duration <= 0.0:
        raise ValueError("phasor window has no duration")
    kernel = np.exp(-2j * np.pi * frequency_hz * time_s)
    return complex(2.0 / duration * np.trapezoid(values * kernel, time_s))


def _cycle_l2(time_s: np.ndarray, values: np.ndarray, frequency_hz: float) -> float:
    period = 1.0 / frequency_hz
    end = time_s[-1]
    phase = np.linspace(0.0, period, 513)
    previous = np.interp(end - 2.0 * period + phase, time_s, values)
    current = np.interp(end - period + phase, time_s, values)
    denominator = max(float(np.linalg.norm(current)), 1e-30)
    return float(np.linalg.norm(current - previous) / denominator)


def analyze_waveforms(
    config: Mapping[str, Any], time_s: np.ndarray, waveforms: Mapping[str, np.ndarray]
) -> SimulationMetrics:
    """Calculate the paper's electrical observables from the saved RF cycles."""
    frequency = float(config["frequency_hz"])
    analysis_cycles = min(20, int(config["transient"]["saved_cycles"]) - 2)
    mask = _window(time_s, analysis_cycles, frequency)
    t = time_s[mask]
    w = {name: values[mask] for name, values in waveforms.items()}
    duration = t[-1] - t[0]

    def mean(values: np.ndarray) -> float:
        return float(np.trapezoid(values, t) / duration)

    p_abs = mean(w["v_plasma"] * w["i_plasma_port"])
    p_source = mean(w["v_source"] * w["i_generator"])
    p_rrf = mean(w["i_generator"] ** 2) * float(config["source_resistance_ohm"])
    p_match = mean(w["i_match_series"] ** 2) * float(config["match_loss_ohm"])
    p_stray = mean(w["i_stray"] ** 2) * float(config["stray_loss_ohm"])

    v_input = _phasor(t, w["v_input"], frequency)
    i_generator = _phasor(t, w["i_generator"], frequency)
    v_load = _phasor(t, w["v_load"], frequency)
    i_load = _phasor(t, w["i_match_series"], frequency)
    z_input = v_input / i_generator
    z_load = v_load / i_load

    plasma_harmonics = tuple(
        abs(_phasor(t, w["i_plasma_port"], harmonic * frequency))
        for harmonic in range(1, 13)
    )
    generator_harmonics = tuple(
        abs(_phasor(t, w["i_generator"], harmonic * frequency))
        for harmonic in range(1, 13)
    )
    thd = float(
        np.sqrt(np.sum(np.square(plasma_harmonics[1:]))) / max(plasma_harmonics[0], 1e-30)
    )
    accounted_power = p_rrf + p_match + p_stray + p_abs
    power_error = abs(p_source - accounted_power) / max(abs(p_source), 1e-30)

    return SimulationMetrics(
        absorbed_power_w=p_abs,
        source_delivered_power_w=p_source,
        source_resistor_loss_w=p_rrf,
        match_loss_w=p_match,
        stray_loss_w=p_stray,
        mean_sheath_powered_v=mean(w["v_sheath_powered"]),
        mean_sheath_grounded_v=mean(w["v_sheath_grounded"]),
        input_impedance_real_ohm=float(z_input.real),
        input_impedance_imag_ohm=float(z_input.imag),
        load_impedance_real_ohm=float(z_load.real),
        load_impedance_imag_ohm=float(z_load.imag),
        generator_current_amplitude_a=abs(i_generator),
        plasma_current_amplitude_a=plasma_harmonics[0],
        load_current_amplitude_a=abs(i_load),
        stray_current_amplitude_a=abs(_phasor(t, w["i_stray"], frequency)),
        plasma_voltage_amplitude_v=abs(_phasor(t, w["v_plasma"], frequency)),
        plasma_voltage_offset_v=mean(w["v_plasma"]),
        plasma_current_thd=thd,
        cycle_l2_voltage=_cycle_l2(time_s, waveforms["v_plasma"], frequency),
        cycle_l2_current=_cycle_l2(time_s, waveforms["i_plasma_port"], frequency),
        power_balance_relative_error=float(power_error),
        harmonic_amplitudes_plasma_a=plasma_harmonics,
        harmonic_amplitudes_generator_a=generator_harmonics,
    )


def run_ngspice(
    config: Mapping[str, Any],
    plasma: PlasmaParameters,
    case_directory: Path,
    match_c1_f: float | None = None,
    match_c2_f: float | None = None,
) -> SimulationResult:
    """Run ngspice for one frozen-density circuit and return analyzed waveforms."""
    case_directory.mkdir(parents=True, exist_ok=True)
    netlist = render_netlist(config, plasma, match_c1_f, match_c2_f)
    netlist_path = case_directory / "case.cir"
    netlist_path.write_text(netlist, encoding="ascii", newline="\n")
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
    if completed.returncode != 0:
        log = (case_directory / "ngspice.log").read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"ngspice failed with exit code {completed.returncode}:\n{log[-4000:]}")

    waveform_path = case_directory / "waveforms.dat"
    if not waveform_path.is_file():
        log = (case_directory / "ngspice.log").read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"ngspice produced no waveform file despite exit code 0:\n{log[-4000:]}")
    time_s, waveforms = _parse_wrdata(waveform_path)
    metrics = analyze_waveforms(config, time_s, waveforms)
    return SimulationResult(metrics, time_s, waveforms, case_directory)
