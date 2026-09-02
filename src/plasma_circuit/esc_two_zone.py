"""Two-zone wafer/focus-ring ESC electrical and transport validation model."""

from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.integrate import solve_ivp

from plasma_circuit.esc_model import (
    EscSurfaceParameters,
    _dielectric_esr,
    _fmt,
    _matching_inductor_esr,
    _source_and_dielectric_lines,
    _surface_parameters,
)
from plasma_circuit.ngspice import _cycle_l2, _phasor, _window
from plasma_circuit.physics import (
    ATOMIC_MASS_UNIT_KG,
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    argon_rate_coefficients,
    neutral_density,
)


ZONE_NAMES = ("wafer", "focus_ring")
VECTOR_SPECS = (
    ("v_source_wafer", "v(w_src)"),
    ("v_source_focus", "v(f_src)"),
    ("v_input_wafer", "v(w_series_in)"),
    ("v_input_focus", "v(f_series_in)"),
    ("v_surface_wafer", "v(wafer)"),
    ("v_surface_focus", "v(focus)"),
    ("v_sheath_wafer", "v(w_sheath_bulk,wafer)"),
    ("v_sheath_focus", "v(f_sheath_bulk,focus)"),
    ("v_sheath_ground_wafer", "v(w_zone)"),
    ("v_sheath_ground_focus", "v(f_zone)"),
    ("v_zone_difference", "v(w_zone,f_zone)"),
    ("v_dielectric_wafer", "v(w_electrode,w_feed)"),
    ("v_dielectric_focus", "v(f_electrode,f_feed)"),
    ("i_source_wafer", "i(Vsense_generator_wafer)"),
    ("i_source_focus", "i(Vsense_generator_focus)"),
    ("i_match_wafer", "i(Lseries_wafer)"),
    ("i_match_focus", "i(Lseries_focus)"),
    ("i_dielectric_wafer", "i(Vsense_dielectric_wafer)"),
    ("i_dielectric_focus", "i(Vsense_dielectric_focus)"),
    ("i_surface_wafer", "i(Vsense_surface_wafer)"),
    ("i_surface_focus", "i(Vsense_surface_focus)"),
    ("i_lateral", "i(Llateral)"),
)


@dataclass(frozen=True)
class ZoneState:
    electron_density_m3: float
    electron_temperature_ev: float


@dataclass(frozen=True)
class ZoneCircuitParameters:
    state: ZoneState
    neutral_density_m3: float
    mean_electron_speed_m_s: float
    bohm_speed_m_s: float
    momentum_collision_frequency_hz: float
    effective_collision_frequency_hz: float
    equilibrium_sheath_voltage_v: float
    powered: EscSurfaceParameters
    ground: EscSurfaceParameters


@dataclass(frozen=True)
class LateralCircuitParameters:
    coupling_scale: float
    interface_density_m3: float
    interface_temperature_ev: float
    momentum_collision_frequency_hz: float
    effective_collision_frequency_hz: float
    inductance_h: float
    resistance_ohm: float
    impedance_magnitude_ohm: float


@dataclass(frozen=True)
class TwoZoneCircuitParameters:
    wafer: ZoneCircuitParameters
    focus_ring: ZoneCircuitParameters
    lateral: LateralCircuitParameters


@dataclass(frozen=True)
class TwoZoneMetrics:
    absorbed_power_total_w: float
    absorbed_power_wafer_port_w: float
    absorbed_power_focus_port_w: float
    source_delivered_total_w: float
    power_balance_relative_error: float
    bulk_voltage_wafer_amplitude_v: float
    bulk_voltage_focus_amplitude_v: float
    bulk_voltage_difference_amplitude_v: float
    bulk_voltage_difference_offset_v: float
    lateral_current_amplitude_a: float
    lateral_resistive_loss_w: float
    mean_sheath_wafer_v: float
    mean_sheath_focus_v: float
    mean_sheath_ground_wafer_v: float
    mean_sheath_ground_focus_v: float
    input_impedance_wafer_real_ohm: float
    input_impedance_wafer_imag_ohm: float
    input_impedance_focus_real_ohm: float
    input_impedance_focus_imag_ohm: float
    cycle_l2_max: float


@dataclass(frozen=True)
class TwoZoneSimulationResult:
    metrics: TwoZoneMetrics
    time_s: np.ndarray
    waveforms: Mapping[str, np.ndarray]
    case_directory: Path


@dataclass(frozen=True)
class TransportValidationResult:
    time_s: np.ndarray
    density_wafer_m3: np.ndarray
    density_focus_m3: np.ndarray
    temperature_wafer_ev: np.ndarray
    temperature_focus_ev: np.ndarray
    initial_particle_count: float
    final_particle_count: float
    initial_energy_j: float
    final_energy_j: float
    particle_conservation_relative_error: float
    energy_conservation_relative_error: float
    final_density_nonuniformity: float
    final_temperature_nonuniformity: float


def load_two_zone_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the staged two-zone validation configuration."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "ngspice_path",
        "frequency_hz",
        "pressure_pa",
        "gas_temperature_k",
        "surfaces",
        "zones",
        "interface",
        "regularization",
        "transient",
        "validation",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing two-zone configuration fields: {missing}")
    if set(config["surfaces"]) != set(ZONE_NAMES):
        raise ValueError(f"surfaces must be exactly {ZONE_NAMES}")
    if set(config["zones"]) != set(ZONE_NAMES):
        raise ValueError(f"zones must be exactly {ZONE_NAMES}")
    if not Path(config["ngspice_path"]).is_file():
        raise FileNotFoundError(f"ngspice executable not found: {config['ngspice_path']}")
    for name in ZONE_NAMES:
        zone = config["zones"][name]
        for field in (
            "volume_m3",
            "grounded_area_m2",
            "bulk_path_length_m",
            "electron_density_m3",
            "electron_temperature_ev",
        ):
            if float(zone[field]) <= 0.0:
                raise ValueError(f"zones.{name}.{field} must be positive")
    for field in ("path_length_m", "area_m2"):
        if float(config["interface"][field]) <= 0.0:
            raise ValueError(f"interface.{field} must be positive")
    for field in (
        "particle_exchange_conductance_m3_s",
        "thermal_exchange_conductance_m3_s",
    ):
        if float(config["interface"][field]) < 0.0:
            raise ValueError(f"interface.{field} cannot be negative")
    return config


def zone_states_from_config(config: Mapping[str, Any]) -> dict[str, ZoneState]:
    return {
        name: ZoneState(
            electron_density_m3=float(config["zones"][name]["electron_density_m3"]),
            electron_temperature_ev=float(
                config["zones"][name]["electron_temperature_ev"]
            ),
        )
        for name in ZONE_NAMES
    }


def _zone_parameters(
    config: Mapping[str, Any], name: str, state: ZoneState
) -> ZoneCircuitParameters:
    if state.electron_density_m3 <= 0.0 or state.electron_temperature_ev <= 0.0:
        raise ValueError("zone density and temperature must be positive")
    zone = config["zones"][name]
    surface = config["surfaces"][name]
    argon_mass = float(config.get("argon_mass_amu", 39.948)) * ATOMIC_MASS_UNIT_KG
    n_gas = neutral_density(float(config["pressure_pa"]), float(config["gas_temperature_k"]))
    rates = argon_rate_coefficients(state.electron_temperature_ev)
    mean_speed = float(
        np.sqrt(
            8.0
            * ELEMENTARY_CHARGE_C
            * state.electron_temperature_ev
            / (np.pi * ELECTRON_MASS_KG)
        )
    )
    bohm_speed = float(
        np.sqrt(ELEMENTARY_CHARGE_C * state.electron_temperature_ev / argon_mass)
    )
    momentum_frequency = float(n_gas * rates["elastic"])
    bulk_length = float(zone["bulk_path_length_m"])
    effective_frequency = momentum_frequency + mean_speed / bulk_length

    def branch(area_m2: float, path_length_m: float) -> EscSurfaceParameters:
        return _surface_parameters(
            area_m2,
            path_length_m,
            state.electron_density_m3,
            mean_speed,
            bohm_speed,
            effective_frequency,
        )

    return ZoneCircuitParameters(
        state=state,
        neutral_density_m3=n_gas,
        mean_electron_speed_m_s=mean_speed,
        bohm_speed_m_s=bohm_speed,
        momentum_collision_frequency_hz=momentum_frequency,
        effective_collision_frequency_hz=effective_frequency,
        equilibrium_sheath_voltage_v=float(
            state.electron_temperature_ev * np.log(mean_speed / bohm_speed)
        ),
        powered=branch(float(surface["area_m2"]), bulk_length),
        ground=branch(float(zone["grounded_area_m2"]), 0.0),
    )


def compute_two_zone_parameters(
    config: Mapping[str, Any],
    states: Mapping[str, ZoneState] | None = None,
    electrical_coupling_scale: float = 1.0,
) -> TwoZoneCircuitParameters:
    """Calculate two local plasma branches and the lateral electron R-L branch."""
    if electrical_coupling_scale <= 0.0:
        raise ValueError("electrical coupling scale must be positive")
    zone_states = zone_states_from_config(config) if states is None else states
    wafer = _zone_parameters(config, "wafer", zone_states["wafer"])
    focus = _zone_parameters(config, "focus_ring", zone_states["focus_ring"])
    interface = config["interface"]
    density_interface = float(
        2.0
        / (
            1.0 / wafer.state.electron_density_m3
            + 1.0 / focus.state.electron_density_m3
        )
    )
    temperature_interface = float(
        0.5
        * (
            wafer.state.electron_temperature_ev
            + focus.state.electron_temperature_ev
        )
    )
    n_gas = neutral_density(float(config["pressure_pa"]), float(config["gas_temperature_k"]))
    rates = argon_rate_coefficients(temperature_interface)
    momentum_frequency = float(n_gas * rates["elastic"])
    mean_speed = float(
        np.sqrt(
            8.0
            * ELEMENTARY_CHARGE_C
            * temperature_interface
            / (np.pi * ELECTRON_MASS_KG)
        )
    )
    path_length = float(interface["path_length_m"])
    effective_frequency = momentum_frequency + mean_speed / path_length
    effective_area = float(interface["area_m2"]) * electrical_coupling_scale
    inductance = float(
        path_length
        * ELECTRON_MASS_KG
        / (ELEMENTARY_CHARGE_C**2 * density_interface * effective_area)
    )
    resistance = float(effective_frequency * inductance)
    omega = 2.0 * np.pi * float(config["frequency_hz"])
    lateral = LateralCircuitParameters(
        coupling_scale=float(electrical_coupling_scale),
        interface_density_m3=density_interface,
        interface_temperature_ev=temperature_interface,
        momentum_collision_frequency_hz=momentum_frequency,
        effective_collision_frequency_hz=effective_frequency,
        inductance_h=inductance,
        resistance_ohm=resistance,
        impedance_magnitude_ohm=float(np.hypot(resistance, omega * inductance)),
    )
    return TwoZoneCircuitParameters(wafer=wafer, focus_ring=focus, lateral=lateral)


def render_two_zone_netlist(
    config: Mapping[str, Any], plasma: TwoZoneCircuitParameters
) -> str:
    """Render two local bulk potentials connected by a finite lateral R-L path."""
    transient = config["transient"]
    regularization = config["regularization"]
    frequency = float(config["frequency_hz"])
    period = 1.0 / frequency
    total_time = int(transient["cycles"]) * period
    saved_start = total_time - int(transient["saved_cycles"]) * period
    step = period / int(transient["samples_per_cycle"])
    vectors = " ".join(spec for _, spec in VECTOR_SPECS)
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
    wafer = plasma.wafer
    focus = plasma.focus_ring
    lateral = plasma.lateral
    return f"""Two-zone wafer/focus-ring ESC plasma equivalent circuit
* Local bulk nodes w_zone and f_zone are connected only through Rlateral-Llateral.
.param freq={_fmt(frequency)}
.param tew={_fmt(wafer.state.electron_temperature_ev)}
.param tef={_fmt(focus.state.electron_temperature_ev)}
.param veps={_fmt(float(regularization['electron_voltage_v']))}
.param vcap={_fmt(float(regularization['capacitance_voltage_v']))}
.param cscale={_fmt(float(regularization['capacitance_scale']))}
.param kshw={_fmt(wafer.powered.sheath_k)}
.param kshf={_fmt(focus.powered.sheath_k)}
.param kshgw={_fmt(wafer.ground.sheath_k)}
.param kshgf={_fmt(focus.ground.sheath_k)}
.param iesatw={_fmt(wafer.powered.electron_saturation_current_a)}
.param iesatf={_fmt(focus.powered.electron_saturation_current_a)}
.param iesatgw={_fmt(wafer.ground.electron_saturation_current_a)}
.param iesatgf={_fmt(focus.ground.electron_saturation_current_a)}
.func spos(x,d) {{(x > 0) ? 0.5*(x+sqrt(x*x+d*d)) : 0.5*d*d/(sqrt(x*x+d*d)-x)}}

{wafer_external}
{focus_external}

* Wafer-zone powered sheath, local electron inertia/collisions, and ground sheath.
Csh_wafer wafer w_sheath_bulk C='cscale*sqrt(kshw/sqrt(v(w_sheath_bulk,wafer)*v(w_sheath_bulk,wafer)+vcap*vcap))'
Belectron_wafer wafer w_sheath_bulk I='iesatw*exp(-spos(v(w_sheath_bulk,wafer),veps)/tew)'
Iion_wafer w_sheath_bulk wafer DC {_fmt(wafer.powered.ion_current_a)}
Lbulk_wafer w_sheath_bulk w_bulk_mid {_fmt(wafer.powered.bulk_inductance_h)}
Rbulk_wafer w_bulk_mid w_zone {_fmt(wafer.powered.bulk_resistance_ohm)}
Csh_ground_wafer w_zone 0 C='cscale*sqrt(kshgw/sqrt(v(w_zone)*v(w_zone)+vcap*vcap))'
Belectron_ground_wafer 0 w_zone I='iesatgw*exp(-spos(v(w_zone),veps)/tew)'
Iion_ground_wafer w_zone 0 DC {_fmt(wafer.ground.ion_current_a)}

* Focus-zone powered sheath, local electron inertia/collisions, and ground sheath.
Csh_focus focus f_sheath_bulk C='cscale*sqrt(kshf/sqrt(v(f_sheath_bulk,focus)*v(f_sheath_bulk,focus)+vcap*vcap))'
Belectron_focus focus f_sheath_bulk I='iesatf*exp(-spos(v(f_sheath_bulk,focus),veps)/tef)'
Iion_focus f_sheath_bulk focus DC {_fmt(focus.powered.ion_current_a)}
Lbulk_focus f_sheath_bulk f_bulk_mid {_fmt(focus.powered.bulk_inductance_h)}
Rbulk_focus f_bulk_mid f_zone {_fmt(focus.powered.bulk_resistance_ohm)}
Csh_ground_focus f_zone 0 C='cscale*sqrt(kshgf/sqrt(v(f_zone)*v(f_zone)+vcap*vcap))'
Belectron_ground_focus 0 f_zone I='iesatgf*exp(-spos(v(f_zone),veps)/tef)'
Iion_ground_focus f_zone 0 DC {_fmt(focus.ground.ion_current_a)}

* Finite lateral electron-current path: positive current is wafer zone -> focus zone.
Rlateral w_zone lateral_mid {_fmt(lateral.resistance_ohm)}
Llateral lateral_mid f_zone {_fmt(lateral.inductance_h)}

.ic v(wafer)=0 v(focus)=0 v(w_sheath_bulk)={_fmt(wafer.equilibrium_sheath_voltage_v)} v(w_bulk_mid)={_fmt(wafer.equilibrium_sheath_voltage_v)} v(w_zone)={_fmt(wafer.equilibrium_sheath_voltage_v)} v(f_sheath_bulk)={_fmt(focus.equilibrium_sheath_voltage_v)} v(f_bulk_mid)={_fmt(focus.equilibrium_sheath_voltage_v)} v(f_zone)={_fmt(focus.equilibrium_sheath_voltage_v)}
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
        raise RuntimeError(f"ngspice two-zone waveform output is empty: {path}")
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    expected_columns = 1 + len(VECTOR_SPECS)
    if data.shape[1] != expected_columns:
        raise RuntimeError(
            f"unexpected two-zone wrdata columns {data.shape[1]}; expected {expected_columns}"
        )
    return data[:, 0], {
        name: data[:, index + 1]
        for index, (name, _) in enumerate(VECTOR_SPECS)
    }


def analyze_two_zone_waveforms(
    config: Mapping[str, Any],
    plasma: TwoZoneCircuitParameters,
    time_s: np.ndarray,
    waveforms: Mapping[str, np.ndarray],
) -> TwoZoneMetrics:
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
    p_source = mean(w["v_source_wafer"] * w["i_source_wafer"]) + mean(
        w["v_source_focus"] * w["i_source_focus"]
    )
    loss_external = 0.0
    for name in ZONE_NAMES:
        label = "focus" if name == "focus_ring" else name
        surface = config["surfaces"][name]
        loss_external += mean(w[f"i_source_{label}"] ** 2) * float(
            surface["source_resistance_ohm"]
        )
        loss_external += mean(w[f"i_match_{label}"] ** 2) * _matching_inductor_esr(
            surface, frequency
        )
        loss_external += mean(w[f"i_dielectric_{label}"] ** 2) * _dielectric_esr(
            surface, frequency
        )
        loss_external += mean(w[f"v_dielectric_{label}"] ** 2) / float(
            surface["leakage_resistance_ohm"]
        )
    power_error = abs(p_source - loss_external - p_absorbed) / max(abs(p_source), 1.0e-30)
    impedance_wafer = _phasor(t, w["v_input_wafer"], frequency) / _phasor(
        t, w["i_source_wafer"], frequency
    )
    impedance_focus = _phasor(t, w["v_input_focus"], frequency) / _phasor(
        t, w["i_source_focus"], frequency
    )
    zone_wafer = w["v_sheath_ground_wafer"]
    zone_focus = w["v_sheath_ground_focus"]
    cycle_l2_values = (
        _cycle_l2(time_s, waveforms["v_surface_wafer"], frequency),
        _cycle_l2(time_s, waveforms["v_surface_focus"], frequency),
        _cycle_l2(time_s, waveforms["v_zone_difference"], frequency),
        _cycle_l2(time_s, waveforms["i_lateral"], frequency),
    )
    return TwoZoneMetrics(
        absorbed_power_total_w=float(p_absorbed),
        absorbed_power_wafer_port_w=float(p_wafer),
        absorbed_power_focus_port_w=float(p_focus),
        source_delivered_total_w=float(p_source),
        power_balance_relative_error=float(power_error),
        bulk_voltage_wafer_amplitude_v=abs(_phasor(t, zone_wafer, frequency)),
        bulk_voltage_focus_amplitude_v=abs(_phasor(t, zone_focus, frequency)),
        bulk_voltage_difference_amplitude_v=abs(
            _phasor(t, w["v_zone_difference"], frequency)
        ),
        bulk_voltage_difference_offset_v=mean(w["v_zone_difference"]),
        lateral_current_amplitude_a=abs(_phasor(t, w["i_lateral"], frequency)),
        lateral_resistive_loss_w=mean(w["i_lateral"] ** 2)
        * plasma.lateral.resistance_ohm,
        mean_sheath_wafer_v=mean(w["v_sheath_wafer"]),
        mean_sheath_focus_v=mean(w["v_sheath_focus"]),
        mean_sheath_ground_wafer_v=mean(w["v_sheath_ground_wafer"]),
        mean_sheath_ground_focus_v=mean(w["v_sheath_ground_focus"]),
        input_impedance_wafer_real_ohm=float(impedance_wafer.real),
        input_impedance_wafer_imag_ohm=float(impedance_wafer.imag),
        input_impedance_focus_real_ohm=float(impedance_focus.real),
        input_impedance_focus_imag_ohm=float(impedance_focus.imag),
        cycle_l2_max=float(max(cycle_l2_values)),
    )


def run_two_zone_ngspice(
    config: Mapping[str, Any],
    plasma: TwoZoneCircuitParameters,
    case_directory: Path,
) -> TwoZoneSimulationResult:
    case_directory.mkdir(parents=True, exist_ok=True)
    netlist_path = case_directory / "case.cir"
    netlist_path.write_text(
        render_two_zone_netlist(config, plasma), encoding="ascii", newline="\n"
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
        raise RuntimeError(
            f"ngspice two-zone case failed ({completed.returncode}):\n{log[-4000:]}"
        )
    waveform_path = case_directory / "waveforms.dat"
    if not waveform_path.is_file():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"ngspice two-zone case produced no waveforms:\n{log[-4000:]}")
    time_s, waveforms = _parse_waveforms(waveform_path)
    metrics = analyze_two_zone_waveforms(config, plasma, time_s, waveforms)
    return TwoZoneSimulationResult(metrics, time_s, waveforms, case_directory)


def run_closed_transport_validation(
    config: Mapping[str, Any], exchange_scale: float = 1.0
) -> TransportValidationResult:
    """Relax a closed two-zone imbalance while conserving particles and energy.

    This is intentionally a transport-only test. Ionization, wall loss, and RF
    heating are disabled so any drift in total N or U exposes an exchange-term bug.
    """
    if exchange_scale < 0.0:
        raise ValueError("exchange scale cannot be negative")
    validation = config["validation"]["transport"]
    volumes = np.array(
        [
            float(config["zones"]["wafer"]["volume_m3"]),
            float(config["zones"]["focus_ring"]["volume_m3"]),
        ]
    )
    initial_density = np.array(
        [
            float(validation["initial_density_wafer_m3"]),
            float(validation["initial_density_focus_m3"]),
        ]
    )
    initial_temperature = np.array(
        [
            float(validation["initial_temperature_wafer_ev"]),
            float(validation["initial_temperature_focus_ev"]),
        ]
    )
    initial_particles = initial_density * volumes
    initial_energy = 1.5 * ELEMENTARY_CHARGE_C * initial_particles * initial_temperature
    y0 = np.concatenate((initial_particles, initial_energy))
    particle_conductance = (
        float(config["interface"]["particle_exchange_conductance_m3_s"])
        * exchange_scale
    )
    thermal_conductance = (
        float(config["interface"]["thermal_exchange_conductance_m3_s"])
        * exchange_scale
    )

    def rhs(_time_s: float, state: np.ndarray) -> np.ndarray:
        particles = np.maximum(state[:2], 1.0)
        energy = np.maximum(state[2:], 1.0e-30)
        density = particles / volumes
        temperature = energy / (1.5 * ELEMENTARY_CHARGE_C * particles)
        particle_rate = particle_conductance * (density[0] - density[1])
        donor_temperature = temperature[0] if particle_rate >= 0.0 else temperature[1]
        advected_power = 1.5 * ELEMENTARY_CHARGE_C * donor_temperature * particle_rate
        mean_density = 0.5 * (density[0] + density[1])
        conducted_power = (
            1.5
            * ELEMENTARY_CHARGE_C
            * thermal_conductance
            * mean_density
            * (temperature[0] - temperature[1])
        )
        return np.array(
            [
                -particle_rate,
                particle_rate,
                -advected_power - conducted_power,
                advected_power + conducted_power,
            ]
        )

    duration = float(validation["duration_s"])
    samples = int(validation.get("samples", 401))
    solution = solve_ivp(
        rhs,
        (0.0, duration),
        y0,
        t_eval=np.linspace(0.0, duration, samples),
        method="DOP853",
        rtol=1.0e-10,
        atol=np.array([1.0e3, 1.0e3, 1.0e-15, 1.0e-15]),
    )
    if not solution.success:
        raise RuntimeError(f"two-zone transport integration failed: {solution.message}")
    particles = solution.y[:2]
    energy = solution.y[2:]
    density = particles / volumes[:, None]
    temperature = energy / (1.5 * ELEMENTARY_CHARGE_C * particles)
    initial_particle_count = float(np.sum(particles[:, 0]))
    final_particle_count = float(np.sum(particles[:, -1]))
    initial_energy_j = float(np.sum(energy[:, 0]))
    final_energy_j = float(np.sum(energy[:, -1]))

    def nonuniformity(values: np.ndarray) -> float:
        return float(abs(values[0] - values[1]) / max(np.mean(values), 1.0e-30))

    return TransportValidationResult(
        time_s=solution.t,
        density_wafer_m3=density[0],
        density_focus_m3=density[1],
        temperature_wafer_ev=temperature[0],
        temperature_focus_ev=temperature[1],
        initial_particle_count=initial_particle_count,
        final_particle_count=final_particle_count,
        initial_energy_j=initial_energy_j,
        final_energy_j=final_energy_j,
        particle_conservation_relative_error=abs(
            final_particle_count - initial_particle_count
        )
        / initial_particle_count,
        energy_conservation_relative_error=abs(final_energy_j - initial_energy_j)
        / initial_energy_j,
        final_density_nonuniformity=nonuniformity(density[:, -1]),
        final_temperature_nonuniformity=nonuniformity(temperature[:, -1]),
    )


def electrical_coupling_sweep(
    config: Mapping[str, Any], output_directory: Path
) -> list[dict[str, Any]]:
    """Run the finite-coupling electrical limit cases declared in the config."""
    results: list[dict[str, Any]] = []
    states = zone_states_from_config(config)
    for scale in config["validation"]["electrical_coupling_scales"]:
        value = float(scale)
        plasma = compute_two_zone_parameters(config, states, value)
        label = f"scale_{value:.6g}".replace(".", "p")
        simulation = run_two_zone_ngspice(config, plasma, output_directory / label)
        results.append(
            {
                "electrical_coupling_scale": value,
                "lateral": asdict(plasma.lateral),
                "metrics": asdict(simulation.metrics),
            }
        )
    return results


def config_with_electrical_scale(
    config: Mapping[str, Any], electrical_coupling_scale: float
) -> tuple[dict[str, Any], TwoZoneCircuitParameters]:
    """Small public helper used by tests and interactive sensitivity studies."""
    copied = copy.deepcopy(config)
    return copied, compute_two_zone_parameters(copied, electrical_coupling_scale=electrical_coupling_scale)
