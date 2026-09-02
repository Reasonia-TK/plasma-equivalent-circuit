"""Self-consistent two-zone global model coupled to the ngspice ESC circuit."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares

from plasma_circuit.esc_two_zone import (
    TwoZoneMetrics,
    TwoZoneSimulationResult,
    ZoneState,
    compute_two_zone_parameters,
    run_two_zone_ngspice,
    zone_states_from_config,
)
from plasma_circuit.physics import (
    ATOMIC_MASS_UNIT_KG,
    ELEMENTARY_CHARGE_C,
    argon_rate_coefficients,
    collisional_energy_loss_ev,
    neutral_density,
)


ZONE_NAMES = ("wafer", "focus_ring")


@dataclass(frozen=True)
class ZoneBalanceTerms:
    ionization_rate_s: float
    wall_loss_rate_s: float
    collisional_loss_w: float
    electron_wall_loss_w: float
    ion_wall_loss_w: float
    total_loss_w: float
    ion_flux_m2_s: float


@dataclass(frozen=True)
class TwoZoneBalanceEvaluation:
    wafer: ZoneBalanceTerms
    focus_ring: ZoneBalanceTerms
    particle_exchange_wafer_to_focus_s: float
    advected_electron_power_wafer_to_focus_w: float
    conducted_electron_power_wafer_to_focus_w: float
    total_energy_exchange_wafer_to_focus_w: float
    particle_residual_wafer_s: float
    particle_residual_focus_s: float
    power_residual_wafer_w: float
    power_residual_focus_w: float
    normalized_particle_residual_wafer: float
    normalized_particle_residual_focus: float
    normalized_power_residual_wafer: float
    normalized_power_residual_focus: float

    @property
    def max_normalized_residual(self) -> float:
        return float(
            max(
                abs(self.normalized_particle_residual_wafer),
                abs(self.normalized_particle_residual_focus),
                abs(self.normalized_power_residual_wafer),
                abs(self.normalized_power_residual_focus),
            )
        )


@dataclass(frozen=True)
class BalanceSolveResult:
    states: Mapping[str, ZoneState]
    evaluation: TwoZoneBalanceEvaluation
    success: bool
    function_evaluations: int
    optimality: float
    message: str


@dataclass(frozen=True)
class TwoZoneCoupledIteration:
    iteration: int
    density_wafer_input_m3: float
    density_focus_input_m3: float
    temperature_wafer_input_ev: float
    temperature_focus_input_ev: float
    density_wafer_target_m3: float
    density_focus_target_m3: float
    temperature_wafer_target_ev: float
    temperature_focus_target_ev: float
    density_wafer_output_m3: float
    density_focus_output_m3: float
    temperature_wafer_output_ev: float
    temperature_focus_output_ev: float
    target_relative_change_max: float
    balance_residual_input_max: float
    balance_residual_target_max: float
    absorbed_power_total_w: float
    absorbed_power_wafer_allocated_w: float
    absorbed_power_focus_allocated_w: float
    lateral_power_transfer_w: float
    cycle_l2_max: float


@dataclass(frozen=True)
class TwoZoneUniformityMetrics:
    density_nonuniformity: float
    temperature_nonuniformity: float
    ion_flux_nonuniformity: float
    density_ratio_focus_to_wafer: float
    temperature_ratio_focus_to_wafer: float
    ion_flux_ratio_focus_to_wafer: float


@dataclass(frozen=True)
class TwoZoneCoupledResult:
    states: Mapping[str, ZoneState]
    converged: bool
    history: tuple[TwoZoneCoupledIteration, ...]
    final_simulation: TwoZoneSimulationResult
    final_balance: TwoZoneBalanceEvaluation
    uniformity: TwoZoneUniformityMetrics


def _surface_sheath_voltages(metrics: TwoZoneMetrics) -> dict[str, tuple[float, float]]:
    return {
        "wafer": (
            metrics.mean_sheath_wafer_v,
            metrics.mean_sheath_ground_wafer_v,
        ),
        "focus_ring": (
            metrics.mean_sheath_focus_v,
            metrics.mean_sheath_ground_focus_v,
        ),
    }


def _allocated_powers(metrics: TwoZoneMetrics) -> dict[str, float]:
    return {
        "wafer": metrics.absorbed_power_wafer_allocated_w,
        "focus_ring": metrics.absorbed_power_focus_allocated_w,
    }


def _zone_balance_terms(
    config: Mapping[str, Any],
    name: str,
    state: ZoneState,
    sheath_voltages_v: tuple[float, float],
) -> ZoneBalanceTerms:
    zone = config["zones"][name]
    surface = config["surfaces"][name]
    volume = float(zone["volume_m3"])
    powered_area = float(surface["area_m2"])
    ground_area = float(zone["grounded_area_m2"])
    total_area = powered_area + ground_area
    argon_mass_amu = float(config.get("argon_mass_amu", 39.948))
    argon_mass_kg = argon_mass_amu * ATOMIC_MASS_UNIT_KG
    gas_density = neutral_density(
        float(config["pressure_pa"]), float(config["gas_temperature_k"])
    )
    ionization_coefficient = argon_rate_coefficients(
        state.electron_temperature_ev
    )["ionization"]
    bohm_speed = float(
        np.sqrt(
            ELEMENTARY_CHARGE_C
            * state.electron_temperature_ev
            / argon_mass_kg
        )
    )
    ionization_rate = (
        state.electron_density_m3
        * gas_density
        * ionization_coefficient
        * volume
    )
    wall_loss_rate = state.electron_density_m3 * bohm_speed * total_area
    collisional_loss = (
        ionization_rate
        * collisional_energy_loss_ev(state.electron_temperature_ev, argon_mass_amu)
        * ELEMENTARY_CHARGE_C
    )
    electron_wall_loss = (
        wall_loss_rate
        * 2.0
        * state.electron_temperature_ev
        * ELEMENTARY_CHARGE_C
    )
    ion_wall_loss = (
        state.electron_density_m3
        * bohm_speed
        * ELEMENTARY_CHARGE_C
        * (
            powered_area
            * (max(sheath_voltages_v[0], 0.0) + 0.5 * state.electron_temperature_ev)
            + ground_area
            * (max(sheath_voltages_v[1], 0.0) + 0.5 * state.electron_temperature_ev)
        )
    )
    return ZoneBalanceTerms(
        ionization_rate_s=float(ionization_rate),
        wall_loss_rate_s=float(wall_loss_rate),
        collisional_loss_w=float(collisional_loss),
        electron_wall_loss_w=float(electron_wall_loss),
        ion_wall_loss_w=float(ion_wall_loss),
        total_loss_w=float(collisional_loss + electron_wall_loss + ion_wall_loss),
        ion_flux_m2_s=float(state.electron_density_m3 * bohm_speed),
    )


def evaluate_two_zone_balances(
    config: Mapping[str, Any],
    states: Mapping[str, ZoneState],
    allocated_powers_w: Mapping[str, float],
    sheath_voltages_v: Mapping[str, tuple[float, float]],
    transport_scale: float = 1.0,
) -> TwoZoneBalanceEvaluation:
    """Evaluate conservative particle and electron-energy balances."""
    if transport_scale < 0.0:
        raise ValueError("transport scale cannot be negative")
    if any(float(allocated_powers_w[name]) <= 0.0 for name in ZONE_NAMES):
        raise ValueError("both allocated local powers must be positive")
    wafer = _zone_balance_terms(config, "wafer", states["wafer"], sheath_voltages_v["wafer"])
    focus = _zone_balance_terms(
        config,
        "focus_ring",
        states["focus_ring"],
        sheath_voltages_v["focus_ring"],
    )
    interface = config["interface"]
    particle_conductance = (
        float(interface["particle_exchange_conductance_m3_s"]) * transport_scale
    )
    thermal_conductance = (
        float(interface["thermal_exchange_conductance_m3_s"]) * transport_scale
    )
    density_wafer = states["wafer"].electron_density_m3
    density_focus = states["focus_ring"].electron_density_m3
    temperature_wafer = states["wafer"].electron_temperature_ev
    temperature_focus = states["focus_ring"].electron_temperature_ev
    particle_exchange = particle_conductance * (density_wafer - density_focus)
    donor_temperature = temperature_wafer if particle_exchange >= 0.0 else temperature_focus
    energy_factor = float(
        config.get("self_consistent", {}).get("particle_energy_transport_factor", 1.5)
    )
    advected_power = (
        energy_factor
        * ELEMENTARY_CHARGE_C
        * donor_temperature
        * particle_exchange
    )
    mean_density = 0.5 * (density_wafer + density_focus)
    conducted_power = (
        1.5
        * ELEMENTARY_CHARGE_C
        * thermal_conductance
        * mean_density
        * (temperature_wafer - temperature_focus)
    )
    energy_exchange = advected_power + conducted_power
    particle_residual_wafer = (
        wafer.ionization_rate_s - wafer.wall_loss_rate_s - particle_exchange
    )
    particle_residual_focus = (
        focus.ionization_rate_s - focus.wall_loss_rate_s + particle_exchange
    )
    power_residual_wafer = (
        float(allocated_powers_w["wafer"]) - wafer.total_loss_w - energy_exchange
    )
    power_residual_focus = (
        float(allocated_powers_w["focus_ring"])
        - focus.total_loss_w
        + energy_exchange
    )
    particle_scale_wafer = max(
        wafer.ionization_rate_s,
        wafer.wall_loss_rate_s,
        abs(particle_exchange),
        1.0,
    )
    particle_scale_focus = max(
        focus.ionization_rate_s,
        focus.wall_loss_rate_s,
        abs(particle_exchange),
        1.0,
    )
    power_scale_wafer = max(
        abs(float(allocated_powers_w["wafer"])),
        wafer.total_loss_w,
        abs(energy_exchange),
        1.0e-12,
    )
    power_scale_focus = max(
        abs(float(allocated_powers_w["focus_ring"])),
        focus.total_loss_w,
        abs(energy_exchange),
        1.0e-12,
    )
    return TwoZoneBalanceEvaluation(
        wafer=wafer,
        focus_ring=focus,
        particle_exchange_wafer_to_focus_s=float(particle_exchange),
        advected_electron_power_wafer_to_focus_w=float(advected_power),
        conducted_electron_power_wafer_to_focus_w=float(conducted_power),
        total_energy_exchange_wafer_to_focus_w=float(energy_exchange),
        particle_residual_wafer_s=float(particle_residual_wafer),
        particle_residual_focus_s=float(particle_residual_focus),
        power_residual_wafer_w=float(power_residual_wafer),
        power_residual_focus_w=float(power_residual_focus),
        normalized_particle_residual_wafer=float(
            particle_residual_wafer / particle_scale_wafer
        ),
        normalized_particle_residual_focus=float(
            particle_residual_focus / particle_scale_focus
        ),
        normalized_power_residual_wafer=float(power_residual_wafer / power_scale_wafer),
        normalized_power_residual_focus=float(power_residual_focus / power_scale_focus),
    )


def _states_to_log_vector(states: Mapping[str, ZoneState]) -> np.ndarray:
    return np.log(
        np.array(
            [
                states["wafer"].electron_density_m3,
                states["focus_ring"].electron_density_m3,
                states["wafer"].electron_temperature_ev,
                states["focus_ring"].electron_temperature_ev,
            ],
            dtype=float,
        )
    )


def _log_vector_to_states(values: np.ndarray) -> dict[str, ZoneState]:
    density_wafer, density_focus, temperature_wafer, temperature_focus = np.exp(values)
    return {
        "wafer": ZoneState(float(density_wafer), float(temperature_wafer)),
        "focus_ring": ZoneState(float(density_focus), float(temperature_focus)),
    }


def _evaluation_vector(evaluation: TwoZoneBalanceEvaluation) -> np.ndarray:
    return np.array(
        [
            evaluation.normalized_particle_residual_wafer,
            evaluation.normalized_particle_residual_focus,
            evaluation.normalized_power_residual_wafer,
            evaluation.normalized_power_residual_focus,
        ]
    )


def solve_two_zone_balance_state(
    config: Mapping[str, Any],
    allocated_powers_w: Mapping[str, float],
    sheath_voltages_v: Mapping[str, tuple[float, float]],
    initial_states: Mapping[str, ZoneState],
    transport_scale: float = 1.0,
) -> BalanceSolveResult:
    """Solve the four steady global balances for frozen circuit observables."""
    solver_config = config.get("self_consistent", {})
    density_bounds = solver_config.get("density_bounds_m3", [1.0e11, 1.0e17])
    temperature_bounds = solver_config.get("temperature_bounds_ev", [0.2, 20.0])
    lower = np.log(
        [density_bounds[0], density_bounds[0], temperature_bounds[0], temperature_bounds[0]]
    )
    upper = np.log(
        [density_bounds[1], density_bounds[1], temperature_bounds[1], temperature_bounds[1]]
    )

    def residual(log_values: np.ndarray) -> np.ndarray:
        evaluation = evaluate_two_zone_balances(
            config,
            _log_vector_to_states(log_values),
            allocated_powers_w,
            sheath_voltages_v,
            transport_scale,
        )
        return _evaluation_vector(evaluation)

    solution = least_squares(
        residual,
        _states_to_log_vector(initial_states),
        bounds=(lower, upper),
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
        max_nfev=int(solver_config.get("balance_max_function_evaluations", 300)),
    )
    states = _log_vector_to_states(solution.x)
    evaluation = evaluate_two_zone_balances(
        config,
        states,
        allocated_powers_w,
        sheath_voltages_v,
        transport_scale,
    )
    residual_tolerance = float(
        solver_config.get("balance_solver_residual_tolerance", 1.0e-7)
    )
    success = bool(solution.success and evaluation.max_normalized_residual <= residual_tolerance)
    return BalanceSolveResult(
        states=states,
        evaluation=evaluation,
        success=success,
        function_evaluations=int(solution.nfev),
        optimality=float(solution.optimality),
        message=str(solution.message),
    )


def _relative_target_change(
    states: Mapping[str, ZoneState], targets: Mapping[str, ZoneState]
) -> float:
    current = np.exp(_states_to_log_vector(states))
    target = np.exp(_states_to_log_vector(targets))
    return float(np.max(np.abs(target / current - 1.0)))


def _relax_states(
    states: Mapping[str, ZoneState],
    targets: Mapping[str, ZoneState],
    relaxation: float,
) -> dict[str, ZoneState]:
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("state relaxation must be in (0, 1]")
    values = _states_to_log_vector(states)
    target_values = _states_to_log_vector(targets)
    return _log_vector_to_states(values + relaxation * (target_values - values))


def compute_uniformity_metrics(
    config: Mapping[str, Any], states: Mapping[str, ZoneState]
) -> TwoZoneUniformityMetrics:
    argon_mass = float(config.get("argon_mass_amu", 39.948)) * ATOMIC_MASS_UNIT_KG

    def ion_flux(state: ZoneState) -> float:
        bohm_speed = np.sqrt(
            ELEMENTARY_CHARGE_C * state.electron_temperature_ev / argon_mass
        )
        return float(state.electron_density_m3 * bohm_speed)

    density = np.array(
        [states["wafer"].electron_density_m3, states["focus_ring"].electron_density_m3]
    )
    temperature = np.array(
        [
            states["wafer"].electron_temperature_ev,
            states["focus_ring"].electron_temperature_ev,
        ]
    )
    flux = np.array([ion_flux(states["wafer"]), ion_flux(states["focus_ring"])])

    def nonuniformity(values: np.ndarray) -> float:
        return float(abs(values[0] - values[1]) / np.mean(values))

    return TwoZoneUniformityMetrics(
        density_nonuniformity=nonuniformity(density),
        temperature_nonuniformity=nonuniformity(temperature),
        ion_flux_nonuniformity=nonuniformity(flux),
        density_ratio_focus_to_wafer=float(density[1] / density[0]),
        temperature_ratio_focus_to_wafer=float(temperature[1] / temperature[0]),
        ion_flux_ratio_focus_to_wafer=float(flux[1] / flux[0]),
    )


def solve_two_zone_coupled_model(
    config: Mapping[str, Any],
    output_directory: Path,
    initial_states: Mapping[str, ZoneState] | None = None,
    transport_scale: float | None = None,
) -> TwoZoneCoupledResult:
    """Iterate the two-zone balances and nonlinear ngspice circuit to a fixed point."""
    if "self_consistent" not in config:
        raise ValueError("two-zone self_consistent configuration is required")
    coupled = config["self_consistent"]
    state = dict(zone_states_from_config(config) if initial_states is None else initial_states)
    relaxation = float(coupled["relaxation"])
    relative_tolerance = float(coupled["relative_tolerance"])
    fixed_point_residual_tolerance = float(coupled["fixed_point_residual_tolerance"])
    cycle_l2_tolerance = float(coupled["cycle_l2_tolerance"])
    required_consecutive = int(coupled.get("consecutive_converged", 2))
    maximum_iterations = int(coupled["max_iterations"])
    electrical_scale = float(coupled.get("electrical_coupling_scale", 1.0))
    actual_transport_scale = float(
        coupled.get("transport_scale", 1.0)
        if transport_scale is None
        else transport_scale
    )
    iteration_cycles = int(coupled.get("iteration_cycles", config["transient"]["cycles"]))
    history: list[TwoZoneCoupledIteration] = []
    consecutive = 0
    converged_in_loop = False

    for iteration in range(maximum_iterations):
        iteration_config = copy.deepcopy(config)
        iteration_config["transient"]["cycles"] = iteration_cycles
        plasma = compute_two_zone_parameters(
            iteration_config,
            state,
            electrical_coupling_scale=electrical_scale,
        )
        simulation = run_two_zone_ngspice(
            iteration_config,
            plasma,
            output_directory / f"iteration_{iteration:02d}",
        )
        powers = _allocated_powers(simulation.metrics)
        sheaths = _surface_sheath_voltages(simulation.metrics)
        input_evaluation = evaluate_two_zone_balances(
            config, state, powers, sheaths, actual_transport_scale
        )
        target = solve_two_zone_balance_state(
            config,
            powers,
            sheaths,
            state,
            actual_transport_scale,
        )
        if not target.success:
            raise RuntimeError(
                "two-zone frozen-circuit balance solve did not converge: "
                f"residual={target.evaluation.max_normalized_residual:.3e}; "
                f"message={target.message}"
            )
        relative_change = _relative_target_change(state, target.states)
        output_state = _relax_states(state, target.states, relaxation)
        history.append(
            TwoZoneCoupledIteration(
                iteration=iteration,
                density_wafer_input_m3=state["wafer"].electron_density_m3,
                density_focus_input_m3=state["focus_ring"].electron_density_m3,
                temperature_wafer_input_ev=state["wafer"].electron_temperature_ev,
                temperature_focus_input_ev=state["focus_ring"].electron_temperature_ev,
                density_wafer_target_m3=target.states["wafer"].electron_density_m3,
                density_focus_target_m3=target.states["focus_ring"].electron_density_m3,
                temperature_wafer_target_ev=target.states["wafer"].electron_temperature_ev,
                temperature_focus_target_ev=target.states["focus_ring"].electron_temperature_ev,
                density_wafer_output_m3=output_state["wafer"].electron_density_m3,
                density_focus_output_m3=output_state["focus_ring"].electron_density_m3,
                temperature_wafer_output_ev=output_state["wafer"].electron_temperature_ev,
                temperature_focus_output_ev=output_state["focus_ring"].electron_temperature_ev,
                target_relative_change_max=relative_change,
                balance_residual_input_max=input_evaluation.max_normalized_residual,
                balance_residual_target_max=target.evaluation.max_normalized_residual,
                absorbed_power_total_w=simulation.metrics.absorbed_power_total_w,
                absorbed_power_wafer_allocated_w=powers["wafer"],
                absorbed_power_focus_allocated_w=powers["focus_ring"],
                lateral_power_transfer_w=simulation.metrics.lateral_power_transfer_midpoint_w,
                cycle_l2_max=simulation.metrics.cycle_l2_max,
            )
        )
        fixed_point_ok = (
            relative_change <= relative_tolerance
            and input_evaluation.max_normalized_residual
            <= fixed_point_residual_tolerance
            and simulation.metrics.cycle_l2_max <= cycle_l2_tolerance
        )
        consecutive = consecutive + 1 if fixed_point_ok else 0
        if consecutive >= required_consecutive:
            converged_in_loop = True
            break
        state = output_state

    final_plasma = compute_two_zone_parameters(
        config, state, electrical_coupling_scale=electrical_scale
    )
    final_simulation = run_two_zone_ngspice(
        config, final_plasma, output_directory / "final"
    )
    final_balance = evaluate_two_zone_balances(
        config,
        state,
        _allocated_powers(final_simulation.metrics),
        _surface_sheath_voltages(final_simulation.metrics),
        actual_transport_scale,
    )
    converged = bool(
        converged_in_loop
        and final_balance.max_normalized_residual <= fixed_point_residual_tolerance
        and final_simulation.metrics.cycle_l2_max <= cycle_l2_tolerance
    )
    return TwoZoneCoupledResult(
        states=state,
        converged=converged,
        history=tuple(history),
        final_simulation=final_simulation,
        final_balance=final_balance,
        uniformity=compute_uniformity_metrics(config, state),
    )


def coupled_result_to_dict(result: TwoZoneCoupledResult) -> dict[str, Any]:
    return {
        "converged": result.converged,
        "states": {name: asdict(state) for name, state in result.states.items()},
        "uniformity": asdict(result.uniformity),
        "final_metrics": asdict(result.final_simulation.metrics),
        "final_balance": asdict(result.final_balance),
        "history": [asdict(item) for item in result.history],
    }
