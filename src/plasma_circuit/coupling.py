"""Self-consistent density and circuit iteration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from plasma_circuit.ngspice import SimulationResult, run_ngspice
from plasma_circuit.physics import (
    compute_plasma_parameters,
    density_from_power_balance,
    electron_temperature_from_particle_balance,
)


@dataclass(frozen=True)
class DensityIteration:
    iteration: int
    density_input_m3: float
    density_target_m3: float
    density_output_m3: float
    relative_change: float
    absorbed_power_w: float
    mean_sheath_powered_v: float
    mean_sheath_grounded_v: float
    input_impedance_real_ohm: float
    input_impedance_imag_ohm: float
    cycle_l2_voltage: float
    cycle_l2_current: float


@dataclass(frozen=True)
class CoupledResult:
    electron_temperature_ev: float
    electron_density_m3: float
    match_c1_f: float
    match_c2_f: float
    converged: bool
    history: tuple[DensityIteration, ...]
    final_simulation: SimulationResult


@dataclass(frozen=True)
class MatchingIteration:
    iteration: int
    match_c1_f: float
    match_c2_f: float
    target_c1_f: float
    target_c2_f: float
    electron_density_m3: float
    input_impedance_real_ohm: float
    input_impedance_imag_ohm: float
    load_impedance_real_ohm: float
    load_impedance_imag_ohm: float
    capacitance_relative_change: float
    impedance_relative_error: float


@dataclass(frozen=True)
class MatchedResult:
    converged: bool
    match_history: tuple[MatchingIteration, ...]
    coupled_result: CoupledResult


def solve_self_consistent_density(
    config: Mapping[str, Any],
    output_directory: Path,
    match_c1_f: float | None = None,
    match_c2_f: float | None = None,
    initial_density_m3: float | None = None,
) -> CoupledResult:
    """Iterate the circuit power and global power balance to a fixed density."""
    coupling = config["coupling"]
    density = float(
        coupling["initial_density_m3"] if initial_density_m3 is None else initial_density_m3
    )
    relaxation = float(coupling["relaxation"])
    tolerance = float(coupling["relative_tolerance"])
    required_consecutive = int(coupling.get("consecutive_converged", 1))
    cycle_l2_tolerance = float(coupling.get("cycle_l2_tolerance", np.inf))
    max_iterations = int(coupling["max_iterations"])
    temperature = electron_temperature_from_particle_balance(config)
    c1 = float(config["match_c1_f"] if match_c1_f is None else match_c1_f)
    c2 = float(config["match_c2_f"] if match_c2_f is None else match_c2_f)
    history: list[DensityIteration] = []
    final_simulation: SimulationResult | None = None
    lower_log_density: float | None = None
    upper_log_density: float | None = None
    consecutive_converged = 0

    for iteration in range(max_iterations):
        plasma = compute_plasma_parameters(config, density, temperature)
        simulation = run_ngspice(
            config,
            plasma,
            output_directory / f"density_{iteration:02d}",
            c1,
            c2,
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
        # The power-balance map becomes steep around the high-Q match. Once opposite
        # residual signs are observed, bracket the root in log-density and bisect it.
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
        history.append(
            DensityIteration(
                iteration=iteration,
                density_input_m3=density,
                density_target_m3=target,
                density_output_m3=density_next,
                relative_change=relative_change,
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
        density_converged = abs(np.expm1(log_residual)) < tolerance
        rf_converged = max(metrics.cycle_l2_voltage, metrics.cycle_l2_current) < cycle_l2_tolerance
        consecutive_converged = (
            consecutive_converged + 1 if density_converged and rf_converged else 0
        )
        if consecutive_converged >= required_consecutive:
            return CoupledResult(
                electron_temperature_ev=temperature,
                electron_density_m3=density,
                match_c1_f=c1,
                match_c2_f=c2,
                converged=True,
                history=tuple(history),
                final_simulation=final_simulation,
            )
        density = density_next

    assert final_simulation is not None
    return CoupledResult(
        electron_temperature_ev=temperature,
        # The final waveform belongs to the density evaluated in the last history row.
        electron_density_m3=history[-1].density_input_m3,
        match_c1_f=c1,
        match_c2_f=c2,
        converged=False,
        history=tuple(history),
        final_simulation=final_simulation,
    )


def matching_capacitances_from_load(
    load_impedance: complex,
    frequency_hz: float,
    match_inductance_h: float,
    source_resistance_ohm: float,
) -> tuple[float, float]:
    """Return the L-network capacitors that transform a measured load to a real source resistance.

    The topology is the paper's Figure 1: shunt C1 at the source-side node, followed by
    series C2 and L. The positive branch-reactance solution is selected because C1 must be
    positive for this topology.
    """
    resistance = float(load_impedance.real)
    source_resistance = float(source_resistance_ohm)
    if not 0.0 < resistance < source_resistance:
        raise ValueError(
            "paper L-network requires 0 < Re(Z_load) < source resistance; "
            f"got Re(Z_load)={resistance:.6g} ohm"
        )
    omega = 2.0 * np.pi * frequency_hz
    branch_reactance = float(np.sqrt(resistance * (source_resistance - resistance)))
    branch_denominator = resistance**2 + branch_reactance**2
    c1 = branch_reactance / (omega * branch_denominator)
    capacitor_reactance = (
        branch_reactance - load_impedance.imag - omega * match_inductance_h
    )
    if capacitor_reactance >= 0.0:
        raise ValueError(
            "required series reactance is not capacitive; paper topology cannot match this load"
        )
    c2 = -1.0 / (omega * capacitor_reactance)
    return float(c1), float(c2)


def solve_self_consistent_matching(
    config: Mapping[str, Any], output_directory: Path
) -> MatchedResult:
    """Alternate density convergence and analytical fundamental-frequency matching."""
    matching = config.get("matching", {})
    relaxation = float(matching.get("relaxation", 0.7))
    capacitance_tolerance = float(matching.get("relative_tolerance", 2e-3))
    impedance_tolerance = float(matching.get("impedance_tolerance", 2e-3))
    max_iterations = int(matching.get("max_iterations", 10))
    c1 = float(config["match_c1_f"])
    c2 = float(config["match_c2_f"])
    density: float | None = None
    history: list[MatchingIteration] = []
    final: CoupledResult | None = None

    for iteration in range(max_iterations):
        coupled = solve_self_consistent_density(
            config,
            output_directory / f"matching_{iteration:02d}",
            c1,
            c2,
            initial_density_m3=density,
        )
        final = coupled
        density = coupled.electron_density_m3
        metrics = coupled.final_simulation.metrics
        z_load = complex(metrics.load_impedance_real_ohm, metrics.load_impedance_imag_ohm)
        target_c1, target_c2 = matching_capacitances_from_load(
            z_load,
            float(config["frequency_hz"]),
            float(config["match_l_h"]),
            float(config["source_resistance_ohm"]),
        )
        c1_next = float(np.exp((1.0 - relaxation) * np.log(c1) + relaxation * np.log(target_c1)))
        c2_next = float(np.exp((1.0 - relaxation) * np.log(c2) + relaxation * np.log(target_c2)))
        capacitance_change = max(abs(c1_next / c1 - 1.0), abs(c2_next / c2 - 1.0))
        impedance_error = float(
            np.hypot(
                metrics.input_impedance_real_ohm - float(config["source_resistance_ohm"]),
                metrics.input_impedance_imag_ohm,
            )
            / float(config["source_resistance_ohm"])
        )
        history.append(
            MatchingIteration(
                iteration=iteration,
                match_c1_f=c1,
                match_c2_f=c2,
                target_c1_f=target_c1,
                target_c2_f=target_c2,
                electron_density_m3=density,
                input_impedance_real_ohm=metrics.input_impedance_real_ohm,
                input_impedance_imag_ohm=metrics.input_impedance_imag_ohm,
                load_impedance_real_ohm=metrics.load_impedance_real_ohm,
                load_impedance_imag_ohm=metrics.load_impedance_imag_ohm,
                capacitance_relative_change=capacitance_change,
                impedance_relative_error=impedance_error,
            )
        )
        if capacitance_change < capacitance_tolerance and impedance_error < impedance_tolerance:
            return MatchedResult(True, tuple(history), coupled)
        c1, c2 = c1_next, c2_next

    assert final is not None
    # Ensure the reported result uses the last updated capacitors rather than the previous iterate.
    final = solve_self_consistent_density(
        config,
        output_directory / "final",
        c1,
        c2,
        initial_density_m3=density,
    )
    return MatchedResult(False, tuple(history), final)


def coupled_result_to_dict(result: CoupledResult) -> dict[str, Any]:
    """Convert a coupled result to a JSON-compatible dictionary."""
    return {
        "electron_temperature_ev": result.electron_temperature_ev,
        "electron_density_m3": result.electron_density_m3,
        "match_c1_f": result.match_c1_f,
        "match_c2_f": result.match_c2_f,
        "converged": result.converged,
        "history": [asdict(row) for row in result.history],
        "metrics": asdict(result.final_simulation.metrics),
    }


def matched_result_to_dict(result: MatchedResult) -> dict[str, Any]:
    """Convert a matched result to a JSON-compatible dictionary."""
    data = coupled_result_to_dict(result.coupled_result)
    data["matching_converged"] = result.converged
    data["matching_history"] = [asdict(row) for row in result.match_history]
    return data
