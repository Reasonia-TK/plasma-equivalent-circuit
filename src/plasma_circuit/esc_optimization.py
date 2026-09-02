"""Optimize two L-type input matching networks around the nonlinear ESC plasma."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from plasma_circuit.esc_model import (
    EscCoupledResult,
    EscMetrics,
    compute_esc_plasma_parameters,
    esc_result_to_dict,
    load_esc_config,
    run_esc_ngspice,
    solve_esc_model,
)


SURFACES = ("wafer", "focus_ring")
VARIABLES = (
    ("wafer", "series_inductance_h"),
    ("wafer", "shunt_capacitance_f"),
    ("focus_ring", "series_inductance_h"),
    ("focus_ring", "shunt_capacitance_f"),
)


def apparent_impedance(metrics: EscMetrics, surface: str) -> complex:
    """Return the fundamental apparent port impedance for the driven state."""
    suffix = "wafer" if surface == "wafer" else "focus"
    return complex(
        getattr(metrics, f"input_impedance_{suffix}_real_ohm"),
        getattr(metrics, f"input_impedance_{suffix}_imag_ohm"),
    )


def reflection_coefficient(impedance: complex, reference_ohm: float) -> complex:
    """Return the voltage-wave reflection coefficient at a real reference impedance."""
    denominator = impedance + reference_ohm
    if abs(denominator) < 1.0e-30:
        return complex(np.inf)
    return (impedance - reference_ohm) / denominator


def deembed_l_match_load(
    input_impedance: complex,
    series_inductance_h: float,
    shunt_capacitance_f: float,
    angular_frequency_rad_s: float,
) -> complex:
    """Remove an input shunt C and series L from an apparent port impedance."""
    branch_admittance = (
        1.0 / input_impedance
        - 1j * angular_frequency_rad_s * shunt_capacitance_f
    )
    return 1.0 / branch_admittance - 1j * angular_frequency_rad_s * series_inductance_h


def analytic_l_match(
    load_impedance: complex,
    reference_ohm: float,
    angular_frequency_rad_s: float,
) -> tuple[float, float]:
    """Synthesize input-shunt-C/series-L matching values for a complex load.

    The positive-reactance branch is selected so both elements are passive when
    the load is capacitively dominated.  A ValueError signals that this fixed
    topology cannot match the supplied load with positive L and C.
    """
    resistance = float(load_impedance.real)
    if not (0.0 < resistance < reference_ohm):
        raise ValueError("analytic L match requires 0 < Re(Zload) < Z0")
    branch_reactance = float(np.sqrt(resistance * (reference_ohm - resistance)))
    inductance = (branch_reactance - float(load_impedance.imag)) / angular_frequency_rad_s
    capacitance = branch_reactance / (
        angular_frequency_rad_s * (resistance**2 + branch_reactance**2)
    )
    if inductance <= 0.0 or capacitance <= 0.0:
        raise ValueError("selected L-match topology requires positive L and C")
    return float(inductance), float(capacitance)


def matching_objective(
    metrics: EscMetrics, reference_impedances_ohm: Mapping[str, float]
) -> tuple[float, dict[str, complex]]:
    """Return the sum of squared simultaneous-drive apparent reflections."""
    reflections = {
        surface: reflection_coefficient(
            apparent_impedance(metrics, surface),
            float(reference_impedances_ohm[surface]),
        )
        for surface in SURFACES
    }
    value = float(sum(abs(value) ** 2 for value in reflections.values()))
    return value, reflections


def _matching_values(config: Mapping[str, Any]) -> dict[str, float]:
    return {
        f"{surface}_{field}": float(config["surfaces"][surface].get(field, 0.0))
        for surface, field in VARIABLES
    }


def _apply_values(config: dict[str, Any], values: Sequence[float]) -> None:
    for value, (surface, field) in zip(values, VARIABLES, strict=True):
        config["surfaces"][surface][field] = float(value)


def _metrics_summary(
    result: EscCoupledResult, reference_impedances_ohm: Mapping[str, float]
) -> dict[str, Any]:
    metrics = result.final_simulation.metrics
    objective, reflections = matching_objective(metrics, reference_impedances_ohm)
    cycle_l2_max = max(
        metrics.cycle_l2_voltage_wafer,
        metrics.cycle_l2_voltage_focus,
        metrics.cycle_l2_current_wafer,
        metrics.cycle_l2_current_focus,
    )
    output = esc_result_to_dict(result)
    output.update(
        {
            "matching_objective": objective,
            "apparent_reflection_magnitude": {
                surface: float(abs(value)) for surface, value in reflections.items()
            },
            "cycle_l2_max": float(cycle_l2_max),
        }
    )
    return output


def _bounds(optimization: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lower: list[float] = []
    upper: list[float] = []
    for surface, field in VARIABLES:
        key = "series_inductance_h" if field == "series_inductance_h" else "shunt_capacitance_f"
        values = optimization["bounds"][surface][key]
        lower.append(float(values[0]))
        upper.append(float(values[1]))
    return np.asarray(lower), np.asarray(upper)


def _analytic_initial_values(
    config: Mapping[str, Any],
    metrics: EscMetrics,
    reference_impedances_ohm: Mapping[str, float],
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    omega = 2.0 * np.pi * float(config["frequency_hz"])
    current = np.asarray(
        [float(config["surfaces"][surface].get(field, 0.0)) for surface, field in VARIABLES]
    )
    initial = current.copy()
    diagnostics: dict[str, Any] = {}
    for surface, offset in (("wafer", 0), ("focus_ring", 2)):
        surface_config = config["surfaces"][surface]
        measured = apparent_impedance(metrics, surface)
        load = deembed_l_match_load(
            measured,
            float(surface_config["series_inductance_h"]),
            float(surface_config.get("shunt_capacitance_f", 0.0)),
            omega,
        )
        row: dict[str, Any] = {
            "measured_apparent_impedance_ohm": [measured.real, measured.imag],
            "deembedded_load_impedance_ohm": [load.real, load.imag],
        }
        try:
            inductance, capacitance = analytic_l_match(
                load, float(reference_impedances_ohm[surface]), omega
            )
            initial[offset : offset + 2] = (inductance, capacitance)
            row["unclamped_analytic_values"] = {
                "series_inductance_h": inductance,
                "shunt_capacitance_f": capacitance,
            }
            row["status"] = "analytic"
        except ValueError as error:
            row["status"] = "retained-current-values"
            row["reason"] = str(error)
        diagnostics[surface] = row
    return np.clip(initial, lower, upper), diagnostics


def _frozen_optimize(
    config: Mapping[str, Any],
    result: EscCoupledResult,
    optimization: Mapping[str, Any],
    round_directory: Path,
    reference_impedances_ohm: Mapping[str, float],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    lower, upper = _bounds(optimization)
    initial, analytic = _analytic_initial_values(
        config,
        result.final_simulation.metrics,
        reference_impedances_ohm,
        lower,
        upper,
    )
    span = upper - lower
    analytic_normalized = (initial - lower) / span
    current_values = np.asarray(
        [
            float(config["surfaces"][surface].get(field, 0.0))
            for surface, field in VARIABLES
        ]
    )
    current_normalized = np.clip((current_values - lower) / span, 0.0, 1.0)
    frozen_config = copy.deepcopy(config)
    frozen_config["transient"].update(optimization.get("frozen_transient", {}))
    frozen_config["source_ramp_cycles"] = float(
        optimization.get(
            "frozen_source_ramp_cycles", frozen_config.get("source_ramp_cycles", 10.0)
        )
    )
    plasma = compute_esc_plasma_parameters(
        frozen_config,
        result.electron_density_m3,
        result.electron_temperature_ev,
    )
    evaluations: list[dict[str, Any]] = []

    def objective(normalized: np.ndarray) -> float:
        values = lower + np.clip(normalized, 0.0, 1.0) * span
        candidate = copy.deepcopy(frozen_config)
        _apply_values(candidate, values)
        evaluation = len(evaluations)
        row: dict[str, Any] = {
            "evaluation": evaluation,
            "values": _matching_values(candidate),
        }
        try:
            simulation = run_esc_ngspice(
                candidate,
                plasma,
                round_directory / f"candidate_{evaluation:03d}",
            )
            raw_value, reflections = matching_objective(
                simulation.metrics, reference_impedances_ohm
            )
            cycle_l2_max = float(
                max(
                    simulation.metrics.cycle_l2_voltage_wafer,
                    simulation.metrics.cycle_l2_voltage_focus,
                    simulation.metrics.cycle_l2_current_wafer,
                    simulation.metrics.cycle_l2_current_focus,
                )
            )
            settling = optimization["settling_penalty"]
            cycle_ratio_excess = max(
                0.0,
                cycle_l2_max / float(settling["cycle_l2_tolerance"]) - 1.0,
            )
            penalty = float(settling["weight"]) * cycle_ratio_excess**2
            value = raw_value + penalty
            row.update(
                {
                    "objective": value,
                    "matching_objective": raw_value,
                    "settling_penalty": penalty,
                    "apparent_reflection_magnitude": {
                        surface: float(abs(reflection))
                        for surface, reflection in reflections.items()
                    },
                    "apparent_impedance_ohm": {
                        surface: [
                            apparent_impedance(simulation.metrics, surface).real,
                            apparent_impedance(simulation.metrics, surface).imag,
                        ]
                        for surface in SURFACES
                    },
                    "cycle_l2_max": cycle_l2_max,
                }
            )
        except Exception as error:
            value = 1.0e6
            row.update(
                {
                    "objective": value,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        evaluations.append(row)
        print(
            f"round={round_directory.name} eval={evaluation:03d} "
            f"objective={value:.6g}",
            flush=True,
        )
        return value

    analytic_objective = objective(analytic_normalized)
    current_objective = objective(current_normalized)
    x0 = (
        analytic_normalized
        if analytic_objective <= current_objective
        else current_normalized
    )
    start_source = "analytic" if analytic_objective <= current_objective else "current"
    options = optimization["optimizer"]
    optimized = minimize(
        objective,
        x0,
        method="Powell",
        bounds=[(0.0, 1.0)] * len(VARIABLES),
        options={
            "maxfev": int(options["max_function_evaluations"]),
            "xtol": float(options["normalized_parameter_tolerance"]),
            "ftol": float(options["objective_tolerance"]),
            "disp": False,
        },
    )
    best = min(evaluations, key=lambda row: row["objective"])
    best_values = np.asarray(
        [best["values"][f"{surface}_{field}"] for surface, field in VARIABLES]
    )
    solver = {
        "success": bool(optimized.success),
        "message": str(optimized.message),
        "function_evaluations": int(optimized.nfev),
        "reported_objective": float(optimized.fun),
        "selected_best_evaluation": int(best["evaluation"]),
        "analytic_initialization": analytic,
        "start_source": start_source,
        "analytic_start_objective": analytic_objective,
        "current_start_objective": current_objective,
        "initial_values": {
            f"{surface}_{field}": float(value)
            for value, (surface, field) in zip(initial, VARIABLES, strict=True)
        },
    }
    return best_values, evaluations, solver


def run_matching_optimization(
    optimization_path: Path,
    raw_directory: Path,
    summary_output: Path,
) -> dict[str, Any]:
    """Run analytic initialization, frozen-plasma tuning, and coupled verification."""
    optimization = json.loads(optimization_path.read_text(encoding="utf-8"))
    base_config_path = Path(optimization["base_config"])
    config = load_esc_config(base_config_path)
    for surface in SURFACES:
        config["surfaces"][surface]["series_inductor_quality_factor"] = float(
            optimization["series_inductor_quality_factor"][surface]
        )
    reference_impedances = {
        surface: float(optimization["reference_impedance_ohm"][surface])
        for surface in SURFACES
    }
    baseline = solve_esc_model(config, raw_directory / "baseline")
    output: dict[str, Any] = {
        "description": optimization["description"],
        "base_config": str(base_config_path),
        "optimization_config": str(optimization_path),
        "reference_impedance_ohm": reference_impedances,
        "baseline_matching_values": _matching_values(config),
        "baseline": _metrics_summary(baseline, reference_impedances),
        "rounds": [],
    }
    current_result = baseline
    current_config = config
    for round_index in range(int(optimization["outer_rounds"])):
        round_directory = raw_directory / f"round_{round_index + 1:02d}"
        _values, evaluations, solver = _frozen_optimize(
            current_config,
            current_result,
            optimization,
            round_directory,
            reference_impedances,
        )
        successful_evaluations = sorted(
            (row for row in evaluations if "error" not in row),
            key=lambda row: row["objective"],
        )
        verification_attempts: list[dict[str, Any]] = []
        verified: EscCoupledResult | None = None
        selected_config: dict[str, Any] | None = None
        seen_values: set[tuple[float, ...]] = set()
        max_candidates = int(optimization.get("verification_candidates", 5))
        for row in successful_evaluations:
            values = tuple(
                float(row["values"][f"{surface}_{field}"])
                for surface, field in VARIABLES
            )
            if values in seen_values:
                continue
            seen_values.add(values)
            candidate_config = copy.deepcopy(current_config)
            _apply_values(candidate_config, values)
            candidate_config["transient"].update(
                optimization.get("verification_transient", {})
            )
            candidate_config["coupling"].update(
                optimization.get("verification_coupling", {})
            )
            candidate_config["source_ramp_cycles"] = float(
                optimization.get(
                    "verification_source_ramp_cycles",
                    candidate_config.get("source_ramp_cycles", 10.0),
                )
            )
            candidate_config["source_ramp_retry_cycles"] = [
                float(value)
                for value in optimization.get(
                    "verification_source_ramp_retry_cycles", []
                )
            ]
            attempt: dict[str, Any] = {
                "frozen_evaluation": int(row["evaluation"]),
                "frozen_objective": float(row["objective"]),
                "matching_values": _matching_values(candidate_config),
            }
            try:
                candidate_result = solve_esc_model(
                    candidate_config,
                    round_directory
                    / f"coupled_candidate_{int(row['evaluation']):03d}",
                    initial_density_m3=current_result.electron_density_m3,
                )
                attempt["converged"] = bool(candidate_result.converged)
                attempt["density_iterations"] = len(candidate_result.history)
                verification_attempts.append(attempt)
                if candidate_result.converged:
                    verified = candidate_result
                    selected_config = candidate_config
                    break
            except Exception as error:
                attempt.update(
                    {
                        "converged": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                verification_attempts.append(attempt)
            if len(verification_attempts) >= max_candidates:
                break
        if verified is None or selected_config is None:
            raise RuntimeError(
                f"no self-consistent matching candidate converged in round {round_index + 1}: "
                f"{verification_attempts}"
            )
        current_config = selected_config
        round_summary = {
            "round": round_index + 1,
            "matching_values": _matching_values(current_config),
            "solver": solver,
            "evaluations": evaluations,
            "verification_attempts": verification_attempts,
            "selected_frozen_evaluation": verification_attempts[-1][
                "frozen_evaluation"
            ],
            "coupled_verification": _metrics_summary(verified, reference_impedances),
        }
        output["rounds"].append(round_summary)
        (round_directory / "round_summary.json").write_text(
            json.dumps(round_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        current_result = verified
    output["optimized_matching_values"] = _matching_values(current_config)
    output["optimized"] = _metrics_summary(current_result, reference_impedances)
    output["all_coupled_runs_converged"] = bool(
        baseline.converged
        and all(row["coupled_verification"]["converged"] for row in output["rounds"])
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (raw_directory / "optimized_config.json").write_text(
        json.dumps(current_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def make_matching_figures(output: Mapping[str, Any], figure_directory: Path) -> None:
    """Create convergence and before/after figures for the educational report."""
    figure_directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "font.size": 9})
    fig, axis = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    offset = 0
    for round_row in output["rounds"]:
        evaluations = round_row["evaluations"]
        x = np.arange(offset, offset + len(evaluations))
        best_so_far = np.minimum.accumulate([row["objective"] for row in evaluations])
        axis.semilogy(x, best_so_far, label=f"Round {round_row['round']}")
        offset += len(evaluations)
        axis.axvline(offset - 0.5, color="#b0bec5", linewidth=0.8)
    axis.set(
        xlabel="ngspice candidate evaluation",
        ylabel="Best penalized objective",
        title="Frozen-plasma local matching optimization",
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.savefig(figure_directory / "esc_matching_convergence.png", bbox_inches="tight")
    plt.close(fig)

    baseline = output["baseline"]
    optimized = output["optimized"]
    metrics_before = baseline["metrics"]
    metrics_after = optimized["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    labels = ("Wafer", "Focus ring")
    x = np.arange(2)
    width = 0.34
    axes[0].bar(
        x - width / 2,
        [baseline["apparent_reflection_magnitude"][surface] for surface in SURFACES],
        width,
        label="Before",
    )
    axes[0].bar(
        x + width / 2,
        [optimized["apparent_reflection_magnitude"][surface] for surface in SURFACES],
        width,
        label="After",
    )
    axes[0].set(xticks=x, xticklabels=labels, ylabel=r"Apparent $|\Gamma|$", title="Port matching")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    power_labels = ("Plasma", "Source R", "Coil ESR", "Delivered")
    before_power = (
        metrics_before["absorbed_power_total_w"],
        metrics_before["source_resistor_loss_total_w"],
        metrics_before["matching_inductor_loss_total_w"],
        metrics_before["source_delivered_total_w"],
    )
    after_power = (
        metrics_after["absorbed_power_total_w"],
        metrics_after["source_resistor_loss_total_w"],
        metrics_after["matching_inductor_loss_total_w"],
        metrics_after["source_delivered_total_w"],
    )
    axes[1].bar(x=np.arange(4) - width / 2, height=before_power, width=width, label="Before")
    axes[1].bar(x=np.arange(4) + width / 2, height=after_power, width=width, label="After")
    axes[1].set(xticks=np.arange(4), xticklabels=power_labels, ylabel="Power (W)", title="Power redistribution")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(figure_directory / "esc_matching_before_after.png", bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimization",
        type=Path,
        default=Path("configs/esc_matching_optimization.json"),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("artifacts/esc_wafer_focus_ring/matching_optimization"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/data/esc_matching_optimization.json"),
    )
    parser.add_argument(
        "--figure-directory", type=Path, default=Path("reports/figures")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = run_matching_optimization(
        args.optimization, args.raw_output, args.summary_output
    )
    make_matching_figures(output, args.figure_directory)
    print(
        json.dumps(
            {
                "converged": output["all_coupled_runs_converged"],
                "baseline_objective": output["baseline"]["matching_objective"],
                "optimized_objective": output["optimized"]["matching_objective"],
                "summary": str(args.summary_output),
            }
        ),
        flush=True,
    )
    return 0 if output["all_coupled_runs_converged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
