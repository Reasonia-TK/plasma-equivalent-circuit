"""Constrained wafer/focus ion-flux uniformity optimization."""

from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from plasma_circuit.esc_optimization import reflection_coefficient
from plasma_circuit.esc_two_zone import (
    TwoZoneMetrics,
    ZoneState,
    compute_two_zone_parameters,
    load_two_zone_config,
    run_two_zone_ngspice,
)
from plasma_circuit.esc_two_zone_coupled import (
    TwoZoneCoupledResult,
    coupled_result_to_dict,
    solve_two_zone_coupled_model,
)


VARIABLE_NAMES = (
    "focus_source_amplitude_v",
    "focus_source_phase_deg",
    "focus_coupling_capacitance_f",
)


def design_from_config(config: Mapping[str, Any]) -> dict[str, float]:
    focus = config["surfaces"]["focus_ring"]
    return {
        "focus_source_amplitude_v": float(focus["source_amplitude_v"]),
        "focus_source_phase_deg": float(focus["source_phase_deg"]),
        "focus_coupling_capacitance_f": float(focus["coupling_capacitance_f"]),
    }


def apply_design(config: dict[str, Any], design: Mapping[str, float]) -> None:
    focus = config["surfaces"]["focus_ring"]
    focus["source_amplitude_v"] = float(design["focus_source_amplitude_v"])
    focus["source_phase_deg"] = float(design["focus_source_phase_deg"])
    focus["coupling_capacitance_f"] = float(
        design["focus_coupling_capacitance_f"]
    )


def _bounds(optimization: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([float(optimization["bounds"][name][0]) for name in VARIABLE_NAMES])
    upper = np.array([float(optimization["bounds"][name][1]) for name in VARIABLE_NAMES])
    if np.any(upper <= lower):
        raise ValueError("uniformity-optimization bounds must have positive spans")
    return lower, upper


def _design_to_values(design: Mapping[str, float]) -> np.ndarray:
    return np.array([float(design[name]) for name in VARIABLE_NAMES])


def _values_to_design(values: Sequence[float]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(VARIABLE_NAMES, values, strict=True)}


def generate_initial_designs(
    config: Mapping[str, Any], optimization: Mapping[str, Any]
) -> list[dict[str, float]]:
    """Return baseline, axial bounds, and deterministic Sobol candidates."""
    lower, upper = _bounds(optimization)
    baseline = np.clip(_design_to_values(design_from_config(config)), lower, upper)
    values: list[np.ndarray] = [baseline]
    for dimension in range(len(VARIABLE_NAMES)):
        for edge in (lower[dimension], upper[dimension]):
            candidate = baseline.copy()
            candidate[dimension] = edge
            values.append(candidate)
    power = int(optimization.get("sobol_power", 4))
    sampler = qmc.Sobol(
        d=len(VARIABLE_NAMES),
        scramble=True,
        seed=int(optimization.get("sobol_seed", 20260902)),
    )
    normalized = sampler.random_base2(power)
    values.extend(lower + row * (upper - lower) for row in normalized)
    return _deduplicate_designs(_values_to_design(row) for row in values)


def generate_local_designs(
    center: Mapping[str, float],
    optimization: Mapping[str, Any],
    step_fraction: float | None = None,
) -> list[dict[str, float]]:
    lower, upper = _bounds(optimization)
    center_values = np.clip(_design_to_values(center), lower, upper)
    fraction = float(
        optimization.get("local_step_fraction", 0.06)
        if step_fraction is None
        else step_fraction
    )
    step = fraction * (upper - lower)
    if bool(optimization.get("local_full_factorial", False)):
        values = [
            np.clip(
                center_values + np.asarray(directions, dtype=float) * step,
                lower,
                upper,
            )
            for directions in product((-1.0, 0.0, 1.0), repeat=len(VARIABLE_NAMES))
        ]
    else:
        values = [center_values]
        for dimension in range(len(VARIABLE_NAMES)):
            for direction in (-1.0, 1.0):
                candidate = center_values.copy()
                candidate[dimension] = np.clip(
                    candidate[dimension] + direction * step[dimension],
                    lower[dimension],
                    upper[dimension],
                )
                values.append(candidate)
    return _deduplicate_designs(_values_to_design(row) for row in values)


def _deduplicate_designs(
    designs: Sequence[Mapping[str, float]] | Any,
) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()
    for design in designs:
        values = tuple(float(design[name]) for name in VARIABLE_NAMES)
        key = tuple(round(value, 15) for value in values)
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(zip(VARIABLE_NAMES, values, strict=True)))
    return output


def apparent_reflections(metrics: TwoZoneMetrics) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, suffix in (("wafer", "wafer"), ("focus_ring", "focus")):
        impedance = complex(
            getattr(metrics, f"input_impedance_{suffix}_real_ohm"),
            getattr(metrics, f"input_impedance_{suffix}_imag_ohm"),
        )
        output[name] = float(abs(reflection_coefficient(impedance, 50.0)))
    return output


def power_density_nonuniformity(
    config: Mapping[str, Any], metrics: TwoZoneMetrics
) -> float:
    densities = np.array(
        [
            metrics.absorbed_power_wafer_allocated_w
            / float(config["zones"]["wafer"]["volume_m3"]),
            metrics.absorbed_power_focus_allocated_w
            / float(config["zones"]["focus_ring"]["volume_m3"]),
        ]
    )
    return float(abs(densities[0] - densities[1]) / np.mean(densities))


def constraint_assessment(
    metrics: TwoZoneMetrics,
    reference_total_power_w: float,
    constraints: Mapping[str, Any],
) -> dict[str, Any]:
    reflections = apparent_reflections(metrics)
    total_deviation = abs(metrics.absorbed_power_total_w / reference_total_power_w - 1.0)
    powered_sheath_max = max(
        metrics.mean_sheath_wafer_v, metrics.mean_sheath_focus_v
    )
    violations = {
        "total_power": max(
            0.0,
            total_deviation / float(constraints["total_power_relative_change_max"]) - 1.0,
        ),
        "powered_sheath": max(
            0.0,
            powered_sheath_max / float(constraints["powered_sheath_mean_max_v"]) - 1.0,
        ),
        "focus_port_power": max(
            0.0,
            (
                float(constraints["focus_port_power_min_w"])
                - metrics.absorbed_power_focus_port_w
            )
            / max(float(constraints["focus_port_power_min_w"]), 1.0e-12),
        ),
        "wafer_allocated_power": max(
            0.0,
            (
                float(constraints["local_allocated_power_min_w"])
                - metrics.absorbed_power_wafer_allocated_w
            )
            / float(constraints["local_allocated_power_min_w"]),
        ),
        "focus_allocated_power": max(
            0.0,
            (
                float(constraints["local_allocated_power_min_w"])
                - metrics.absorbed_power_focus_allocated_w
            )
            / float(constraints["local_allocated_power_min_w"]),
        ),
        "cycle_l2": max(
            0.0,
            metrics.cycle_l2_max / float(constraints["cycle_l2_max"]) - 1.0,
        ),
        "wafer_reflection": max(
            0.0,
            reflections["wafer"] / float(constraints["reflection_magnitude_max"]["wafer"])
            - 1.0,
        ),
        "focus_reflection": max(
            0.0,
            reflections["focus_ring"]
            / float(constraints["reflection_magnitude_max"]["focus_ring"])
            - 1.0,
        ),
    }
    return {
        "feasible": all(value <= 0.0 for value in violations.values()),
        "violations": violations,
        "total_power_relative_change": float(total_deviation),
        "powered_sheath_mean_max_v": float(powered_sheath_max),
        "apparent_reflection_magnitude": reflections,
    }


def _penalized_objective(
    raw_objective: float,
    assessment: Mapping[str, Any],
    penalty_weight: float,
) -> float:
    penalty = penalty_weight * sum(
        float(value) ** 2 for value in assessment["violations"].values()
    )
    return float(raw_objective + penalty)


def _evaluate_frozen_candidate(
    candidate_id: int,
    design: Mapping[str, float],
    config: Mapping[str, Any],
    states: Mapping[str, ZoneState],
    reference_total_power_w: float,
    optimization: Mapping[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    candidate = copy.deepcopy(config)
    apply_design(candidate, design)
    candidate["transient"].update(optimization.get("frozen_transient", {}))
    row: dict[str, Any] = {"candidate_id": candidate_id, "design": dict(design)}
    try:
        plasma = compute_two_zone_parameters(
            candidate,
            states,
            electrical_coupling_scale=float(
                candidate["self_consistent"].get("electrical_coupling_scale", 1.0)
            ),
        )
        simulation = run_two_zone_ngspice(
            candidate, plasma, output_directory / f"candidate_{candidate_id:03d}"
        )
        raw_objective = power_density_nonuniformity(candidate, simulation.metrics)
        assessment = constraint_assessment(
            simulation.metrics,
            reference_total_power_w,
            optimization["constraints"],
        )
        row.update(
            {
                "raw_power_density_nonuniformity": raw_objective,
                "objective": _penalized_objective(
                    raw_objective,
                    assessment,
                    float(optimization.get("penalty_weight", 1.0)),
                ),
                "constraints": assessment,
                "metrics": asdict(simulation.metrics),
            }
        )
    except Exception as error:
        row.update(
            {
                "objective": float(optimization.get("failure_objective", 1.0e6)),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    return row


def _run_frozen_batch(
    designs: list[dict[str, float]],
    starting_id: int,
    config: Mapping[str, Any],
    states: Mapping[str, ZoneState],
    reference_total_power_w: float,
    optimization: Mapping[str, Any],
    output_directory: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    workers = int(optimization.get("parallel_workers", 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _evaluate_frozen_candidate,
                starting_id + offset,
                design,
                config,
                states,
                reference_total_power_w,
                optimization,
                output_directory,
            ): starting_id + offset
            for offset, design in enumerate(designs)
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"frozen candidate={row['candidate_id']:03d} "
                f"objective={row['objective']:.6g}",
                flush=True,
            )
    return sorted(rows, key=lambda row: int(row["candidate_id"]))


def _coupled_candidate_summary(
    design: Mapping[str, float],
    result: TwoZoneCoupledResult,
    reference_total_power_w: float,
    optimization: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = result.final_simulation.metrics
    assessment = constraint_assessment(
        metrics, reference_total_power_w, optimization["constraints"]
    )
    raw_objective = result.uniformity.ion_flux_nonuniformity
    return {
        "design": dict(design),
        "raw_ion_flux_nonuniformity": raw_objective,
        "objective": _penalized_objective(
            raw_objective,
            assessment,
            float(optimization.get("penalty_weight", 1.0)),
        ),
        "constraints": assessment,
        "result": coupled_result_to_dict(result),
    }


def run_uniformity_optimization(
    optimization_path: Path,
    raw_directory: Path,
    summary_output: Path,
) -> dict[str, Any]:
    optimization = json.loads(optimization_path.read_text(encoding="utf-8"))
    config = load_two_zone_config(optimization["base_config"])
    baseline = solve_two_zone_coupled_model(config, raw_directory / "baseline")
    if not baseline.converged:
        raise RuntimeError("uniformity-optimization baseline did not converge")
    reference_total_power = baseline.final_simulation.metrics.absorbed_power_total_w
    baseline_design = design_from_config(config)
    baseline_summary = _coupled_candidate_summary(
        baseline_design, baseline, reference_total_power, optimization
    )

    initial_designs = generate_initial_designs(config, optimization)
    frozen_rows = _run_frozen_batch(
        initial_designs,
        0,
        config,
        baseline.states,
        reference_total_power,
        optimization,
        raw_directory / "frozen_initial",
    )
    successful = [row for row in frozen_rows if "error" not in row]
    if not successful:
        raise RuntimeError("all initial frozen-plasma uniformity candidates failed")
    best_initial = min(successful, key=lambda row: float(row["objective"]))
    existing_keys = {
        tuple(round(float(row["design"][name]), 15) for name in VARIABLE_NAMES)
        for row in frozen_rows
    }
    local_center = best_initial["design"]
    step_fractions = optimization.get(
        "local_step_fractions",
        [float(optimization.get("local_step_fraction", 0.06))],
    )
    for stage_index, raw_fraction in enumerate(step_fractions):
        local_designs = [
            design
            for design in generate_local_designs(
                local_center, optimization, float(raw_fraction)
            )
            if tuple(round(float(design[name]), 15) for name in VARIABLE_NAMES)
            not in existing_keys
        ]
        for design in local_designs:
            existing_keys.add(
                tuple(round(float(design[name]), 15) for name in VARIABLE_NAMES)
            )
        frozen_rows.extend(
            _run_frozen_batch(
                local_designs,
                len(frozen_rows),
                config,
                baseline.states,
                reference_total_power,
                optimization,
                raw_directory / f"frozen_local_{stage_index + 1:02d}",
            )
        )
        local_center = min(
            (row for row in frozen_rows if "error" not in row),
            key=lambda row: (
                not bool(row["constraints"]["feasible"]),
                float(row["objective"]),
            ),
        )["design"]
    ranked = sorted(
        (row for row in frozen_rows if "error" not in row),
        key=lambda row: (not bool(row["constraints"]["feasible"]), float(row["objective"])),
    )
    verification_count = int(optimization.get("verification_candidates", 4))
    verification_rows: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    baseline_key = tuple(float(baseline_design[name]) for name in VARIABLE_NAMES)
    for row in ranked:
        design = row["design"]
        key = tuple(float(design[name]) for name in VARIABLE_NAMES)
        if key in seen:
            continue
        seen.add(key)
        candidate = copy.deepcopy(config)
        apply_design(candidate, design)
        candidate["self_consistent"].update(optimization.get("verification_self_consistent", {}))
        verification: dict[str, Any] = {
            "frozen_candidate_id": int(row["candidate_id"]),
            "frozen_objective": float(row["objective"]),
            "design": design,
        }
        try:
            result = (
                baseline
                if key == baseline_key
                else solve_two_zone_coupled_model(
                    candidate,
                    raw_directory / f"coupled_candidate_{int(row['candidate_id']):03d}",
                    initial_states=baseline.states,
                )
            )
            verification.update(
                _coupled_candidate_summary(
                    design, result, reference_total_power, optimization
                )
            )
        except Exception as error:
            verification.update(
                {
                    "objective": float(optimization.get("failure_objective", 1.0e6)),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        verification_rows.append(verification)
        print(
            f"coupled candidate={row['candidate_id']:03d} "
            f"objective={verification['objective']:.6g}",
            flush=True,
        )
        if len(verification_rows) >= verification_count:
            break
    valid_verifications = [
        row
        for row in verification_rows
        if "error" not in row and bool(row["result"]["converged"])
    ]
    if not valid_verifications:
        raise RuntimeError("no self-consistent uniformity candidate converged")
    baseline_verification = {
        "frozen_candidate_id": None,
        "frozen_objective": None,
        **baseline_summary,
    }
    optimized = min(
        [baseline_verification, *valid_verifications],
        key=lambda row: (not bool(row["constraints"]["feasible"]), float(row["objective"])),
    )
    optimized_config = copy.deepcopy(config)
    apply_design(optimized_config, optimized["design"])
    optimized_config["self_consistent"].update(
        optimization.get("final_confirmation_self_consistent", {})
    )
    confirmation_initial_states = {
        name: ZoneState(**state)
        for name, state in optimized["result"]["states"].items()
    }
    confirmation_result = solve_two_zone_coupled_model(
        optimized_config,
        raw_directory / "final_confirmation",
        initial_states=confirmation_initial_states,
    )
    if not confirmation_result.converged:
        raise RuntimeError(
            "selected uniformity design failed strict final confirmation: "
            f"residual={confirmation_result.final_balance.max_normalized_residual:.3e}"
        )
    confirmed_optimized = {
        "selected_frozen_candidate_id": optimized.get("frozen_candidate_id"),
        "screening_objective": optimized["objective"],
        **_coupled_candidate_summary(
            optimized["design"],
            confirmation_result,
            reference_total_power,
            optimization,
        ),
    }
    output = {
        "description": optimization["description"],
        "optimization_config": str(optimization_path),
        "base_config": str(optimization["base_config"]),
        "baseline": baseline_summary,
        "frozen_candidates": frozen_rows,
        "coupled_verifications": verification_rows,
        "selected_verification": optimized,
        "optimized": confirmed_optimized,
        "improvement_fraction": float(
            1.0
            - confirmed_optimized["raw_ion_flux_nonuniformity"]
            / baseline_summary["raw_ion_flux_nonuniformity"]
        ),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    raw_directory.mkdir(parents=True, exist_ok=True)
    (raw_directory / "optimized_config.json").write_text(
        json.dumps(optimized_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def make_uniformity_figures(output: Mapping[str, Any], figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    frozen = [row for row in output["frozen_candidates"] if "error" not in row]
    x = np.arange(len(frozen))
    raw = 100.0 * np.array(
        [row["raw_power_density_nonuniformity"] for row in frozen]
    )
    best = np.minimum.accumulate(
        [row["objective"] for row in frozen]
    ) * 100.0
    baseline = output["baseline"]
    optimized = output["optimized"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    axes[0].scatter(x, raw, s=20, alpha=0.65, label="candidate")
    axes[0].plot(x, best, color="#d55e00", label="best penalized")
    axes[0].set_xlabel("Frozen-plasma candidate")
    axes[0].set_ylabel("Local power-density nonuniformity (%)")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    labels = ("Ion flux", "Density", r"$T_e$")
    before = 100.0 * np.array(
        [
            baseline["result"]["uniformity"]["ion_flux_nonuniformity"],
            baseline["result"]["uniformity"]["density_nonuniformity"],
            baseline["result"]["uniformity"]["temperature_nonuniformity"],
        ]
    )
    after = 100.0 * np.array(
        [
            optimized["result"]["uniformity"]["ion_flux_nonuniformity"],
            optimized["result"]["uniformity"]["density_nonuniformity"],
            optimized["result"]["uniformity"]["temperature_nonuniformity"],
        ]
    )
    positions = np.arange(3)
    width = 0.36
    axes[1].bar(positions - width / 2, before, width, label="Before")
    axes[1].bar(positions + width / 2, after, width, label="After")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Wafer-Focus nonuniformity (%)")
    axes[1].set_yscale("log")
    axes[1].grid(True, axis="y", which="both", alpha=0.25)
    axes[1].legend(fontsize=8)

    design_before = baseline["design"]
    design_after = optimized["design"]
    design_labels = ("Amplitude (V peak)", "Phase (deg)", "ESC C (pF)")
    before_design = np.array(
        [
            design_before["focus_source_amplitude_v"],
            design_before["focus_source_phase_deg"],
            design_before["focus_coupling_capacitance_f"] * 1.0e12,
        ]
    )
    after_design = np.array(
        [
            design_after["focus_source_amplitude_v"],
            design_after["focus_source_phase_deg"],
            design_after["focus_coupling_capacitance_f"] * 1.0e12,
        ]
    )
    axes[2].axis("off")
    cell_text = [
        [
            f"{before_design[index]:.5g}",
            f"{after_design[index]:.5g}",
            f"{after_design[index] - before_design[index]:+.4g}",
        ]
        for index in range(3)
    ]
    table = axes[2].table(
        cellText=cell_text,
        rowLabels=design_labels,
        colLabels=("Before", "After", "Change"),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    axes[2].set_title("Selected focus-electrode design", pad=18)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimization",
        type=Path,
        default=Path("configs/esc_uniformity_optimization.json"),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("artifacts/esc_two_zone/uniformity_optimization"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/data/esc_uniformity_optimization.json"),
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("reports/figures/esc_uniformity_optimization.png"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = run_uniformity_optimization(
        args.optimization, args.raw_output, args.summary_output
    )
    make_uniformity_figures(output, args.figure_output)
    print(
        json.dumps(
            {
                "baseline_ion_flux_nonuniformity": output["baseline"][
                    "raw_ion_flux_nonuniformity"
                ],
                "optimized_ion_flux_nonuniformity": output["optimized"][
                    "raw_ion_flux_nonuniformity"
                ],
                "improvement_fraction": output["improvement_fraction"],
                "optimized_design": output["optimized"]["design"],
                "feasible": output["optimized"]["constraints"]["feasible"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
