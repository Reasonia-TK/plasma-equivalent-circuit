"""Sweep focus-ring coupling capacitance in the two-surface ESC model."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from plasma_circuit.esc_model import load_esc_config, solve_esc_model


def focus_capacitance_execution_order(
    capacitances_f: list[float], anchor_f: float
) -> list[float]:
    """Return a continuation order starting at the base configuration value."""
    if anchor_f not in capacitances_f:
        raise ValueError("focus capacitance sweep must include the base-config anchor")
    lower = sorted((value for value in capacitances_f if value < anchor_f), reverse=True)
    higher = sorted(value for value in capacitances_f if value > anchor_f)
    return [anchor_f, *lower, *higher]


def _row(capacitance_f: float, result: Any) -> dict[str, Any]:
    metrics = result.final_simulation.metrics
    cycle_l2_max = max(
        metrics.cycle_l2_voltage_wafer,
        metrics.cycle_l2_voltage_focus,
        metrics.cycle_l2_current_wafer,
        metrics.cycle_l2_current_focus,
    )
    return {
        "focus_capacitance_f": capacitance_f,
        "converged": result.converged,
        "electron_temperature_ev": result.electron_temperature_ev,
        "electron_density_m3": result.electron_density_m3,
        "density_iterations": len(result.history),
        "cycle_l2_max": cycle_l2_max,
        "metrics": asdict(metrics),
    }


def run_focus_capacitance_sweep(
    sweep_path: Path,
    raw_directory: Path,
    summary_output: Path,
) -> dict[str, Any]:
    """Run a self-consistent focus-ring capacitance sweep with continuation."""
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    base_config_path = Path(sweep["base_config"])
    base_config = load_esc_config(base_config_path)
    base_config["coupling"].update(sweep.get("coupling_overrides", {}))
    capacitances = [float(value) for value in sweep["focus_capacitances_f"]]
    anchor = float(base_config["surfaces"]["focus_ring"]["coupling_capacitance_f"])
    execution_order = focus_capacitance_execution_order(capacitances, anchor)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    previous_density: float | None = None
    for capacitance in execution_order:
        config = copy.deepcopy(base_config)
        config["surfaces"]["focus_ring"]["coupling_capacitance_f"] = capacitance
        label = f"focus_{capacitance * 1e12:g}_pF".replace(".", "p")
        try:
            result = solve_esc_model(
                config,
                raw_directory / label,
                initial_density_m3=previous_density,
            )
            row = _row(capacitance, result)
            results.append(row)
            previous_density = result.electron_density_m3
            (raw_directory / label / "summary.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"{label}: converged={result.converged} "
                f"ne={result.electron_density_m3:.6e} "
                f"Vsf={result.final_simulation.metrics.mean_sheath_focus_v:.6g} "
                f"Pfocus={result.final_simulation.metrics.absorbed_power_focus_port_w:.6g}",
                flush=True,
            )
        except Exception as error:  # preserve the remaining sensitivity diagnostics
            failure = {
                "focus_capacitance_f": capacitance,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            print(f"{label}: FAILED {type(error).__name__}: {error}", flush=True)
    results.sort(key=lambda item: item["focus_capacitance_f"])
    output = {
        "description": sweep["description"],
        "base_config": str(base_config_path),
        "focus_capacitances_f": capacitances,
        "capacitance_scale": base_config["regularization"]["capacitance_scale"],
        "results": results,
        "nonconverged_capacitances_f": [
            row["focus_capacitance_f"]
            for row in results
            if not row["converged"]
        ],
        "failures": failures,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def make_focus_capacitance_figure(
    output: Mapping[str, Any], figure_directory: Path
) -> None:
    """Plot the electrical and plasma response to focus-ring capacitance."""
    figure_directory.mkdir(parents=True, exist_ok=True)
    rows = output["results"]
    capacitance_pf = np.asarray([row["focus_capacitance_f"] * 1e12 for row in rows])
    metrics = [row["metrics"] for row in rows]
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "font.size": 8})
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), constrained_layout=True)
    axes[0, 0].plot(
        capacitance_pf,
        [item["mean_sheath_wafer_v"] for item in metrics],
        marker="o",
        label="Wafer",
    )
    axes[0, 0].plot(
        capacitance_pf,
        [item["mean_sheath_focus_v"] for item in metrics],
        marker="s",
        label="Focus ring",
    )
    axes[0, 0].plot(
        capacitance_pf,
        [item["mean_sheath_ground_v"] for item in metrics],
        marker="^",
        label="Ground",
    )
    axes[0, 0].set(ylabel="Mean sheath voltage (V)", title="Sheath voltages")
    axes[0, 0].legend()
    axes[0, 1].plot(
        capacitance_pf,
        [item["surface_voltage_wafer_amplitude_v"] for item in metrics],
        marker="o",
        label="Wafer",
    )
    axes[0, 1].plot(
        capacitance_pf,
        [item["surface_voltage_focus_amplitude_v"] for item in metrics],
        marker="s",
        label="Focus ring",
    )
    axes[0, 1].set(ylabel="Surface-voltage amplitude (V)", title="Surface coupling")
    axes[0, 1].legend()
    axes[1, 0].plot(
        capacitance_pf,
        [item["absorbed_power_wafer_port_w"] for item in metrics],
        marker="o",
        label="Wafer port",
    )
    axes[1, 0].plot(
        capacitance_pf,
        [item["absorbed_power_focus_port_w"] for item in metrics],
        marker="s",
        label="Focus port",
    )
    axes[1, 0].set(ylabel="Absorbed power (W)", title="Power partition")
    axes[1, 0].legend()
    axes[1, 1].plot(
        capacitance_pf,
        [row["electron_density_m3"] / 1e15 for row in rows],
        marker="o",
        color="#6a1b9a",
    )
    axes[1, 1].set(ylabel="Electron density (1e15 m-3)", title="Global response")
    for axis in axes.flat:
        axis.set(xlabel="Focus-ring coupling capacitance (pF)", xscale="log")
        axis.grid(alpha=0.25)
    fig.savefig(
        figure_directory / "esc_focus_capacitance_sensitivity.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        type=Path,
        default=Path("configs/esc_focus_capacitance_sweep.json"),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("artifacts/esc_wafer_focus_ring/focus_capacitance_sweep"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/data/esc_focus_capacitance_sweep.json"),
    )
    parser.add_argument(
        "--figure-directory",
        type=Path,
        default=Path("reports/figures"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = run_focus_capacitance_sweep(
        args.sweep,
        args.raw_output,
        args.summary_output,
    )
    make_focus_capacitance_figure(output, args.figure_directory)
    print(
        json.dumps(
            {
                "conditions": len(output["results"]),
                "nonconverged": len(output["nonconverged_capacitances_f"]),
                "failures": len(output["failures"]),
                "summary": str(args.summary_output),
            }
        ),
        flush=True,
    )
    return (
        0
        if not output["failures"] and not output["nonconverged_capacitances_f"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
