"""Generate a self-consistent pressure-voltage prediction map."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from plasma_circuit.config import load_config
from plasma_circuit.coupling import coupled_result_to_dict, solve_self_consistent_density


def fundamental_reflection(input_impedance: complex, reference_ohm: float) -> complex:
    """Return the fundamental voltage-wave reflection coefficient."""
    return (input_impedance - reference_ohm) / (input_impedance + reference_ohm)


def _condition_label(pressure_pa: float, amplitude_v: float) -> str:
    pressure = f"{pressure_pa:g}".replace(".", "p")
    amplitude = f"{amplitude_v:g}".replace(".", "p")
    return f"p_{pressure}_v_{amplitude}"


def _result_row(
    pressure_pa: float,
    amplitude_v: float,
    config: Mapping[str, Any],
    result: Any,
    retry_stage: str | None,
) -> dict[str, Any]:
    metrics = result.final_simulation.metrics
    reference = float(config["source_resistance_ohm"])
    impedance = complex(
        metrics.input_impedance_real_ohm, metrics.input_impedance_imag_ohm
    )
    reflection = fundamental_reflection(impedance, reference)
    available_power = amplitude_v**2 / (8.0 * reference)
    load_power = metrics.source_delivered_power_w - metrics.source_resistor_loss_w
    return {
        "pressure_pa": pressure_pa,
        "source_amplitude_v": amplitude_v,
        "electron_temperature_ev": result.electron_temperature_ev,
        "electron_density_m3": result.electron_density_m3,
        "converged": result.converged,
        "retried": retry_stage is not None,
        "retry_stage": retry_stage,
        "transient_cycles": int(config["transient"]["cycles"]),
        "density_iterations": len(result.history),
        "available_power_w": available_power,
        "load_power_w": load_power,
        "plasma_efficiency_available": metrics.absorbed_power_w / available_power,
        "load_efficiency_available": load_power / available_power,
        "reflection_real": float(reflection.real),
        "reflection_imag": float(reflection.imag),
        "reflection_magnitude": float(abs(reflection)),
        "metrics": asdict(metrics),
    }


def run_operating_map(
    sweep_path: Path,
    raw_directory: Path,
    summary_output: Path,
) -> dict[str, Any]:
    """Run every map point with continuation in pressure and source amplitude."""
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    base_config_path = Path(sweep["base_config"])
    config = load_config(base_config_path)
    config["coupling"].update(sweep.get("coupling_overrides", {}))
    pressures = [float(value) for value in sweep["pressures_pa"]]
    amplitudes = [float(value) for value in sweep["source_amplitudes_v"]]
    if 100.0 not in amplitudes:
        raise ValueError("continuation map requires a 100 V anchor in source_amplitudes_v")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    previous_anchor_density: float | None = None
    for pressure in pressures:
        row_densities: dict[float, float] = {}
        execution_order = [100.0]
        execution_order.extend(
            sorted((value for value in amplitudes if value < 100.0), reverse=True)
        )
        execution_order.extend(sorted(value for value in amplitudes if value > 100.0))
        for amplitude in execution_order:
            condition = copy.deepcopy(config)
            condition["pressure_pa"] = pressure
            condition["source_amplitude_v"] = amplitude
            if amplitude == 100.0:
                initial_density = previous_anchor_density
            elif amplitude < 100.0:
                initial_density = row_densities.get(100.0)
            else:
                lower = [value for value in row_densities if value < amplitude]
                initial_density = row_densities[max(lower)] if lower else row_densities.get(100.0)
            label = _condition_label(pressure, amplitude)
            try:
                result = solve_self_consistent_density(
                    condition,
                    raw_directory / label,
                    initial_density_m3=initial_density,
                )
                result_condition = condition
                retry_stage: str | None = None
                if not result.converged:
                    retry_stage = "density_iterations"
                    retry_condition = copy.deepcopy(condition)
                    retry_condition["coupling"]["max_iterations"] = 2 * int(
                        condition["coupling"]["max_iterations"]
                    )
                    result = solve_self_consistent_density(
                        retry_condition,
                        raw_directory / label / "retry",
                        initial_density_m3=result.electron_density_m3,
                    )
                    result_condition = retry_condition
                if not result.converged:
                    retry_stage = "settling_time"
                    settling_condition = copy.deepcopy(condition)
                    settling_condition["transient"]["cycles"] = (
                        3 * int(condition["transient"]["cycles"]) + 1
                    ) // 2
                    result = solve_self_consistent_density(
                        settling_condition,
                        raw_directory / label / "settling_retry",
                        initial_density_m3=result.electron_density_m3,
                    )
                    result_condition = settling_condition
                row = _result_row(
                    pressure,
                    amplitude,
                    result_condition,
                    result,
                    retry_stage,
                )
                results.append(row)
                row_densities[amplitude] = result.electron_density_m3
                if amplitude == 100.0:
                    previous_anchor_density = result.electron_density_m3
                condition_summary = coupled_result_to_dict(result)
                condition_summary["map_metrics"] = {
                    key: value for key, value in row.items() if key != "metrics"
                }
                (raw_directory / label / "summary.json").write_text(
                    json.dumps(condition_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"{label}: converged={result.converged} "
                    f"ne={result.electron_density_m3:.6e} "
                    f"Ppl={result.final_simulation.metrics.absorbed_power_w:.6g} "
                    f"|Gamma|={row['reflection_magnitude']:.6g}",
                    flush=True,
                )
            except Exception as error:  # keep the rest of the map diagnosable
                failure = {
                    "pressure_pa": pressure,
                    "source_amplitude_v": amplitude,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                print(f"{label}: FAILED {type(error).__name__}: {error}", flush=True)

    results.sort(key=lambda row: (row["pressure_pa"], row["source_amplitude_v"]))
    output = {
        "description": sweep["description"],
        "base_config": str(base_config_path),
        "network_mode": sweep["network_mode"],
        "coupling_overrides": sweep.get("coupling_overrides", {}),
        "capacitance_convention": config["regularization"].get("capacitance_convention"),
        "capacitance_scale": config["regularization"]["capacitance_scale"],
        "pressures_pa": pressures,
        "source_amplitudes_v": amplitudes,
        "results": results,
        "nonconverged_conditions": [
            {
                "pressure_pa": row["pressure_pa"],
                "source_amplitude_v": row["source_amplitude_v"],
            }
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


def _matrix(
    output: Mapping[str, Any], value: Any
) -> np.ndarray:
    pressures = output["pressures_pa"]
    amplitudes = output["source_amplitudes_v"]
    lookup = {
        (row["pressure_pa"], row["source_amplitude_v"]): row
        for row in output["results"]
    }
    return np.asarray(
        [
            [
                float(value(lookup[(pressure, amplitude)]))
                if (pressure, amplitude) in lookup
                else np.nan
                for amplitude in amplitudes
            ]
            for pressure in pressures
        ]
    )


def _annotated_heatmap(
    ax: Any,
    values: np.ndarray,
    output: Mapping[str, Any],
    title: str,
    fmt: str,
    cmap: str = "viridis",
) -> None:
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set(
        title=title,
        xticks=np.arange(len(output["source_amplitudes_v"])),
        xticklabels=[f"{value:g}" for value in output["source_amplitudes_v"]],
        yticks=np.arange(len(output["pressures_pa"])),
        yticklabels=[f"{value:g}" for value in output["pressures_pa"]],
        xlabel="Source peak voltage (V)",
        ylabel="Pressure (Pa)",
    )
    finite = values[np.isfinite(values)]
    threshold = float(np.nanmedian(finite)) if finite.size else 0.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                color = "white" if value < threshold else "black"
                ax.text(
                    column,
                    row,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )
    for condition in output.get("nonconverged_conditions", []):
        row = output["pressures_pa"].index(condition["pressure_pa"])
        column = output["source_amplitudes_v"].index(condition["source_amplitude_v"])
        ax.add_patch(
            Rectangle(
                (column - 0.48, row - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor="#ff1744",
                linewidth=1.8,
            )
        )
    plt.colorbar(image, ax=ax, shrink=0.8)


def make_operating_map_figures(
    output: Mapping[str, Any], figure_directory: Path
) -> None:
    figure_directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "font.size": 8})
    plasma_panels = (
        (
            _matrix(output, lambda row: row["electron_density_m3"] / 1e15),
            "Electron density (1e15 m-3)",
            ".2f",
        ),
        (
            _matrix(output, lambda row: row["electron_temperature_ev"]),
            "Electron temperature (eV)",
            ".2f",
        ),
        (
            _matrix(output, lambda row: row["metrics"]["absorbed_power_w"]),
            "Plasma absorbed power (W)",
            ".2f",
        ),
        (
            _matrix(output, lambda row: row["metrics"]["plasma_voltage_offset_v"]),
            "Plasma DC offset (V)",
            ".0f",
            "coolwarm",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.7), constrained_layout=True)
    for ax, panel in zip(axes.flat, plasma_panels, strict=True):
        _annotated_heatmap(ax, panel[0], output, *panel[1:])
    if output.get("nonconverged_conditions"):
        fig.suptitle("Red border: coupling tolerance not met", color="#c62828")
    fig.savefig(figure_directory / "pressure_voltage_plasma_map.png", bbox_inches="tight")
    plt.close(fig)

    circuit_panels = (
        (
            _matrix(output, lambda row: row["reflection_magnitude"]),
            "Fundamental |Gamma|",
            ".2f",
            "magma",
        ),
        (
            _matrix(output, lambda row: 100.0 * row["plasma_efficiency_available"]),
            "Plasma / available power (%)",
            ".1f",
        ),
        (
            _matrix(output, lambda row: row["metrics"]["plasma_current_thd"]),
            "Plasma current THD",
            ".2f",
        ),
        (
            _matrix(
                output,
                lambda row: row["metrics"]["power_balance_relative_error"]
                * 100.0,
            ),
            "Power balance residual (%)",
            ".3f",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.7), constrained_layout=True)
    for ax, panel in zip(axes.flat, circuit_panels, strict=True):
        _annotated_heatmap(ax, panel[0], output, *panel[1:])
    if output.get("nonconverged_conditions"):
        fig.suptitle("Red border: coupling tolerance not met", color="#c62828")
    fig.savefig(figure_directory / "pressure_voltage_circuit_map.png", bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep", type=Path, default=Path("configs/pressure_voltage_sweep.json")
    )
    parser.add_argument(
        "--raw-output", type=Path, default=Path("artifacts/pressure_voltage_map")
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/data/pressure_voltage_map.json"),
    )
    parser.add_argument(
        "--figure-directory", type=Path, default=Path("reports/figures")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = run_operating_map(args.sweep, args.raw_output, args.summary_output)
    make_operating_map_figures(output, args.figure_directory)
    print(
        json.dumps(
            {
                "conditions": len(output["results"]),
                "nonconverged": len(output["nonconverged_conditions"]),
                "failures": len(output["failures"]),
                "summary": str(args.summary_output),
            }
        ),
        flush=True,
    )
    return (
        0
        if not output["failures"] and not output["nonconverged_conditions"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
