"""Run the two-surface wafer/focus-ring ESC plasma model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from plasma_circuit.esc_model import (
    EscCoupledResult,
    esc_result_to_dict,
    load_esc_config,
    solve_esc_model,
)


def make_esc_figures(
    result: EscCoupledResult,
    config: Mapping[str, Any],
    figure_directory: Path,
) -> None:
    """Generate waveform and power/voltage comparison figures."""
    figure_directory.mkdir(parents=True, exist_ok=True)
    make_esc_topology_figure(figure_directory)
    simulation = result.final_simulation
    time_s = simulation.time_s
    waveforms = simulation.waveforms
    frequency = float(config["frequency_hz"])
    end = time_s[-1]
    mask = time_s >= end - 2.0 / frequency
    phase = (time_s[mask] - (end - 2.0 / frequency)) * frequency
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "font.size": 9})

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.0), sharex=True, constrained_layout=True)
    axes[0].plot(
        phase,
        waveforms["v_surface_wafer"][mask],
        label="Wafer surface",
        color="#1565c0",
    )
    axes[0].plot(
        phase,
        waveforms["v_surface_focus"][mask],
        label="Focus-ring surface",
        color="#d84315",
    )
    axes[0].set(ylabel="Surface voltage (V)", xlim=(0.0, 2.0))
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        phase,
        waveforms["i_surface_wafer"][mask],
        label="Wafer branch",
        color="#1565c0",
    )
    axes[1].plot(
        phase,
        waveforms["i_surface_focus"][mask],
        label="Focus-ring branch",
        color="#d84315",
    )
    axes[1].set(xlabel="RF cycles", ylabel="Surface current (A)", xlim=(0.0, 2.0))
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.savefig(figure_directory / "esc_surface_waveforms.png", bbox_inches="tight")
    plt.close(fig)

    metrics = simulation.metrics
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1), constrained_layout=True)
    voltage_labels = ("Wafer\nsheath", "Focus\nsheath", "Ground\nsheath")
    voltage_values = (
        metrics.mean_sheath_wafer_v,
        metrics.mean_sheath_focus_v,
        metrics.mean_sheath_ground_v,
    )
    axes[0].bar(
        voltage_labels,
        voltage_values,
        color=("#1565c0", "#d84315", "#455a64"),
    )
    axes[0].set(
        ylabel="Mean sheath voltage (V)", title="Area-resolved sheath voltage"
    )
    axes[0].grid(axis="y", alpha=0.25)
    power_labels = ("Wafer\nport", "Focus\nport", "Source R", "Dielectric")
    power_values = (
        metrics.absorbed_power_wafer_port_w,
        metrics.absorbed_power_focus_port_w,
        metrics.source_resistor_loss_total_w,
        metrics.dielectric_loss_total_w,
    )
    axes[1].bar(
        power_labels,
        power_values,
        color=("#1565c0", "#d84315", "#6a1b9a", "#00897b"),
    )
    axes[1].set(ylabel="Mean power (W)", title="Power paths")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(figure_directory / "esc_voltage_power.png", bbox_inches="tight")
    plt.close(fig)


def make_esc_topology_figure(figure_directory: Path) -> None:
    """Draw the implemented two-surface equivalent-circuit topology."""
    fig, axis = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    axis.set(xlim=(0.0, 12.0), ylim=(0.0, 6.0))
    axis.axis("off")

    def box(x: float, y: float, width: float, text: str, color: str) -> None:
        axis.add_patch(
            FancyBboxPatch(
                (x, y - 0.38),
                width,
                0.76,
                boxstyle="round,pad=0.04",
                facecolor=color,
                edgecolor="#263238",
                linewidth=1.2,
            )
        )
        axis.text(x + width / 2.0, y, text, ha="center", va="center", fontsize=9)

    def connect(x1: float, y1: float, x2: float, y2: float) -> None:
        axis.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "-", "color": "#37474f", "linewidth": 1.5},
        )

    rows = ((4.7, "Wafer"), (2.8, "Focus ring"))
    for y, surface in rows:
        box(0.3, y, 1.1, f"RF source\n{surface}", "#e3f2fd")
        box(1.9, y, 1.1, "Rsource\nLseries", "#ede7f6")
        box(3.5, y, 1.25, "CESC + ESR\n(Rleak ||)", "#e0f2f1")
        box(5.25, y, 1.2, f"{surface}\nsurface", "#fff3e0")
        box(6.95, y, 1.1, "Nonlinear\nsheath", "#fce4ec")
        box(8.55, y, 1.1, "Bulk\nR-L branch", "#f3e5f5")
        connect(1.4, y, 1.9, y)
        connect(3.0, y, 3.5, y)
        connect(4.75, y, 5.25, y)
        connect(6.45, y, 6.95, y)
        connect(8.05, y, 8.55, y)
        connect(9.65, y, 10.25, 3.75)
    box(10.25, 3.75, 1.25, "Common\nplasma bulk", "#e8eaf6")
    box(10.25, 1.1, 1.25, "Grounded\nsheath", "#eceff1")
    connect(10.875, 3.37, 10.875, 1.48)
    connect(10.875, 0.72, 10.875, 0.25)
    axis.plot([10.45, 11.3], [0.25, 0.25], color="#37474f", linewidth=1.5)
    axis.plot([10.57, 11.18], [0.12, 0.12], color="#37474f", linewidth=1.5)
    axis.text(
        6.0,
        5.65,
        "Two capacitively coupled ESC surfaces sharing one global plasma state",
        ha="center",
        fontsize=13,
        weight="bold",
    )
    fig.savefig(figure_directory / "esc_equivalent_circuit.png", bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/esc_wafer_focus_ring.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/esc_wafer_focus_ring/baseline"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/data/esc_wafer_focus_ring.json"),
    )
    parser.add_argument(
        "--figure-directory",
        type=Path,
        default=Path("reports/figures"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_esc_config(args.config)
    result = solve_esc_model(config, args.output)
    summary = esc_result_to_dict(result)
    summary["config"] = str(args.config)
    summary["description"] = config.get("description", "")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_esc_figures(result, config, args.figure_directory)
    metrics = result.final_simulation.metrics
    print(
        json.dumps(
            {
                "converged": result.converged,
                "electron_temperature_ev": result.electron_temperature_ev,
                "electron_density_m3": result.electron_density_m3,
                "absorbed_power_w": metrics.absorbed_power_total_w,
                "summary": str(args.summary_output),
            }
        ),
        flush=True,
    )
    return 0 if result.converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
