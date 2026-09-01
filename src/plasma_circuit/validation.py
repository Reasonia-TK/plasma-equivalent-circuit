"""Run numerical sensitivities and generate report figures for the reproduction."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from plasma_circuit.config import load_config
from plasma_circuit.ngspice import _parse_wrdata, run_ngspice
from plasma_circuit.physics import compute_plasma_parameters


PAPER_TARGETS: dict[str, float] = {
    "electron_temperature_ev": 4.75,
    "electron_density_m3": 1.25e15,
    "input_impedance_real_ohm": 50.01,
    "input_impedance_imag_ohm": -0.02,
    "generator_current_amplitude_a": 1.0,
    "plasma_current_amplitude_a": 0.6,
    "load_current_amplitude_a": 6.7,
    "stray_current_amplitude_a": 6.1,
    "plasma_voltage_amplitude_v": 360.0,
    "plasma_voltage_offset_v": -250.0,
    "source_resistor_loss_w": 25.0,
    "match_loss_w": 11.0,
    "stray_loss_w": 9.0,
    "absorbed_power_w": 5.0,
}

SELECTED_METRICS = (
    "absorbed_power_w",
    "input_impedance_real_ohm",
    "input_impedance_imag_ohm",
    "plasma_voltage_amplitude_v",
    "plasma_voltage_offset_v",
    "generator_current_amplitude_a",
    "plasma_current_amplitude_a",
    "load_current_amplitude_a",
    "stray_current_amplitude_a",
    "source_resistor_loss_w",
    "match_loss_w",
    "stray_loss_w",
    "cycle_l2_voltage",
    "cycle_l2_current",
    "power_balance_relative_error",
)


def _metric_subset(metrics: Any) -> dict[str, float]:
    values = asdict(metrics)
    return {name: float(values[name]) for name in SELECTED_METRICS}


def _run_sweep_case(
    base_config: Mapping[str, Any],
    density_m3: float,
    temperature_ev: float,
    raw_directory: Path,
    label: str,
    *,
    samples_per_cycle: int | None = None,
    regularization_width_v: float | None = None,
    capacitance_scale: float | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    if samples_per_cycle is not None:
        config["transient"]["samples_per_cycle"] = samples_per_cycle
    if regularization_width_v is not None:
        config["regularization"]["capacitance_voltage_v"] = regularization_width_v
        config["regularization"]["electron_voltage_v"] = regularization_width_v
    if capacitance_scale is not None:
        config["regularization"]["capacitance_scale"] = capacitance_scale
    plasma = compute_plasma_parameters(config, density_m3, temperature_ev)
    result = run_ngspice(config, plasma, raw_directory / label)
    return {
        "label": label,
        "samples_per_cycle": samples_per_cycle,
        "regularization_width_v": regularization_width_v,
        "capacitance_scale": capacitance_scale,
        "metrics": _metric_subset(result.metrics),
    }


def _ngspice_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "-v"], capture_output=True, text=True, timeout=30, check=False
    )
    return (completed.stdout + completed.stderr).strip()


def run_validation(
    config_path: Path,
    base_summary_path: Path,
    raw_directory: Path,
    summary_output: Path,
) -> dict[str, Any]:
    """Run fixed-state numerical sweeps around the converged coupled solution."""
    config = load_config(config_path)
    base = json.loads(base_summary_path.read_text(encoding="utf-8"))
    density = float(base["electron_density_m3"])
    temperature = float(base["electron_temperature_ev"])
    timestep = [
        _run_sweep_case(
            config,
            density,
            temperature,
            raw_directory,
            f"step_{samples}",
            samples_per_cycle=samples,
        )
        for samples in (120, 240, 480)
    ]
    regularization = [
        _run_sweep_case(
            config,
            density,
            temperature,
            raw_directory,
            f"regularization_{width:g}",
            regularization_width_v=width,
        )
        for width in (0.005, 0.05, 0.5)
    ]
    capacitance_convention = [
        _run_sweep_case(
            config,
            float(PAPER_TARGETS["electron_density_m3"]),
            temperature,
            raw_directory,
            f"capacitance_scale_{scale:g}",
            capacitance_scale=scale,
        )
        for scale in (0.5, float(config["regularization"]["capacitance_scale"]), 1.0)
    ]
    output = {
        "paper": "Schmidt, Mussenbrock, Trieschmann, PSST 27 (2018) 105017",
        "config": str(config_path),
        "ngspice_version": _ngspice_version(str(config["ngspice_path"])),
        "paper_targets": PAPER_TARGETS,
        "base_result": base,
        "timestep_sensitivity": timestep,
        "regularization_sensitivity": regularization,
        "capacitance_convention_sensitivity": capacitance_convention,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 9,
        }
    )


def make_figures(
    validation: Mapping[str, Any],
    base_output_directory: Path,
    figure_directory: Path,
) -> None:
    """Generate the compact evidence figures referenced by the report."""
    _set_plot_style()
    figure_directory.mkdir(parents=True, exist_ok=True)
    base = validation["base_result"]
    metrics = base["metrics"]
    last_iteration = int(base["history"][-1]["iteration"])
    waveform_path = base_output_directory / f"density_{last_iteration:02d}" / "waveforms.dat"
    time_s, waveforms = _parse_wrdata(waveform_path)
    frequency = 13.56e6
    start = time_s[-1] - 2.0 / frequency
    mask = time_s >= start
    phase_cycles = (time_s[mask] - start) * frequency

    fig, ax_voltage = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    ax_current = ax_voltage.twinx()
    voltage_line = ax_voltage.plot(
        phase_cycles, waveforms["v_plasma"][mask], color="#1565c0", label="Plasma voltage"
    )[0]
    current_line = ax_current.plot(
        phase_cycles, waveforms["i_plasma_port"][mask], color="#d84315", label="Plasma current"
    )[0]
    ax_voltage.set(xlabel="RF cycles", ylabel="Voltage (V)", xlim=(0, 2))
    ax_current.set_ylabel("Current (A)")
    ax_voltage.legend(handles=[voltage_line, current_line], loc="upper right")
    fig.savefig(figure_directory / "waveforms.png", bbox_inches="tight")
    plt.close(fig)

    comparison_keys = [
        "electron_density_m3",
        "plasma_voltage_amplitude_v",
        "plasma_voltage_offset_v",
        "generator_current_amplitude_a",
        "plasma_current_amplitude_a",
        "load_current_amplitude_a",
        "stray_current_amplitude_a",
        "source_resistor_loss_w",
        "match_loss_w",
        "stray_loss_w",
        "absorbed_power_w",
    ]
    labels = ["ne", "V amp", "V offset", "Irf", "Ipl", "IL", "Istray", "PRrf", "PRm", "Pstray", "Ppl"]
    reproduced = {**metrics, "electron_density_m3": base["electron_density_m3"]}
    ratios = [100.0 * abs(float(reproduced[k])) / abs(PAPER_TARGETS[k]) for k in comparison_keys]
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    ax.bar(labels, ratios, color="#2e7d32")
    ax.axhline(100.0, color="black", linewidth=1)
    ax.axhspan(90.0, 110.0, color="#a5d6a7", alpha=0.25)
    ax.set(ylabel="Reproduction / paper (%)", ylim=(80, 115))
    fig.savefig(figure_directory / "paper_comparison.png", bbox_inches="tight")
    plt.close(fig)

    power_names = ["source_resistor_loss_w", "match_loss_w", "stray_loss_w", "absorbed_power_w"]
    power_labels = ["Generator R", "Matching R", "Stray R", "Plasma"]
    paper_power = [PAPER_TARGETS[name] for name in power_names]
    reproduced_power = [float(metrics[name]) for name in power_names]
    x = np.arange(len(power_names))
    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    ax.bar(x - 0.18, paper_power, 0.36, label="Paper", color="#90a4ae")
    ax.bar(x + 0.18, reproduced_power, 0.36, label="Reproduction", color="#00897b")
    ax.set(xticks=x, xticklabels=power_labels, ylabel="Mean power (W)")
    ax.legend()
    fig.savefig(figure_directory / "power_breakdown.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    for ax, sweep_name, x_key, xlabel in (
        (axes[0], "timestep_sensitivity", "samples_per_cycle", "Samples / RF cycle"),
        (axes[1], "regularization_sensitivity", "regularization_width_v", "Smoothing width (V)"),
    ):
        sweep = validation[sweep_name]
        reference = next(item for item in sweep if item[x_key] in (240, 0.05))
        for metric_name, label in (
            ("absorbed_power_w", "Ppl"),
            ("plasma_voltage_amplitude_v", "V amp"),
            ("plasma_current_amplitude_a", "Ipl"),
        ):
            baseline = float(reference["metrics"][metric_name])
            x_values = [float(item[x_key]) for item in sweep]
            deviations = [
                100.0 * (float(item["metrics"][metric_name]) / baseline - 1.0)
                for item in sweep
            ]
            ax.plot(x_values, deviations, marker="o", label=label)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set(xlabel=xlabel, ylabel="Deviation from standard (%)")
    axes[1].set_xscale("log")
    axes[0].legend()
    fig.savefig(figure_directory / "numerical_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/schmidt2018.json"))
    parser.add_argument(
        "--base-summary",
        type=Path,
        default=Path("artifacts/schmidt2018/reproduction_strict/summary.json"),
    )
    parser.add_argument(
        "--base-output",
        type=Path,
        default=Path("artifacts/schmidt2018/reproduction_strict"),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("artifacts/schmidt2018/validation"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/data/schmidt2018_validation.json"),
    )
    parser.add_argument(
        "--figure-directory", type=Path, default=Path("reports/figures")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validation = run_validation(
        args.config, args.base_summary, args.raw_output, args.summary_output
    )
    make_figures(validation, args.base_output, args.figure_directory)
    print(json.dumps({"summary": str(args.summary_output), "figures": str(args.figure_directory)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
