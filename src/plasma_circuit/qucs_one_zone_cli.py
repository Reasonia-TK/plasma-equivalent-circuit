"""Run a Qucs-S RLC netlist coupled to the one-zone global plasma model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from plasma_circuit.qucs_one_zone import (
    load_qucs_one_zone_config,
    qucs_one_zone_result_to_dict,
    solve_qucs_one_zone_coupled,
)


def make_figure(
    config: Mapping[str, Any], payload: Mapping[str, Any], result: Any, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history = payload["result"]["history"]
    iterations = np.array([row["iteration"] for row in history])
    density = np.array([row["density_input_m3"] for row in history])
    target = np.array([row["density_target_m3"] for row in history])
    residual = np.array([row["balance_relative_residual"] for row in history])
    simulation = result.final_simulation
    frequency = float(config["frequency_hz"])
    start = simulation.time_s[-1] - 2.0 / frequency
    mask = simulation.time_s >= start
    time_ns = (simulation.time_s[mask] - simulation.time_s[-1]) * 1.0e9
    waves = simulation.waveforms
    metrics = payload["result"]["metrics"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    axes[0].plot(iterations, density / 1.0e14, "o-", label="circuit input")
    axes[0].plot(iterations, target / 1.0e14, "s--", label="global target")
    axes[0].set_xlabel("Outer iteration")
    axes[0].set_ylabel(r"Electron density ($10^{14}$ m$^{-3}$)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    residual_axis = axes[0].twinx()
    residual_axis.semilogy(iterations, residual, "k:", alpha=0.7)
    residual_axis.set_ylabel("Balance residual", color="black")

    axes[1].plot(time_ns, waves["v_source"][mask], label="source")
    axes[1].plot(time_ns, waves["v_feed"][mask], label="w_feed")
    axes[1].plot(time_ns, waves["v_plasma"][mask], label="plasma surface")
    axes[1].set_xlabel("Time from final sample (ns)")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    labels = ["source", "R1 loss", "plasma"]
    values = [
        metrics["source_delivered_power_w"],
        metrics["source_resistor_loss_w"],
        metrics["absorbed_power_w"],
    ]
    axes[2].bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[2].set_ylabel("Cycle-averaged power (W)")
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].set_title(
        "Power balance error\n"
        f"{100.0 * metrics['power_balance_relative_error']:.4f}%"
    )
    fig.suptitle("Qucs-S RLC + one-zone global plasma coupling")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/qucs_rlc_one_zone.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/qucs_rlc_one_zone/self_consistent")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("reports/data/qucs_rlc_one_zone.json")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("reports/figures/qucs_rlc_one_zone.png")
    )
    args = parser.parse_args()
    config = load_qucs_one_zone_config(args.config)
    result = solve_qucs_one_zone_coupled(config, args.output)
    result_payload = qucs_one_zone_result_to_dict(result)
    payload = {
        "model_scope": "Qucs-S series RLC external netlist coupled to one-zone Ar CCP global model",
        "config": str(args.config),
        "qucs_source_netlist": str(config["qucs_netlist"]["path"]),
        "result": result_payload,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figure(config, payload, result, args.figure)
    print(
        json.dumps(
            {
                "converged": result.converged,
                "iterations": len(result.history),
                "electron_temperature_ev": result.electron_temperature_ev,
                "electron_density_m3": result.electron_density_m3,
                "final_balance_relative_residual": result.final_balance_relative_residual,
                "absorbed_power_w": result.final_simulation.metrics.absorbed_power_w,
                "power_balance_relative_error": result.final_simulation.metrics.power_balance_relative_error,
                "cycle_l2_max": max(
                    result.final_simulation.metrics.cycle_l2_voltage,
                    result.final_simulation.metrics.cycle_l2_current,
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not result.converged:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
