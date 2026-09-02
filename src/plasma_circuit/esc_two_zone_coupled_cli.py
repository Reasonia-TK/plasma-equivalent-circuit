"""Run the self-consistent two-zone ESC global model and transport sensitivity."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from plasma_circuit.esc_two_zone import load_two_zone_config
from plasma_circuit.esc_two_zone_coupled import (
    coupled_result_to_dict,
    solve_two_zone_coupled_model,
)


def _sweep_config(config: dict, scale: float) -> dict:
    result = copy.deepcopy(config)
    coupled = result["self_consistent"]
    threshold = float(coupled.get("transport_sweep_high_scale_threshold", 5.0))
    coupled["relative_tolerance"] = max(
        float(coupled["relative_tolerance"]),
        float(coupled.get("transport_sweep_relative_tolerance", 1.0e-3)),
    )
    if scale >= threshold:
        coupled["relaxation"] = min(
            float(coupled["relaxation"]),
            float(coupled.get("transport_sweep_high_scale_relaxation", 0.15)),
        )
        coupled["max_iterations"] = max(
            int(coupled["max_iterations"]),
            int(coupled.get("transport_sweep_max_iterations", 40)),
        )
    return result


def _plot(
    baseline: Mapping[str, Any],
    sensitivity: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = baseline["history"]
    iterations = np.array([item["iteration"] for item in history])
    density_wafer = np.array(
        [item["density_wafer_input_m3"] for item in history]
    )
    density_focus = np.array(
        [item["density_focus_input_m3"] for item in history]
    )
    residual = np.array(
        [item["balance_residual_input_max"] for item in history]
    )
    target_change = np.array(
        [item["target_relative_change_max"] for item in history]
    )
    scales = np.array([item["transport_scale"] for item in sensitivity])
    density_nonuniformity = 100.0 * np.array(
        [item["result"]["uniformity"]["density_nonuniformity"] for item in sensitivity]
    )
    temperature_nonuniformity = 100.0 * np.array(
        [
            item["result"]["uniformity"]["temperature_nonuniformity"]
            for item in sensitivity
        ]
    )
    flux_nonuniformity = 100.0 * np.array(
        [item["result"]["uniformity"]["ion_flux_nonuniformity"] for item in sensitivity]
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    axes[0].plot(iterations, density_wafer / 1.0e14, "o-", label=r"$n_{e,W}$")
    axes[0].plot(iterations, density_focus / 1.0e14, "o-", label=r"$n_{e,F}$")
    axes[0].set_xlabel("Outer iteration")
    axes[0].set_ylabel(r"Density ($10^{14}$ m$^{-3}$)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].semilogy(iterations, residual, "o-", label="balance residual")
    axes[1].semilogy(iterations, target_change, "o-", label="target change")
    axes[1].axhline(1.0e-3, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Outer iteration")
    axes[1].set_ylabel("Maximum relative residual")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)
    scale_positions = np.arange(len(scales))
    axes[2].plot(scale_positions, density_nonuniformity, "o-", label=r"$n_e$")
    axes[2].plot(scale_positions, flux_nonuniformity, "s-", label="ion flux")
    axes[2].plot(scale_positions, temperature_nonuniformity, "^-", label=r"$T_e$")
    axes[2].set_xticks(scale_positions, [f"{scale:g}" for scale in scales])
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Particle / thermal transport scale (log-spaced cases)")
    axes[2].set_ylabel("Wafer-Focus nonuniformity (%)")
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/esc_two_zone/self_consistent"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/data/esc_two_zone_self_consistent.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("reports/figures/esc_two_zone_self_consistent.png"),
    )
    args = parser.parse_args()
    config = load_two_zone_config(args.config)
    baseline_scale = float(config["self_consistent"].get("transport_scale", 1.0))
    baseline = solve_two_zone_coupled_model(
        config,
        args.output / "baseline",
        transport_scale=baseline_scale,
    )
    if not baseline.converged:
        raise RuntimeError("baseline two-zone self-consistent calculation did not converge")
    sensitivity: list[dict] = []
    for raw_scale in config["self_consistent"]["transport_sweep_scales"]:
        scale = float(raw_scale)
        if np.isclose(scale, baseline_scale, rtol=0.0, atol=1.0e-15):
            result = baseline
        else:
            result = solve_two_zone_coupled_model(
                _sweep_config(config, scale),
                args.output / f"transport_{scale:.6g}".replace(".", "p"),
                initial_states=baseline.states,
                transport_scale=scale,
            )
        if not result.converged:
            raise RuntimeError(
                f"transport sensitivity scale {scale:g} did not converge; "
                f"final residual={result.final_balance.max_normalized_residual:.3e}"
            )
        sensitivity.append(
            {
                "transport_scale": scale,
                "result": coupled_result_to_dict(result),
            }
        )
    baseline_payload = coupled_result_to_dict(baseline)
    payload = {
        "model_scope": "self-consistent two-zone global balances coupled to nonlinear ngspice circuit",
        "config": str(args.config),
        "baseline_transport_scale": baseline_scale,
        "baseline": baseline_payload,
        "transport_sensitivity": sensitivity,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot(baseline_payload, sensitivity, args.figure)
    print(
        json.dumps(
            {
                "converged": baseline.converged,
                "iterations": len(baseline.history),
                "states": payload["baseline"]["states"],
                "uniformity": payload["baseline"]["uniformity"],
                "final_balance_residual": baseline.final_balance.max_normalized_residual,
                "transport_sensitivity": [
                    {
                        "transport_scale": item["transport_scale"],
                        "converged": item["result"]["converged"],
                        "ion_flux_nonuniformity": item["result"]["uniformity"][
                            "ion_flux_nonuniformity"
                        ],
                    }
                    for item in sensitivity
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
