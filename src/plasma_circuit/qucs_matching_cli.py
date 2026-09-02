"""Verify and report the selected Qucs-S one-zone matching condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from plasma_circuit.qucs_matching import verify_selected_matching_design


def make_matching_figure(
    payload: Mapping[str, Any], result: Any, path: Path
) -> None:
    """Plot density convergence, impedance, waveforms, and power flow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = payload["baseline"]
    selected = payload["selected"]
    history = selected["history"]
    simulation = result.final_simulation
    frequency = float(payload["frequency_hz"])
    mask = simulation.time_s >= simulation.time_s[-1] - 2.0 / frequency
    time_ns = (simulation.time_s[mask] - simulation.time_s[-1]) * 1.0e9

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
    iterations = np.array([row["iteration"] for row in history])
    axes[0, 0].plot(
        iterations,
        np.array([row["density_input_m3"] for row in history]) / 1.0e15,
        "o-",
        label="circuit input",
    )
    axes[0, 0].plot(
        iterations,
        np.array([row["density_target_m3"] for row in history]) / 1.0e15,
        "s--",
        label="global target",
    )
    axes[0, 0].set(
        xlabel="Outer iteration",
        ylabel=r"Electron density ($10^{15}$ m$^{-3}$)",
    )
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    for label, row, marker in (
        ("baseline", baseline["matching"], "o"),
        ("selected", selected["matching"], "s"),
    ):
        axes[0, 1].scatter(
            row["external_impedance_real_ohm"],
            row["external_impedance_imag_ohm"],
            s=70,
            marker=marker,
            label=label,
        )
    axes[0, 1].scatter(50.0, 0.0, marker="*", s=120, color="black", label="50 ohm target")
    axes[0, 1].set(xlabel="Re(Z external) (ohm)", ylabel="Im(Z external) (ohm)")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    waves = simulation.waveforms
    axes[1, 0].plot(time_ns, waves["v_source"][mask], label="source")
    axes[1, 0].plot(time_ns, waves["v_feed"][mask], label="w_feed")
    axes[1, 0].plot(time_ns, waves["v_plasma"][mask], label="plasma surface")
    axes[1, 0].set(xlabel="Time from final sample (ns)", ylabel="Voltage (V)")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    labels = ["baseline", "selected"]
    r1 = [
        baseline["metrics"]["source_resistor_loss_w"],
        selected["metrics"]["source_resistor_loss_w"],
    ]
    coil = [baseline["metrics"]["match_loss_w"], selected["metrics"]["match_loss_w"]]
    plasma = [baseline["metrics"]["absorbed_power_w"], selected["metrics"]["absorbed_power_w"]]
    axes[1, 1].bar(labels, r1, label="source R1")
    axes[1, 1].bar(labels, coil, bottom=r1, label="inductor ESR")
    axes[1, 1].bar(labels, plasma, bottom=np.array(r1) + np.array(coil), label="plasma")
    axes[1, 1].set(ylabel="Cycle-averaged power (W)")
    axes[1, 1].grid(True, axis="y", alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Qucs-S one-zone finite-Q matching verification")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search", type=Path, default=Path("configs/qucs_rlc_matching_search.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/qucs_rlc_matching/selected")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("reports/data/qucs_rlc_matching.json")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("reports/figures/qucs_rlc_matching.png")
    )
    args = parser.parse_args()
    payload, result = verify_selected_matching_design(args.search, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_matching_figure(payload, result, args.figure)
    selected = payload["selected"]
    print(
        json.dumps(
            {
                "converged": selected["converged"],
                "iterations": len(selected["history"]),
                "electron_density_m3": selected["electron_density_m3"],
                "reflection_magnitude": selected["matching"]["reflection_magnitude"],
                "absorbed_power_w": selected["metrics"]["absorbed_power_w"],
                "power_balance_relative_error": selected["metrics"][
                    "power_balance_relative_error"
                ],
                "gates": payload["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not all(payload["gates"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
