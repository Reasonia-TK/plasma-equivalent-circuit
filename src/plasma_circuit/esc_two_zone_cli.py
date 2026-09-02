"""CLI for the staged two-zone ESC validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from plasma_circuit.esc_two_zone import (
    electrical_coupling_sweep,
    load_two_zone_config,
    run_closed_transport_validation,
)


def _plot_results(
    electrical: list[dict], transport, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scales = [row["electrical_coupling_scale"] for row in electrical]
    voltage_difference = [
        row["metrics"]["bulk_voltage_difference_amplitude_v"] for row in electrical
    ]
    lateral_current = [row["metrics"]["lateral_current_amplitude_a"] for row in electrical]
    time_ms = transport.time_s * 1.0e3
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    axes[0].loglog(scales, voltage_difference, "o-", color="#0068b4")
    axes[0].set_xlabel("Electrical coupling scale")
    axes[0].set_ylabel(r"$|V_{b,W}-V_{b,F}|$ (V peak)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].loglog(scales, lateral_current, "o-", color="#d55e00")
    axes[1].set_xlabel("Electrical coupling scale")
    axes[1].set_ylabel(r"$|I_{lat}|$ (A peak)")
    axes[1].grid(True, which="both", alpha=0.25)
    density_lines = axes[2].plot(
        time_ms,
        transport.density_wafer_m3 / 1.0e14,
        label=r"$n_{e,W}$",
    )
    density_lines += axes[2].plot(
        time_ms,
        transport.density_focus_m3 / 1.0e14,
        label=r"$n_{e,F}$",
    )
    temperature_axis = axes[2].twinx()
    temperature_lines = temperature_axis.plot(
        time_ms,
        transport.temperature_wafer_ev,
        "--",
        color="#009e73",
        label=r"$T_{e,W}$",
    )
    temperature_lines += temperature_axis.plot(
        time_ms,
        transport.temperature_focus_ev,
        "--",
        color="#cc3311",
        label=r"$T_{e,F}$",
    )
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel(r"Density ($10^{14}$ m$^{-3}$)")
    temperature_axis.set_ylabel(r"$T_e$ (eV)")
    axes[2].grid(True, alpha=0.25)
    lines = density_lines + temperature_lines
    axes[2].legend(lines, [line.get_label() for line in lines], fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/esc_two_zone/validation")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("reports/data/esc_two_zone_validation.json")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("reports/figures/esc_two_zone_validation.png")
    )
    args = parser.parse_args()
    config = load_two_zone_config(args.config)
    electrical = electrical_coupling_sweep(config, args.output / "electrical")
    transport = run_closed_transport_validation(config)
    payload = {
        "model_scope": "frozen-zone electrical sweep plus closed conservative transport test",
        "config": str(args.config),
        "electrical": electrical,
        "transport": {
            key: value
            for key, value in asdict(transport).items()
            if key
            not in {
                "time_s",
                "density_wafer_m3",
                "density_focus_m3",
                "temperature_wafer_ev",
                "temperature_focus_ev",
            }
        },
        "transport_final": {
            "density_wafer_m3": float(transport.density_wafer_m3[-1]),
            "density_focus_m3": float(transport.density_focus_m3[-1]),
            "temperature_wafer_ev": float(transport.temperature_wafer_ev[-1]),
            "temperature_focus_ev": float(transport.temperature_focus_ev[-1]),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_results(electrical, transport, args.figure)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
