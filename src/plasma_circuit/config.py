"""Configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON configuration and validate required top-level fields."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config: dict[str, Any] = json.load(stream)

    required = {
        "ngspice_path",
        "frequency_hz",
        "source_amplitude_v",
        "source_resistance_ohm",
        "pressure_pa",
        "gas_temperature_k",
        "powered_area_m2",
        "grounded_area_m2",
        "bulk_length_m",
        "match_c1_f",
        "match_c2_f",
        "match_l_h",
        "match_loss_ohm",
        "stray_c_f",
        "stray_loss_ohm",
        "regularization",
        "transient",
        "coupling",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing configuration fields: {missing}")

    ngspice_path = Path(config["ngspice_path"])
    if not ngspice_path.is_file():
        raise FileNotFoundError(f"ngspice executable not found: {ngspice_path}")
    return config

