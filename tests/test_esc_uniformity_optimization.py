from __future__ import annotations

import json
from pathlib import Path

import pytest

from plasma_circuit.esc_two_zone import TwoZoneMetrics
from plasma_circuit.esc_uniformity_optimization import (
    apply_design,
    design_from_config,
    generate_initial_designs,
    generate_local_designs,
    power_density_nonuniformity,
)


CONFIG = Path("configs/esc_two_zone.json")
OPTIMIZATION = Path("configs/esc_uniformity_optimization.json")


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _optimization() -> dict:
    return json.loads(OPTIMIZATION.read_text(encoding="utf-8"))


def test_design_round_trip_updates_only_focus_drive_fields() -> None:
    config = _config()
    wafer_before = dict(config["surfaces"]["wafer"])
    design = {
        "focus_source_amplitude_v": 105.0,
        "focus_source_phase_deg": -7.0,
        "focus_coupling_capacitance_f": 220.0e-12,
    }
    apply_design(config, design)
    assert design_from_config(config) == pytest.approx(design)
    assert config["surfaces"]["wafer"] == wafer_before


def test_initial_designs_are_deterministic_and_include_baseline() -> None:
    config = _config()
    optimization = _optimization()
    first = generate_initial_designs(config, optimization)
    second = generate_initial_designs(config, optimization)
    assert first == second
    assert first[0] == pytest.approx(design_from_config(config))
    assert len(first) == 23


def test_full_factorial_local_designs_have_27_unique_points() -> None:
    config = _config()
    optimization = _optimization()
    designs = generate_local_designs(
        design_from_config(config), optimization, step_fraction=0.003
    )
    assert len(designs) == 27


def test_power_density_uniformity_is_zero_for_volume_proportional_power() -> None:
    config = _config()
    wafer_volume = float(config["zones"]["wafer"]["volume_m3"])
    focus_volume = float(config["zones"]["focus_ring"]["volume_m3"])
    metrics = object.__new__(TwoZoneMetrics)
    object.__setattr__(metrics, "absorbed_power_wafer_allocated_w", 1000.0 * wafer_volume)
    object.__setattr__(metrics, "absorbed_power_focus_allocated_w", 1000.0 * focus_volume)
    assert power_density_nonuniformity(config, metrics) == pytest.approx(0.0)
