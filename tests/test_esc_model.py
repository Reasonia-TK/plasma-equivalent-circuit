from __future__ import annotations

import json
from pathlib import Path

import pytest

from plasma_circuit.esc_model import (
    compute_esc_plasma_parameters,
    render_esc_netlist,
)
from plasma_circuit.esc_sweep import focus_capacitance_execution_order
from plasma_circuit.physics import electron_temperature_from_particle_balance_geometry


CONFIG_PATH = Path("configs/esc_wafer_focus_ring.json")


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_esc_config_uses_physical_differential_sheath_capacitance() -> None:
    config = _config()
    assert config["regularization"]["capacitance_scale"] == pytest.approx(0.5)
    assert (
        config["regularization"]["capacitance_convention"]
        == "physical-matrix-sheath-differential"
    )


def test_esc_surface_parameters_scale_with_area() -> None:
    config = _config()
    wafer_area = config["surfaces"]["wafer"]["area_m2"]
    focus_area = config["surfaces"]["focus_ring"]["area_m2"]
    temperature = electron_temperature_from_particle_balance_geometry(
        config,
        plasma_volume_m3=config["plasma_volume_m3"],
        loss_area_m2=wafer_area + focus_area + config["grounded_area_m2"],
    )
    plasma = compute_esc_plasma_parameters(config, 1.0e15, temperature)
    area_ratio = wafer_area / focus_area
    assert plasma.wafer.ion_current_a / plasma.focus_ring.ion_current_a == pytest.approx(
        area_ratio
    )
    assert (
        plasma.wafer.electron_saturation_current_a
        / plasma.focus_ring.electron_saturation_current_a
        == pytest.approx(area_ratio)
    )
    assert plasma.wafer.sheath_k / plasma.focus_ring.sheath_k == pytest.approx(
        area_ratio**2
    )
    assert (
        plasma.focus_ring.bulk_inductance_h / plasma.wafer.bulk_inductance_h
        == pytest.approx(area_ratio)
    )


def test_esc_netlist_contains_two_dielectric_and_three_sheath_branches() -> None:
    config = _config()
    wafer_area = config["surfaces"]["wafer"]["area_m2"]
    focus_area = config["surfaces"]["focus_ring"]["area_m2"]
    temperature = electron_temperature_from_particle_balance_geometry(
        config,
        plasma_volume_m3=config["plasma_volume_m3"],
        loss_area_m2=wafer_area + focus_area + config["grounded_area_m2"],
    )
    plasma = compute_esc_plasma_parameters(config, 1.0e15, temperature)
    netlist = render_esc_netlist(config, plasma)
    assert "Cesc_wafer w_electrode w_cap" in netlist
    assert "Cesc_focus f_electrode f_cap" in netlist
    assert "Csh_wafer wafer w_bulk" in netlist
    assert "Csh_focus focus f_bulk" in netlist
    assert "Csh_ground plasma_bulk 0" in netlist
    assert "Vsense_surface_wafer w_feed wafer 0" in netlist
    assert "Vsense_surface_focus f_feed focus 0" in netlist
    assert "Bsource_wafer w_src 0 V=" in netlist
    assert "tanh(time/" in netlist
    assert ".param cscale=5.0000000000000000e-01" in netlist


def test_focus_capacitance_sweep_starts_at_base_value() -> None:
    order = focus_capacitance_execution_order(
        [90.0e-12, 180.0e-12, 360.0e-12, 720.0e-12],
        180.0e-12,
    )
    assert order == [180.0e-12, 90.0e-12, 360.0e-12, 720.0e-12]
