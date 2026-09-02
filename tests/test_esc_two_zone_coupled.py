from __future__ import annotations

import json
from pathlib import Path

import pytest

from plasma_circuit.esc_two_zone import ZoneState
from plasma_circuit.esc_two_zone_coupled import (
    evaluate_two_zone_balances,
    solve_two_zone_balance_state,
)
from plasma_circuit.physics import electron_temperature_from_particle_balance_geometry
from plasma_circuit.physics import density_from_power_balance_surfaces


CONFIG_PATH = Path("configs/esc_two_zone.json")


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _sheaths() -> dict[str, tuple[float, float]]:
    return {"wafer": (78.0, 40.0), "focus_ring": (77.0, 40.0)}


def test_exchange_terms_cancel_in_summed_particle_and_power_balances() -> None:
    config = _config()
    states = {
        "wafer": ZoneState(3.0e14, 5.0),
        "focus_ring": ZoneState(1.0e14, 3.0),
    }
    powers = {"wafer": 3.0, "focus_ring": 1.0}
    uncoupled = evaluate_two_zone_balances(
        config, states, powers, _sheaths(), transport_scale=0.0
    )
    coupled = evaluate_two_zone_balances(
        config, states, powers, _sheaths(), transport_scale=10.0
    )
    assert (
        coupled.particle_residual_wafer_s + coupled.particle_residual_focus_s
    ) == pytest.approx(
        uncoupled.particle_residual_wafer_s + uncoupled.particle_residual_focus_s,
        rel=1.0e-14,
    )
    assert coupled.power_residual_wafer_w + coupled.power_residual_focus_w == pytest.approx(
        uncoupled.power_residual_wafer_w + uncoupled.power_residual_focus_w,
        rel=1.0e-14,
        abs=1.0e-14,
    )


def test_frozen_circuit_balance_solver_recovers_constructed_solution() -> None:
    config = _config()
    temperature = electron_temperature_from_particle_balance_geometry(
        config,
        plasma_volume_m3=float(config["zones"]["wafer"]["volume_m3"]),
        loss_area_m2=float(config["surfaces"]["wafer"]["area_m2"])
        + float(config["zones"]["wafer"]["grounded_area_m2"]),
    )
    expected = {
        "wafer": ZoneState(2.4e14, temperature),
        "focus_ring": ZoneState(2.4e14, temperature),
    }
    probe = evaluate_two_zone_balances(
        config,
        expected,
        {"wafer": 1.0, "focus_ring": 1.0},
        _sheaths(),
        transport_scale=1.0,
    )
    powers = {
        "wafer": probe.wafer.total_loss_w,
        "focus_ring": probe.focus_ring.total_loss_w,
    }
    initial = {
        "wafer": ZoneState(1.5e14, temperature * 1.1),
        "focus_ring": ZoneState(3.5e14, temperature * 0.9),
    }
    solved = solve_two_zone_balance_state(
        config, powers, _sheaths(), initial, transport_scale=1.0
    )
    assert solved.success
    assert solved.evaluation.max_normalized_residual < 1.0e-8
    assert solved.states["wafer"].electron_density_m3 == pytest.approx(2.4e14, rel=1.0e-7)
    assert solved.states["focus_ring"].electron_density_m3 == pytest.approx(
        2.4e14, rel=1.0e-7
    )
    assert solved.states["wafer"].electron_temperature_ev == pytest.approx(
        temperature, rel=1.0e-8
    )
    assert solved.states["focus_ring"].electron_temperature_ev == pytest.approx(
        temperature, rel=1.0e-8
    )


def test_local_loss_reduces_to_existing_global_power_balance_without_exchange() -> None:
    config = _config()
    temperature = electron_temperature_from_particle_balance_geometry(
        config,
        plasma_volume_m3=float(config["zones"]["wafer"]["volume_m3"]),
        loss_area_m2=float(config["surfaces"]["wafer"]["area_m2"])
        + float(config["zones"]["wafer"]["grounded_area_m2"]),
    )
    density = 2.4e14
    states = {
        "wafer": ZoneState(density, temperature),
        "focus_ring": ZoneState(density, temperature),
    }
    evaluation = evaluate_two_zone_balances(
        config,
        states,
        {"wafer": 1.0, "focus_ring": 1.0},
        _sheaths(),
        transport_scale=0.0,
    )
    recovered = density_from_power_balance_surfaces(
        config,
        temperature,
        evaluation.wafer.total_loss_w,
        surface_areas_m2=(
            float(config["surfaces"]["wafer"]["area_m2"]),
            float(config["zones"]["wafer"]["grounded_area_m2"]),
        ),
        mean_sheath_voltages_v=_sheaths()["wafer"],
        plasma_volume_m3=float(config["zones"]["wafer"]["volume_m3"]),
    )
    assert recovered == pytest.approx(density, rel=1.0e-12)
