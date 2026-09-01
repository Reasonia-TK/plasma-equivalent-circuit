from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plasma_circuit.physics import (
    argon_rate_coefficients,
    compute_plasma_parameters,
    density_from_power_balance,
    density_from_power_balance_surfaces,
    electron_temperature_from_particle_balance,
    electron_temperature_from_particle_balance_geometry,
    matrix_sheath_charge,
    matrix_sheath_differential_capacitance,
    matrix_sheath_secant_capacitance,
    regularized_electron_current,
    regularized_sheath_capacitance,
    smooth_positive,
)


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(Path("configs/schmidt2018.json").read_text(encoding="utf-8"))


def test_particle_balance_reproduces_paper_temperature(config: dict) -> None:
    assert electron_temperature_from_particle_balance(config) == pytest.approx(4.75, abs=0.01)


def test_gudmundsson_rates_at_paper_temperature() -> None:
    rates = argon_rate_coefficients(4.75)
    assert rates["ionization"] == pytest.approx(1.4925e-15, rel=5e-4)
    assert rates["excitation"] > rates["ionization"]
    assert rates["elastic"] > rates["excitation"]


def test_regularized_electron_current_is_smooth_finite_and_bounded(config: dict) -> None:
    plasma = compute_plasma_parameters(config, 1.25e15, 4.75)
    voltages = np.linspace(-1000.0, 1000.0, 200_001)
    currents = regularized_electron_current(
        voltages,
        plasma.electron_saturation_powered_a,
        4.75,
        config["regularization"]["electron_voltage_v"],
    )
    assert np.all(np.isfinite(currents))
    assert np.all(currents > 0.0)
    assert np.max(currents) <= plasma.electron_saturation_powered_a * (1.0 + 1e-12)
    assert np.all(np.diff(currents) <= 1e-12)


def test_smooth_positive_matches_positive_voltage_away_from_zero() -> None:
    width = 0.05
    assert smooth_positive(1.0, width) == pytest.approx(1.0, rel=7e-4)
    assert smooth_positive(-100.0, width) >= 0.0


def test_regularized_capacitance_is_finite_at_zero_and_matches_paper(config: dict) -> None:
    plasma = compute_plasma_parameters(config, 1.25e15, 4.75)
    width = config["regularization"]["capacitance_voltage_v"]
    capacitance_zero = regularized_sheath_capacitance(0.0, plasma.sheath_k_powered, width)
    capacitance_100 = regularized_sheath_capacitance(100.0, plasma.sheath_k_powered, width)
    paper_100 = np.sqrt(plasma.sheath_k_powered / 100.0)
    assert np.isfinite(capacitance_zero)
    assert capacitance_zero > capacitance_100
    assert capacitance_100 == pytest.approx(paper_100, rel=1e-6)


def test_paper_density_gives_expected_bulk_elements(config: dict) -> None:
    plasma = compute_plasma_parameters(config, 1.25e15, 4.75)
    assert plasma.bulk_inductance_h == pytest.approx(1.6182e-7, rel=5e-4)
    assert plasma.bulk_resistance_ohm == pytest.approx(9.6543, rel=5e-4)
    assert plasma.electron_saturation_powered_a > plasma.ion_current_powered_a


def test_matrix_sheath_charge_distinguishes_secant_and_differential_capacitance(
    config: dict,
) -> None:
    plasma = compute_plasma_parameters(config, 1.25e15, 4.75)
    voltage = 100.0
    k_value = plasma.sheath_k_powered
    step = 1.0e-3
    numerical_derivative = (
        matrix_sheath_charge(voltage + step, k_value)
        - matrix_sheath_charge(voltage - step, k_value)
    ) / (2.0 * step)
    secant = matrix_sheath_secant_capacitance(voltage, k_value)
    differential = matrix_sheath_differential_capacitance(voltage, k_value)
    assert differential == pytest.approx(0.5 * secant, rel=1e-12)
    assert numerical_derivative == pytest.approx(differential, rel=1e-10)


def test_generic_particle_balance_matches_two_surface_wrapper(config: dict) -> None:
    expected = electron_temperature_from_particle_balance(config)
    generic = electron_temperature_from_particle_balance_geometry(
        config,
        plasma_volume_m3=config["powered_area_m2"] * config["bulk_length_m"],
        loss_area_m2=config["powered_area_m2"] + config["grounded_area_m2"],
    )
    assert generic == pytest.approx(expected, rel=1e-12)


def test_multi_surface_power_balance_reduces_to_two_surface_case(config: dict) -> None:
    temperature = 4.75
    power = 5.0
    powered_voltage = 250.0
    grounded_voltage = 80.0
    expected = density_from_power_balance(
        config,
        temperature,
        power,
        powered_voltage,
        grounded_voltage,
    )
    generic = density_from_power_balance_surfaces(
        config,
        temperature,
        power,
        surface_areas_m2=(0.006, 0.004, config["grounded_area_m2"]),
        mean_sheath_voltages_v=(
            powered_voltage,
            powered_voltage,
            grounded_voltage,
        ),
        plasma_volume_m3=config["powered_area_m2"] * config["bulk_length_m"],
    )
    assert generic == pytest.approx(expected, rel=1e-12)
