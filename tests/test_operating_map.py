from __future__ import annotations

from pathlib import Path

import pytest

from plasma_circuit.operating_map import fundamental_reflection, make_operating_map_figures


def test_fundamental_reflection_is_zero_at_match() -> None:
    assert fundamental_reflection(50.0 + 0.0j, 50.0) == pytest.approx(0.0j)


def test_fundamental_reflection_has_unit_magnitude_for_open_and_short_limits() -> None:
    assert abs(fundamental_reflection(1.0e30 + 0.0j, 50.0)) == pytest.approx(1.0)
    assert abs(fundamental_reflection(0.0 + 0.0j, 50.0)) == pytest.approx(1.0)


def test_operating_map_figures_are_generated(tmp_path: Path) -> None:
    output = {
        "pressures_pa": [1.0],
        "source_amplitudes_v": [100.0],
        "nonconverged_conditions": [
            {"pressure_pa": 1.0, "source_amplitude_v": 100.0}
        ],
        "results": [
            {
                "pressure_pa": 1.0,
                "source_amplitude_v": 100.0,
                "electron_density_m3": 1.0e15,
                "electron_temperature_ev": 4.0,
                "reflection_magnitude": 0.1,
                "plasma_efficiency_available": 0.2,
                "metrics": {
                    "absorbed_power_w": 5.0,
                    "plasma_voltage_offset_v": -200.0,
                    "plasma_current_thd": 0.3,
                    "power_balance_relative_error": 1.0e-4,
                },
            }
        ],
    }

    make_operating_map_figures(output, tmp_path)

    assert (tmp_path / "pressure_voltage_plasma_map.png").is_file()
    assert (tmp_path / "pressure_voltage_circuit_map.png").is_file()
