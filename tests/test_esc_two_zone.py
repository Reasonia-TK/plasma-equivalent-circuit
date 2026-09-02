from __future__ import annotations

import json
from pathlib import Path

import pytest

from plasma_circuit.esc_two_zone import (
    allocate_lateral_power,
    compute_two_zone_parameters,
    render_two_zone_netlist,
    run_closed_transport_validation,
)


CONFIG_PATH = Path("configs/esc_two_zone.json")


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_two_zone_netlist_has_distinct_bulk_nodes_and_lateral_rl_path() -> None:
    config = _config()
    plasma = compute_two_zone_parameters(config)
    netlist = render_two_zone_netlist(config, plasma)
    assert "Rbulk_wafer w_bulk_mid w_zone" in netlist
    assert "Rbulk_focus f_bulk_mid f_zone" in netlist
    assert "Rlateral w_zone lateral_mid" in netlist
    assert "Llateral lateral_mid f_zone" in netlist
    assert "Csh_ground_wafer w_zone 0" in netlist
    assert "Csh_ground_focus f_zone 0" in netlist
    assert "plasma_bulk" not in netlist


def test_zone_geometry_partition_preserves_total_volume_and_ground_area() -> None:
    config = _config()
    assert sum(zone["volume_m3"] for zone in config["zones"].values()) == pytest.approx(
        0.00288634
    )
    assert sum(
        zone["grounded_area_m2"] for zone in config["zones"].values()
    ) == pytest.approx(0.2)


def test_lateral_impedance_scales_inverse_with_effective_area() -> None:
    config = _config()
    weak = compute_two_zone_parameters(config, electrical_coupling_scale=0.1)
    strong = compute_two_zone_parameters(config, electrical_coupling_scale=10.0)
    assert weak.lateral.resistance_ohm / strong.lateral.resistance_ohm == pytest.approx(
        100.0
    )
    assert weak.lateral.inductance_h / strong.lateral.inductance_h == pytest.approx(
        100.0
    )
    assert (
        weak.lateral.impedance_magnitude_ohm
        / strong.lateral.impedance_magnitude_ohm
        == pytest.approx(100.0)
    )


def test_lateral_power_allocation_is_conservative_and_splits_loss() -> None:
    wafer, focus, transfer = allocate_lateral_power(
        wafer_port_power_w=3.4,
        focus_port_power_w=0.2,
        wafer_to_branch_power_w=0.8,
        branch_to_focus_power_w=0.7,
    )
    assert transfer == pytest.approx(0.75)
    assert wafer == pytest.approx(2.65)
    assert focus == pytest.approx(0.95)
    assert wafer + focus == pytest.approx(3.6)


def test_closed_transport_exchange_conserves_particles_and_energy() -> None:
    result = run_closed_transport_validation(_config())
    assert result.particle_conservation_relative_error < 1.0e-12
    assert result.energy_conservation_relative_error < 1.0e-12
    assert result.final_density_nonuniformity < 1.0e-6
    assert result.final_temperature_nonuniformity < 1.0e-6


def test_zero_transport_exchange_keeps_each_zone_unchanged() -> None:
    result = run_closed_transport_validation(_config(), exchange_scale=0.0)
    assert result.density_wafer_m3[-1] == pytest.approx(result.density_wafer_m3[0])
    assert result.density_focus_m3[-1] == pytest.approx(result.density_focus_m3[0])
    assert result.temperature_wafer_ev[-1] == pytest.approx(
        result.temperature_wafer_ev[0]
    )
    assert result.temperature_focus_ev[-1] == pytest.approx(
        result.temperature_focus_ev[0]
    )
