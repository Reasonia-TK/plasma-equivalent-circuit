from __future__ import annotations

from plasma_circuit.qucs_matching import (
    apply_selected_design,
    external_impedance,
    load_matching_search_config,
    reflection_coefficient,
)
from plasma_circuit.qucs_one_zone import load_qucs_one_zone_config, series_inductor_esr_ohm


def test_reflection_and_source_resistance_deembedding() -> None:
    assert external_impedance(100.0 + 2.0j, 50.0) == 50.0 + 2.0j
    assert reflection_coefficient(50.0 + 0.0j, 50.0) == 0.0j
    assert abs(reflection_coefficient(25.0 + 0.0j, 50.0)) == 1.0 / 3.0


def test_selected_matching_design_is_applied_with_finite_q() -> None:
    search = load_matching_search_config("configs/qucs_rlc_matching_search.json")
    base = load_qucs_one_zone_config(search["resolved_base_config"])
    selected = apply_selected_design(base, search)
    assert selected["qucs_netlist"]["component_overrides"] == {
        "L1": 5.0e-6,
        "C1": 1.0e-9,
    }
    assert selected["matching"]["shunt_capacitance_f"] == 12.0e-12
    assert selected["matching"]["series_inductor_quality_factor"] == 30.0
    assert series_inductor_esr_ohm(selected) > 14.0
    assert selected["transient"]["cycles"] == 1200
    assert selected["coupling"]["max_iterations"] == 30
