from __future__ import annotations

from pathlib import Path

import pytest

from plasma_circuit.physics import compute_plasma_parameters
from plasma_circuit.qucs_one_zone import (
    apply_component_overrides,
    apply_series_inductor_quality_factor,
    load_qucs_one_zone_config,
    normalize_qucs_netlist,
    render_qucs_one_zone_netlist,
    series_inductor_esr_ohm,
)


@pytest.fixture()
def config() -> dict:
    return load_qucs_one_zone_config("configs/qucs_rlc_one_zone.json")


def test_normalizer_keeps_rlc_interface_and_removes_qucs_analysis(config: dict) -> None:
    raw = Path(config["qucs_netlist"]["resolved_path"]).read_text(encoding="utf-8")
    normalized = normalize_qucs_netlist(raw, config["qucs_netlist"])
    assert "R1 _net0 _net1" in normalized
    assert "L1 _net1 _net2" in normalized
    assert "C1 _net2 w_feed" in normalized
    assert "SIN(0 100 13.56MEG" in normalized
    assert "R2 " not in normalized
    assert "ngspice_mathfunc.inc" not in normalized
    assert ".control" not in normalized.lower()
    assert ".end" not in normalized.lower()


def test_normalizer_rejects_dummy_load(config: dict) -> None:
    raw = """Qucs body
V1 src 0 SIN(0 100 13.56MEG)
R1 src n1 50
L1 n1 n2 138N
C1 n2 w_feed 1N
R2 w_feed 0 50
.end
"""
    with pytest.raises(ValueError, match="forbidden.*R2"):
        normalize_qucs_netlist(raw, config["qucs_netlist"])


def test_component_overrides_change_values_without_changing_topology(config: dict) -> None:
    raw = Path(config["qucs_netlist"]["resolved_path"]).read_text(encoding="utf-8")
    normalized = normalize_qucs_netlist(raw, config["qucs_netlist"])
    overridden = apply_component_overrides(
        normalized,
        {"L1": 12.5e-6, "C1": 680.0e-12},
    )
    assert "L1 _net1 _net2 1.2500000000000001e-05" in overridden
    assert "C1 _net2 w_feed 6.8000000000000003e-10" in overridden
    assert "R1 _net0 _net1  50" in overridden


def test_component_overrides_reject_missing_or_nonpositive_devices(config: dict) -> None:
    raw = Path(config["qucs_netlist"]["resolved_path"]).read_text(encoding="utf-8")
    normalized = normalize_qucs_netlist(raw, config["qucs_netlist"])
    with pytest.raises(ValueError, match="positive"):
        apply_component_overrides(normalized, {"L1": 0.0})
    with pytest.raises(ValueError, match="not found"):
        apply_component_overrides(normalized, {"L404": 1.0e-6})


def test_inductor_quality_factor_adds_expected_series_loss(config: dict) -> None:
    raw = Path(config["qucs_netlist"]["resolved_path"]).read_text(encoding="utf-8")
    normalized = normalize_qucs_netlist(raw, config["qucs_netlist"])
    overridden = apply_component_overrides(normalized, {"L1": 10.0e-6})
    lossy = apply_series_inductor_quality_factor(
        overridden,
        "L1",
        frequency_hz=13.56e6,
        quality_factor=30.0,
    )
    assert "L1 _net1 qucs_l1_esr_internal 1.0000000000000001e-05" in lossy
    assert "Rloss_L1 qucs_l1_esr_internal _net2" in lossy
    resistance_line = next(line for line in lossy.splitlines() if line.startswith("Rloss_L1"))
    assert float(resistance_line.split()[3]) == pytest.approx(28.4, rel=1.0e-3)

    config["qucs_netlist"]["component_overrides"] = {"L1": 10.0e-6}
    config["matching"] = {
        "series_inductor_device": "L1",
        "series_inductor_quality_factor": 30.0,
    }
    assert series_inductor_esr_ohm(config) == pytest.approx(28.4, rel=1.0e-3)


def test_assembled_netlist_inserts_plasma_boundary_and_one_zone_model(
    config: dict,
) -> None:
    raw = Path(config["qucs_netlist"]["resolved_path"]).read_text(encoding="utf-8")
    temperature = 4.75
    plasma = compute_plasma_parameters(config, 5.0e14, temperature)
    netlist = render_qucs_one_zone_netlist(config, plasma, raw)
    assert "Vsense_surface_wafer w_feed plasma 0" in netlist
    assert "Csh1 plasma bulk1" in netlist
    assert "Belectron1 plasma bulk1" in netlist
    assert "Lplasma bulk_sense bulk_mid" in netlist
    assert "Csh2 bulk2 0" in netlist
    assert "i(Vsense_generator_qucs)" in netlist
    assert "i(Vsense_surface_wafer)" in netlist
    assert "Bsource_qucs qucs_source 0" in netlist
    assert "tanh(time/" in netlist
    assert not any(line.startswith("V1 ") for line in netlist.splitlines())
    assert netlist.lower().count(".control") == 1
    assert netlist.lower().count(".endc") == 1
    assert netlist.lower().count(".end\n") == 1


def test_assembled_netlist_can_insert_load_side_shunt_capacitance(config: dict) -> None:
    raw = Path(config["qucs_netlist"]["resolved_path"]).read_text(encoding="utf-8")
    config["matching"] = {"shunt_capacitance_f": 5.0e-12}
    plasma = compute_plasma_parameters(config, 5.0e14, 4.75)
    netlist = render_qucs_one_zone_netlist(config, plasma, raw)
    assert "Cmatch_shunt_qucs w_feed 0 4.9999999999999997e-12" in netlist
    assert netlist.index("Cmatch_shunt_qucs") < netlist.index("Vsense_surface_wafer")


def test_assembled_netlist_can_insert_load_side_shunt_inductance(config: dict) -> None:
    raw = Path(config["qucs_netlist"]["resolved_path"]).read_text(encoding="utf-8")
    config["matching"] = {"shunt_inductance_h": 5.0e-6}
    plasma = compute_plasma_parameters(config, 5.0e14, 4.75)
    netlist = render_qucs_one_zone_netlist(config, plasma, raw)
    assert "Lmatch_shunt_qucs w_feed 0 5.0000000000000004e-06" in netlist


def test_assembled_netlist_rejects_two_shunt_matching_elements(config: dict) -> None:
    raw = Path(config["qucs_netlist"]["resolved_path"]).read_text(encoding="utf-8")
    config["matching"] = {
        "shunt_capacitance_f": 5.0e-12,
        "shunt_inductance_h": 5.0e-6,
    }
    plasma = compute_plasma_parameters(config, 5.0e14, 4.75)
    with pytest.raises(ValueError, match="either shunt capacitance or shunt inductance"):
        render_qucs_one_zone_netlist(config, plasma, raw)
