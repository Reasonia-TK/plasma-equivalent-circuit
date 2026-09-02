from __future__ import annotations

from pathlib import Path

import pytest

from plasma_circuit.physics import compute_plasma_parameters
from plasma_circuit.qucs_one_zone import (
    load_qucs_one_zone_config,
    normalize_qucs_netlist,
    render_qucs_one_zone_netlist,
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
