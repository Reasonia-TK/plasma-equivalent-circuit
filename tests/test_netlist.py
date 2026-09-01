from __future__ import annotations

import json
from pathlib import Path

from plasma_circuit.ngspice import render_netlist
from plasma_circuit.physics import compute_plasma_parameters


def test_netlist_contains_paper_topology_and_smooth_regularization() -> None:
    config = json.loads(Path("configs/schmidt2018.json").read_text(encoding="utf-8"))
    plasma = compute_plasma_parameters(config, 1.25e15, 4.75)
    netlist = render_netlist(config, plasma)
    assert "Cmatch1 match_shunt 0" in netlist
    assert "Cmatch2 match series_c" in netlist
    assert "Lmatch series_c load_sense" in netlist
    assert "Vsense_load load_sense match_loss_in 0" in netlist
    assert "Cstray stray_top stray_mid" in netlist
    assert "sqrt(v(bulk1,plasma)*v(bulk1,plasma)+vcap*vcap)" in netlist
    assert "exp(-spos(v(bulk1,plasma),veps)/te_ev)" in netlist
    assert ".func spos" in netlist
    assert "0 0 9.0000000000000000e+01)" in netlist
    assert "Rmatch match_loss_in load" in netlist
    assert "Vsense_plasma load plasma 0" in netlist
