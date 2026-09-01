"""Integration check for ngspice behavioral-capacitor semantics."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest


NGSPICE = Path(r"C:\Spice64\bin\ngspice_con.exe")


def _run_capacitor_case(tmp_path: Path, formulation: str) -> tuple[float, float]:
    initial_voltage = 10.0
    final_voltage = 20.0
    sheath_k = 1.0e-18
    stop_time = 1.0e-3
    if formulation == "C":
        element = "Ctest out 0 C='sqrt(k/v(out))'"
    elif formulation == "Q":
        element = "Ctest out 0 Q='2*sqrt(k*v(out))'"
    else:  # pragma: no cover - private helper guard
        raise ValueError(formulation)

    netlist = f"""Behavioral capacitor differential-capacitance validation
.param k={sheath_k:.16e}
Vdrive out 0 PWL(0 {initial_voltage:.16e} {stop_time:.16e} {final_voltage:.16e})
{element}
.options method=gear maxord=2 reltol=1e-9 abstol=1e-15 vntol=1e-10
.control
set noaskquit
set wr_singlescale
set wr_vecnames
tran 1e-6 {stop_time:.16e} 0 1e-6
wrdata result.dat v(out) i(Vdrive)
quit
.endc
.end
"""
    netlist_path = tmp_path / f"capacitor_{formulation.lower()}.cir"
    netlist_path.write_text(netlist, encoding="ascii", newline="\n")
    completed = subprocess.run(
        [str(NGSPICE), "-n", "-o", f"{formulation}.log", netlist_path.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (tmp_path / f"{formulation}.log").read_text(
        encoding="utf-8", errors="replace"
    )
    data = np.loadtxt(tmp_path / "result.dat", skiprows=1, ndmin=2)
    sample_index = int(np.argmin(np.abs(data[:, 0] - 0.5 * stop_time)))
    simulated = abs(float(data[sample_index, -1]))
    sample_voltage = float(data[sample_index, 1])
    voltage_slope = (final_voltage - initial_voltage) / stop_time
    analytic = np.sqrt(sheath_k / sample_voltage) * voltage_slope
    return simulated, float(analytic)


@pytest.mark.skipif(not NGSPICE.is_file(), reason="specified ngspice is not installed")
@pytest.mark.parametrize("formulation", ["C", "Q"])
def test_behavioral_capacitor_matches_differential_capacitance(
    tmp_path: Path, formulation: str
) -> None:
    simulated, analytic = _run_capacitor_case(tmp_path, formulation)
    assert simulated == pytest.approx(analytic, rel=2.0e-4)
