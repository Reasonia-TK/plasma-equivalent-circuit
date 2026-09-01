"""End-to-end ngspice check against a series-RLC analytical solution."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest


NGSPICE = Path(r"C:\Spice64\bin\ngspice_con.exe")


@pytest.mark.skipif(not NGSPICE.is_file(), reason="specified ngspice is not installed")
def test_series_rlc_impedance_and_power(tmp_path: Path) -> None:
    frequency = 1.0e6
    resistance = 10.0
    inductance = 1.0e-6
    capacitance = 1.0e-9
    period = 1.0 / frequency
    netlist = f"""Series RLC analytical validation
Vdrive source 0 SIN(0 1 {frequency:.16e})
Vsense source load 0
Rload load rnode {resistance:.16e}
Lload rnode lnode {inductance:.16e}
Cload lnode 0 {capacitance:.16e}
.options method=gear maxord=2 reltol=1e-8 abstol=1e-12 vntol=1e-9
.control
set noaskquit
set wr_singlescale
set wr_vecnames
tran {period / 500:.16e} {100 * period:.16e} {80 * period:.16e} {period / 500:.16e}
wrdata result.dat v(source) i(Vsense)
quit
.endc
.end
"""
    path = tmp_path / "series_rlc.cir"
    path.write_text(netlist, encoding="ascii", newline="\n")
    completed = subprocess.run(
        [str(NGSPICE), "-n", "-o", "ngspice.log", path.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0
    data = np.loadtxt(tmp_path / "result.dat", skiprows=1, ndmin=2)
    time_s, voltage, current = data[:, 0], data[:, 1], data[:, 2]
    duration = time_s[-1] - time_s[0]
    kernel = np.exp(-2j * np.pi * frequency * time_s)
    v_phasor = 2.0 / duration * np.trapezoid(voltage * kernel, time_s)
    i_phasor = 2.0 / duration * np.trapezoid(current * kernel, time_s)
    simulated_impedance = v_phasor / i_phasor
    omega = 2.0 * np.pi * frequency
    analytic_impedance = resistance + 1j * (
        omega * inductance - 1.0 / (omega * capacitance)
    )
    assert simulated_impedance.real == pytest.approx(analytic_impedance.real, rel=1e-3)
    assert simulated_impedance.imag == pytest.approx(analytic_impedance.imag, rel=1e-3)
    simulated_power = np.trapezoid(current**2 * resistance, time_s) / duration
    analytic_current_amplitude = 1.0 / abs(analytic_impedance)
    analytic_power = 0.5 * analytic_current_amplitude**2 * resistance
    assert simulated_power == pytest.approx(analytic_power, rel=1e-3)
