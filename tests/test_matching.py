from __future__ import annotations

import numpy as np
import pytest

from plasma_circuit.coupling import matching_capacitances_from_load


def test_matching_capacitances_transform_complex_load_to_50_ohm() -> None:
    frequency = 13.56e6
    source_resistance = 50.0
    inductance = 1.5e-6
    load = 1.069477304 - 47.818227j
    c1, c2 = matching_capacitances_from_load(
        load, frequency, inductance, source_resistance
    )
    omega = 2.0 * np.pi * frequency
    series_branch = load + 1j * omega * inductance + 1.0 / (1j * omega * c2)
    input_impedance = 1.0 / (1.0 / series_branch + 1j * omega * c1)
    assert c1 == pytest.approx(1.59e-9, rel=0.02)
    assert c2 == pytest.approx(1.61e-10, rel=0.02)
    assert input_impedance.real == pytest.approx(50.0, rel=1e-10)
    assert input_impedance.imag == pytest.approx(0.0, abs=1e-10)
