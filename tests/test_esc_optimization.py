from __future__ import annotations

import numpy as np
import pytest

from plasma_circuit.esc_optimization import (
    analytic_l_match,
    deembed_l_match_load,
    reflection_coefficient,
)


def test_reflection_coefficient_is_zero_at_match() -> None:
    assert reflection_coefficient(50.0 + 0.0j, 50.0) == pytest.approx(0.0j)


def test_analytic_l_match_transforms_capacitive_load_to_50_ohm() -> None:
    frequency_hz = 13.56e6
    omega = 2.0 * np.pi * frequency_hz
    load = 16.9 - 144.9j
    inductance, capacitance = analytic_l_match(load, 50.0, omega)
    branch = load + 1j * omega * inductance
    input_impedance = 1.0 / (1.0 / branch + 1j * omega * capacitance)
    assert input_impedance.real == pytest.approx(50.0, rel=1.0e-12)
    assert input_impedance.imag == pytest.approx(0.0, abs=1.0e-11)


def test_deembedding_recovers_original_load() -> None:
    frequency_hz = 13.56e6
    omega = 2.0 * np.pi * frequency_hz
    load = 11.0 - 455.0j
    inductance = 5.5e-6
    capacitance = 440.0e-12
    input_impedance = 1.0 / (
        1.0 / (load + 1j * omega * inductance) + 1j * omega * capacitance
    )
    recovered = deembed_l_match_load(input_impedance, inductance, capacitance, omega)
    assert recovered == pytest.approx(load)


def test_analytic_l_match_rejects_load_resistance_above_reference() -> None:
    with pytest.raises(ValueError, match="Re\\(Zload\\)"):
        analytic_l_match(60.0 - 100.0j, 50.0, 2.0 * np.pi * 13.56e6)
