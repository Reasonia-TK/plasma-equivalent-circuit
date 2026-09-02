"""Matching-condition verification for the Qucs-S one-zone plasma circuit."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from plasma_circuit.qucs_one_zone import (
    QucsOneZoneCoupledResult,
    load_qucs_one_zone_config,
    qucs_one_zone_result_to_dict,
    series_inductor_esr_ohm,
    solve_qucs_one_zone_coupled,
)


def reflection_coefficient(impedance: complex, reference_ohm: float) -> complex:
    """Return the fundamental voltage-wave reflection coefficient."""
    denominator = impedance + reference_ohm
    if abs(denominator) < 1.0e-30:
        return complex(np.inf)
    return (impedance - reference_ohm) / denominator


def external_impedance(input_impedance: complex, source_resistance_ohm: float) -> complex:
    """De-embed the series source resistor from the measured source impedance."""
    return input_impedance - source_resistance_ohm


def load_matching_search_config(path: str | Path) -> dict[str, Any]:
    """Load a selected matching design and its strict verification settings."""
    search_path = Path(path)
    search = json.loads(search_path.read_text(encoding="utf-8"))
    required = {
        "base_config",
        "baseline_summary",
        "reference_impedance_ohm",
        "selected_design",
        "verification_transient",
        "verification_coupling",
    }
    missing = sorted(required - search.keys())
    if missing:
        raise ValueError(f"missing Qucs matching-search fields: {missing}")
    for field in ("base_config", "baseline_summary"):
        value = Path(search[field])
        resolved = value if value.is_absolute() else search_path.parent / value
        resolved = resolved.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Qucs matching-search {field} not found: {resolved}")
        search[f"resolved_{field}"] = str(resolved)
    return search


def apply_selected_design(
    base_config: Mapping[str, Any], search: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a one-zone configuration with the selected finite-Q L match."""
    config = copy.deepcopy(base_config)
    design = search["selected_design"]
    inductance_h = float(design["series_inductance_h"])
    series_capacitance_f = float(design["series_capacitance_f"])
    shunt_capacitance_f = float(design["shunt_capacitance_f"])
    quality_factor = float(design["series_inductor_quality_factor"])
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in (
            inductance_h,
            series_capacitance_f,
            shunt_capacitance_f,
            quality_factor,
        )
    ):
        raise ValueError("selected matching values must be finite and positive")
    config["qucs_netlist"]["component_overrides"] = {
        "L1": inductance_h,
        "C1": series_capacitance_f,
    }
    config["matching"] = {
        "shunt_capacitance_f": shunt_capacitance_f,
        "series_inductor_device": "L1",
        "series_inductor_quality_factor": quality_factor,
    }
    config["transient"].update(search["verification_transient"])
    config["coupling"].update(search["verification_coupling"])
    config["source_ramp_cycles"] = float(search.get("source_ramp_cycles", 60.0))
    return config


def _matching_metrics(
    result: QucsOneZoneCoupledResult,
    config: Mapping[str, Any],
    reference_ohm: float,
) -> dict[str, Any]:
    metrics = result.final_simulation.metrics
    input_impedance = complex(
        metrics.input_impedance_real_ohm,
        metrics.input_impedance_imag_ohm,
    )
    external = external_impedance(
        input_impedance,
        float(config["source_resistance_ohm"]),
    )
    reflection = reflection_coefficient(external, reference_ohm)
    available_power = (
        float(config["source_amplitude_v"]) ** 2
        / (8.0 * float(config["source_resistance_ohm"]))
    )
    return {
        "external_impedance_real_ohm": float(external.real),
        "external_impedance_imag_ohm": float(external.imag),
        "reflection_real": float(reflection.real),
        "reflection_imag": float(reflection.imag),
        "reflection_magnitude": float(abs(reflection)),
        "available_source_power_w": float(available_power),
        "plasma_to_ideal_source_power_efficiency": float(
            metrics.absorbed_power_w / max(metrics.source_delivered_power_w, 1.0e-30)
        ),
        "plasma_to_available_power_ratio": float(
            metrics.absorbed_power_w / available_power
        ),
        "series_inductor_esr_ohm": series_inductor_esr_ohm(config),
        "cycle_l2_max": float(
            max(metrics.cycle_l2_voltage, metrics.cycle_l2_current)
        ),
    }


def verify_selected_matching_design(
    search_path: Path,
    output_directory: Path,
) -> tuple[dict[str, Any], QucsOneZoneCoupledResult]:
    """Run the selected self-consistent design and compare it with the baseline."""
    search = load_matching_search_config(search_path)
    base_config = load_qucs_one_zone_config(search["resolved_base_config"])
    selected_config = apply_selected_design(base_config, search)
    result = solve_qucs_one_zone_coupled(selected_config, output_directory)
    reference = float(search["reference_impedance_ohm"])
    baseline_payload = json.loads(
        Path(search["resolved_baseline_summary"]).read_text(encoding="utf-8")
    )
    baseline_result = baseline_payload["result"]
    baseline_metrics = baseline_result["metrics"]
    baseline_external = external_impedance(
        complex(
            baseline_metrics["input_impedance_real_ohm"],
            baseline_metrics["input_impedance_imag_ohm"],
        ),
        float(base_config["source_resistance_ohm"]),
    )
    baseline_reflection = reflection_coefficient(baseline_external, reference)
    selected_result = qucs_one_zone_result_to_dict(result)
    selected_result["matching"] = _matching_metrics(
        result,
        selected_config,
        reference,
    )
    output = {
        "description": search.get("description", "Qucs-S RLC matching verification"),
        "search_config": str(search_path),
        "base_config": str(search["base_config"]),
        "frequency_hz": float(selected_config["frequency_hz"]),
        "reference_impedance_ohm": reference,
        "selected_design": search["selected_design"],
        "baseline": {
            "electron_temperature_ev": baseline_result["electron_temperature_ev"],
            "electron_density_m3": baseline_result["electron_density_m3"],
            "metrics": baseline_metrics,
            "matching": {
                "external_impedance_real_ohm": float(baseline_external.real),
                "external_impedance_imag_ohm": float(baseline_external.imag),
                "reflection_magnitude": float(abs(baseline_reflection)),
            },
        },
        "selected": selected_result,
        "gates": {
            "converged": bool(result.converged),
            "density_residual_below_tolerance": bool(
                result.final_balance_relative_residual
                <= float(selected_config["coupling"]["relative_tolerance"])
            ),
            "cycle_l2_below_tolerance": bool(
                selected_result["matching"]["cycle_l2_max"]
                <= float(selected_config["coupling"]["cycle_l2_tolerance"])
            ),
            "power_balance_below_1e_3": bool(
                result.final_simulation.metrics.power_balance_relative_error < 1.0e-3
            ),
        },
    }
    return output, result
