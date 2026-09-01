"""Physics equations used by Schmidt et al. (2018)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import brentq

ELEMENTARY_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837015e-31
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
BOLTZMANN_J_K = 1.380649e-23
VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12


@dataclass(frozen=True)
class PlasmaParameters:
    """Circuit and global-model parameters for one density iterate."""

    electron_temperature_ev: float
    electron_density_m3: float
    neutral_density_m3: float
    mean_electron_speed_m_s: float
    bohm_speed_m_s: float
    momentum_collision_frequency_hz: float
    effective_collision_frequency_hz: float
    bulk_inductance_h: float
    bulk_resistance_ohm: float
    ion_current_powered_a: float
    ion_current_grounded_a: float
    electron_saturation_powered_a: float
    electron_saturation_grounded_a: float
    sheath_k_powered: float
    sheath_k_grounded: float
    equilibrium_sheath_voltage_v: float


def argon_rate_coefficients(electron_temperature_ev: float) -> Mapping[str, float]:
    """Gudmundsson fits used in the Schmidt global model.

    The fits are in m^3/s and use electron temperature in eV.
    """
    te = float(electron_temperature_ev)
    if not np.isfinite(te) or te <= 0.0:
        raise ValueError("electron temperature must be finite and positive")
    log_te = np.log(te)
    k_el = 2.336e-14 * te**1.609 * np.exp(
        0.0618 * log_te**2 - 0.1171 * log_te**3
    )
    k_ex = 2.48e-14 * te**0.33 * np.exp(-12.78 / te)
    k_iz = 2.34e-14 * te**0.59 * np.exp(-17.44 / te)
    return {"elastic": float(k_el), "excitation": float(k_ex), "ionization": float(k_iz)}


def neutral_density(pressure_pa: float, gas_temperature_k: float) -> float:
    """Ideal-gas neutral density in m^-3."""
    if pressure_pa <= 0.0 or gas_temperature_k <= 0.0:
        raise ValueError("pressure and gas temperature must be positive")
    return pressure_pa / (BOLTZMANN_J_K * gas_temperature_k)


def electron_temperature_from_particle_balance(config: Mapping[str, float]) -> float:
    """Solve Vp*n_g*Kiz(Te) = uB(Te)*(A_E+A_G)."""
    area_powered = float(config["powered_area_m2"])
    area_grounded = float(config["grounded_area_m2"])
    bulk_length = float(config["bulk_length_m"])
    return electron_temperature_from_particle_balance_geometry(
        config,
        plasma_volume_m3=area_powered * bulk_length,
        loss_area_m2=area_powered + area_grounded,
    )


def electron_temperature_from_particle_balance_geometry(
    config: Mapping[str, float],
    plasma_volume_m3: float,
    loss_area_m2: float,
) -> float:
    """Solve the global particle balance for arbitrary plasma geometry."""
    if plasma_volume_m3 <= 0.0 or loss_area_m2 <= 0.0:
        raise ValueError("plasma volume and loss area must be positive")
    argon_mass = float(config.get("argon_mass_amu", 39.948)) * ATOMIC_MASS_UNIT_KG
    n_gas = neutral_density(float(config["pressure_pa"]), float(config["gas_temperature_k"]))

    def residual(te_ev: float) -> float:
        k_iz = argon_rate_coefficients(te_ev)["ionization"]
        u_bohm = np.sqrt(ELEMENTARY_CHARGE_C * te_ev / argon_mass)
        return plasma_volume_m3 * n_gas * k_iz - u_bohm * loss_area_m2

    return float(brentq(residual, 0.2, 20.0, xtol=1e-12, rtol=1e-12))


def smooth_positive(voltage_v: np.ndarray | float, width_v: float) -> np.ndarray | float:
    """Smooth positive part that is finite and differentiable at zero."""
    if width_v <= 0.0:
        raise ValueError("regularization width must be positive")
    voltage = np.asarray(voltage_v, dtype=float)
    result = 0.5 * (voltage + np.sqrt(voltage * voltage + width_v * width_v))
    if result.ndim == 0:
        return float(result)
    return result


def regularized_sheath_capacitance(
    voltage_v: np.ndarray | float, sheath_k: float, voltage_width_v: float
) -> np.ndarray | float:
    """Regularized matrix-sheath differential capacitance."""
    voltage = np.asarray(voltage_v, dtype=float)
    soft_abs = np.sqrt(voltage * voltage + voltage_width_v * voltage_width_v)
    result = np.sqrt(sheath_k / soft_abs)
    if result.ndim == 0:
        return float(result)
    return result


def matrix_sheath_charge(sheath_voltage_v: float, sheath_k: float) -> float:
    """Physical matrix-sheath charge Q=sqrt(K*Vs), valid for Vs>0.

    Here K=2*e*n*epsilon_0*A^2. The paper's stated capacitance sqrt(K/Vs)
    is Q/Vs (a secant capacitance), while dQ/dVs is exactly half that value.
    """
    if sheath_voltage_v <= 0.0 or sheath_k <= 0.0:
        raise ValueError("matrix-sheath voltage and K must be positive")
    return float(np.sqrt(sheath_k * sheath_voltage_v))


def matrix_sheath_secant_capacitance(
    sheath_voltage_v: float, sheath_k: float
) -> float:
    """Return Q/V for the matrix sheath, matching the printed paper formula."""
    return matrix_sheath_charge(sheath_voltage_v, sheath_k) / sheath_voltage_v


def matrix_sheath_differential_capacitance(
    sheath_voltage_v: float, sheath_k: float
) -> float:
    """Return dQ/dV for the physical matrix-sheath charge law."""
    return 0.5 * matrix_sheath_secant_capacitance(sheath_voltage_v, sheath_k)


def regularized_electron_current(
    sheath_voltage_v: np.ndarray | float,
    saturation_current_a: float,
    electron_temperature_ev: float,
    voltage_width_v: float,
) -> np.ndarray | float:
    """Bounded electron current with a smooth non-negative sheath barrier."""
    barrier = smooth_positive(sheath_voltage_v, voltage_width_v)
    result = saturation_current_a * np.exp(-np.asarray(barrier) / electron_temperature_ev)
    if np.asarray(result).ndim == 0:
        return float(result)
    return result


def compute_plasma_parameters(
    config: Mapping[str, float], electron_density_m3: float, electron_temperature_ev: float
) -> PlasmaParameters:
    """Calculate the density-dependent equivalent-circuit parameters."""
    if electron_density_m3 <= 0.0 or not np.isfinite(electron_density_m3):
        raise ValueError("electron density must be finite and positive")
    area_powered = float(config["powered_area_m2"])
    area_grounded = float(config["grounded_area_m2"])
    bulk_length = float(config["bulk_length_m"])
    argon_mass = float(config.get("argon_mass_amu", 39.948)) * ATOMIC_MASS_UNIT_KG
    n_gas = neutral_density(float(config["pressure_pa"]), float(config["gas_temperature_k"]))
    rates = argon_rate_coefficients(electron_temperature_ev)
    v_mean = np.sqrt(
        8.0 * ELEMENTARY_CHARGE_C * electron_temperature_ev
        / (np.pi * ELECTRON_MASS_KG)
    )
    u_bohm = np.sqrt(ELEMENTARY_CHARGE_C * electron_temperature_ev / argon_mass)
    nu_m = n_gas * rates["elastic"]
    nu_eff = nu_m + v_mean / bulk_length
    l_plasma = (
        bulk_length
        * ELECTRON_MASS_KG
        / (ELEMENTARY_CHARGE_C**2 * electron_density_m3 * area_powered)
    )
    r_plasma = nu_eff * l_plasma
    current_scale = ELEMENTARY_CHARGE_C * electron_density_m3
    equilibrium_sheath_voltage = electron_temperature_ev * np.log(v_mean / u_bohm)

    return PlasmaParameters(
        electron_temperature_ev=electron_temperature_ev,
        electron_density_m3=electron_density_m3,
        neutral_density_m3=n_gas,
        mean_electron_speed_m_s=float(v_mean),
        bohm_speed_m_s=float(u_bohm),
        momentum_collision_frequency_hz=float(nu_m),
        effective_collision_frequency_hz=float(nu_eff),
        bulk_inductance_h=float(l_plasma),
        bulk_resistance_ohm=float(r_plasma),
        ion_current_powered_a=float(area_powered * current_scale * u_bohm),
        ion_current_grounded_a=float(area_grounded * current_scale * u_bohm),
        electron_saturation_powered_a=float(area_powered * current_scale * v_mean),
        electron_saturation_grounded_a=float(area_grounded * current_scale * v_mean),
        sheath_k_powered=float(
            2.0
            * ELEMENTARY_CHARGE_C
            * electron_density_m3
            * VACUUM_PERMITTIVITY_F_M
            * area_powered**2
        ),
        sheath_k_grounded=float(
            2.0
            * ELEMENTARY_CHARGE_C
            * electron_density_m3
            * VACUUM_PERMITTIVITY_F_M
            * area_grounded**2
        ),
        equilibrium_sheath_voltage_v=float(equilibrium_sheath_voltage),
    )


def collisional_energy_loss_ev(electron_temperature_ev: float, argon_mass_amu: float) -> float:
    """Energy lost in elastic, excitation, and ionization events per created pair."""
    rates = argon_rate_coefficients(electron_temperature_ev)
    argon_mass = argon_mass_amu * ATOMIC_MASS_UNIT_KG
    elastic_loss_ev = 3.0 * ELECTRON_MASS_KG / argon_mass * electron_temperature_ev
    return float(
        15.76
        + rates["excitation"] / rates["ionization"] * 12.14
        + rates["elastic"] / rates["ionization"] * elastic_loss_ev
    )


def density_from_power_balance(
    config: Mapping[str, float],
    electron_temperature_ev: float,
    absorbed_power_w: float,
    mean_sheath_powered_v: float,
    mean_sheath_grounded_v: float,
) -> float:
    """Equation (5) of Schmidt et al., solved for electron density."""
    area_powered = float(config["powered_area_m2"])
    area_grounded = float(config["grounded_area_m2"])
    volume = area_powered * float(config["bulk_length_m"])
    return density_from_power_balance_surfaces(
        config,
        electron_temperature_ev,
        absorbed_power_w,
        surface_areas_m2=(area_powered, area_grounded),
        mean_sheath_voltages_v=(
            mean_sheath_powered_v,
            mean_sheath_grounded_v,
        ),
        plasma_volume_m3=volume,
    )


def density_from_power_balance_surfaces(
    config: Mapping[str, float],
    electron_temperature_ev: float,
    absorbed_power_w: float,
    surface_areas_m2: tuple[float, ...],
    mean_sheath_voltages_v: tuple[float, ...],
    plasma_volume_m3: float,
) -> float:
    """Global power balance for any number of independently biased surfaces."""
    if absorbed_power_w <= 0.0 or not np.isfinite(absorbed_power_w):
        raise ValueError("absorbed power must be finite and positive")
    if plasma_volume_m3 <= 0.0:
        raise ValueError("plasma volume must be positive")
    if len(surface_areas_m2) != len(mean_sheath_voltages_v):
        raise ValueError("surface areas and sheath voltages must have equal length")
    if not surface_areas_m2 or any(area <= 0.0 for area in surface_areas_m2):
        raise ValueError("surface areas must be non-empty and positive")
    total_area = float(sum(surface_areas_m2))
    n_gas = neutral_density(float(config["pressure_pa"]), float(config["gas_temperature_k"]))
    k_iz = argon_rate_coefficients(electron_temperature_ev)["ionization"]
    collision_ev = collisional_energy_loss_ev(
        electron_temperature_ev, float(config.get("argon_mass_amu", 39.948))
    )
    area_weighted_ion_energy_ev = sum(
        area / total_area
        * (max(voltage, 0.0) + 0.5 * electron_temperature_ev)
        for area, voltage in zip(
            surface_areas_m2,
            mean_sheath_voltages_v,
            strict=True,
        )
    )
    energy_per_pair_ev = (
        collision_ev
        + 2.0 * electron_temperature_ev
        + area_weighted_ion_energy_ev
    )
    denominator = (
        plasma_volume_m3
        * n_gas
        * k_iz
        * energy_per_pair_ev
        * ELEMENTARY_CHARGE_C
    )
    return float(absorbed_power_w / denominator)
