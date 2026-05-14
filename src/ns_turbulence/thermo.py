"""
Real-gas thermochemistry for high-Mach compressible flows.

Models temperature-dependent specific heat ratio gamma(T) accounting for
vibrational excitation, dissociation, and ionization. Provides Sutherland
and Blottner viscosity laws, and post-shock thermodynamic states.

Designed for GPU execution: all functions accept and return torch.Tensors.
"""

from __future__ import annotations

import math

import torch

# ---- Specific heat ratio models ----

def gamma_effective(T: torch.Tensor, T_ref: float = 300.0) -> torch.Tensor:
    """Temperature-dependent effective specific heat ratio.

    gamma(T) decreases with temperature due to:
      - Vibrational excitation (T > 800K for N2)
      - Dissociation (T > 2000K)
      - Ionization (T > 9000K)

    Model:
        gamma = 1.4 - 0.1 * sigma(T/2000 - 1) - 0.05 * sigma(T/5000 - 1)

    where sigma is a smooth step (tanh-based).
    """
    T_dim = T * T_ref  # convert to dimensional if non-dim
    vib = 0.1 * torch.sigmoid(6.0 * (T_dim / 2000.0 - 1.0))
    dissoc = 0.05 * torch.sigmoid(6.0 * (T_dim / 5000.0 - 1.0))
    ion = 0.03 * torch.sigmoid(6.0 * (T_dim / 9000.0 - 1.0))
    return 1.4 - vib - dissoc - ion


def ideal_gas_pressure(
    rho: torch.Tensor, T: torch.Tensor, gamma: float = 1.4
) -> torch.Tensor:
    """Ideal gas equation of state: p = rho * T (non-dimensional, R=1)."""
    return rho * T


def real_gas_pressure(
    rho: torch.Tensor, T: torch.Tensor, T_ref: float = 300.0
) -> torch.Tensor:
    """Real-gas pressure using temperature-dependent gamma."""
    _g = gamma_effective(T, T_ref)  # noqa: F841 — computed for future caloric EOS
    return rho * T  # EOS is p = rho * R * T regardless of gamma


# ---- Viscosity models ----

def sutherland_viscosity(
    T: torch.Tensor, T_ref: float = 300.0, mu_ref: float = 1.716e-5
) -> torch.Tensor:
    """Sutherland's law for dynamic viscosity.

    mu(T) = mu_ref * (T/T_ref)^{3/2} * (T_ref + S) / (T + S)

    where S = 110.4 K for air.
    Returns non-dimensional viscosity (mu / mu_ref).
    """
    T_dim = T * T_ref
    S = 110.4
    return (T_dim / T_ref) ** 1.5 * (T_ref + S) / (T_dim + S)


def blottner_viscosity(
    T: torch.Tensor, species: str = "N2", T_ref: float = 300.0
) -> torch.Tensor:
    """Blottner curve fit for high-temperature viscosity.

    ln(mu) = (A * ln(T) + B) * ln(T) + C

    Returns non-dimensional viscosity relative to reference.
    """
    coefficients = {
        "N2": (0.0268142, 0.3177838, -11.3155513),
        "O2": (0.0449290, -0.0826158, -9.2019475),
        "N":  (0.0115572, 0.6031679, -12.4327495),
        "O":  (0.0203144, 0.4294404, -11.6031403),
        "NO": (0.0436378, -0.0335511, -9.5767430),
        "air": (0.0268142, 0.3177838, -11.3155513),
    }
    if species not in coefficients:
        raise ValueError(f"Unknown species '{species}'. Available: {list(coefficients.keys())}")

    A, B, C = coefficients[species]
    T_dim = torch.clamp(T * T_ref, min=100.0)
    ln_T = torch.log(T_dim)
    ln_mu = (A * ln_T + B) * ln_T + C

    # Reference viscosity at T_ref
    ln_T_ref = math.log(T_ref)
    ln_mu_ref = (A * ln_T_ref + B) * ln_T_ref + C

    return torch.exp(ln_mu - ln_mu_ref)


# ---- Shock relations ----

def post_shock_temperature(
    Ma: float, T_pre: float = 1.0, gamma: float = 1.4
) -> float:
    """Normal shock temperature ratio from Rankine-Hugoniot.

    T_post/T_pre = [2*gamma*Ma^2 - (gamma-1)] * [(gamma-1)*Ma^2 + 2]
                    / [(gamma+1)^2 * Ma^2]
    """
    g = gamma
    M2 = Ma**2
    numer = (2.0 * g * M2 - (g - 1.0)) * ((g - 1.0) * M2 + 2.0)
    denom = (g + 1.0) ** 2 * M2
    return T_pre * numer / denom


def post_shock_density_ratio(Ma: float, gamma: float = 1.4) -> float:
    """Normal shock density ratio rho_post/rho_pre."""
    g = gamma
    M2 = Ma**2
    return (g + 1.0) * M2 / ((g - 1.0) * M2 + 2.0)


def post_shock_pressure_ratio(Ma: float, gamma: float = 1.4) -> float:
    """Normal shock pressure ratio p_post/p_pre."""
    g = gamma
    M2 = Ma**2
    return (2.0 * g * M2 - (g - 1.0)) / (g + 1.0)


# ---- Turbulence-thermochemistry coupling ----

def turbulent_mach_number(u_rms: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Turbulent Mach number: M_t = u_rms / c."""
    return u_rms / (c + 1e-30)


def dilatation_fraction(M_t: torch.Tensor) -> torch.Tensor:
    """Dilatation fraction: chi = M_t^2 / (1 + M_t^2).

    Sarkar's model for the fraction of dissipation due to compressive modes.
    """
    return M_t**2 / (1.0 + M_t**2)


def compressible_dissipation_ratio(M_t: torch.Tensor) -> torch.Tensor:
    """Ratio of compressible to solenoidal dissipation.

    epsilon_c / epsilon_s ≈ alpha_1 * M_t^2 (Sarkar 1992)
    """
    alpha_1 = 1.0  # model constant
    return alpha_1 * M_t**2
