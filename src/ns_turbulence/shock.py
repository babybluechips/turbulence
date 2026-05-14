"""
Shock-capturing methods for compressible turbulence.

Provides:
    - Exponential spectral filter (Vandeven-type)
    - Adaptive artificial viscosity (Cook-Cabot / Mani-Larsson)
    - Ducros dilatation-based shock sensor

These methods stabilize pseudo-spectral simulations when shocks
or strong compressions develop, without contaminating the vortical
modes that carry the turbulence physics.
"""

from __future__ import annotations

import torch

from .config import ShockConfig
from .fields import ConservativeFields, PrimitiveFields
from .spectral import SpectralOperator


class ShockCapture:
    """Shock-capturing filter and artificial viscosity manager."""

    def __init__(
        self,
        config: ShockConfig,
        spec: SpectralOperator,
        gamma: float,
    ) -> None:
        self.config = config
        self.spec = spec
        self.gamma = gamma

    def apply_spectral_filter(self, cons: ConservativeFields) -> ConservativeFields:
        """Apply exponential spectral filter to all conservative variables.

        This removes energy at the grid scale to prevent Gibbs-type
        oscillations near discontinuities.
        """
        if not self.config.enabled:
            return cons

        order = self.config.filter_order
        cutoff = self.config.filter_cutoff
        filt = self.spec.exponential_filter

        return ConservativeFields(
            rho=filt(cons.rho, order=order, cutoff=cutoff),
            rhou=filt(cons.rhou, order=order, cutoff=cutoff),
            rhov=filt(cons.rhov, order=order, cutoff=cutoff),
            rhow=filt(cons.rhow, order=order, cutoff=cutoff),
            E=filt(cons.E, order=order, cutoff=cutoff),
        )

    def ducros_sensor(self, prim: PrimitiveFields) -> torch.Tensor:
        """Ducros shock sensor: identifies shocks vs vortical structures.

        phi = (div(u))^2 / ((div(u))^2 + |curl(u)|^2 + eps)

        phi ≈ 1 in shocks (compressive), phi ≈ 0 in vortices.
        """
        div_u = self.spec.divergence(prim.u, prim.v, prim.w)
        wx, wy, wz = self.spec.curl(prim.u, prim.v, prim.w)

        div_sq = div_u**2
        curl_sq = wx**2 + wy**2 + wz**2
        eps = 1e-30

        return div_sq / (div_sq + curl_sq + eps)

    def artificial_viscosity(
        self, cons: ConservativeFields, prim: PrimitiveFields
    ) -> ConservativeFields:
        """Compute artificial viscosity contribution.

        Uses the Cook-Cabot / Mani-Larsson approach:
        mu_av = C_av * rho * h^2 * max(-div(u), 0)

        where h is the grid spacing and C_av is the coefficient.
        Optionally modulated by the Ducros sensor to avoid adding
        viscosity in vortical regions.
        """
        if not self.config.artificial_viscosity:
            zero = torch.zeros_like(cons.rho)
            return ConservativeFields(zero, zero, zero, zero, zero)

        spec = self.spec
        u, v, w = prim.u, prim.v, prim.w

        # Negative dilatation (compression)
        div_u = spec.divergence(u, v, w)
        neg_div = torch.clamp(-div_u, min=0.0)

        # Grid spacing (use geometric mean)
        h = (spec.lx / spec.nx + spec.ly / spec.ny + spec.lz / spec.nz) / 3.0

        # AV coefficient
        mu_av = self.config.av_coefficient * prim.rho * h**2 * neg_div

        # Apply Ducros sensor to localize to shocks
        if self.config.ducros_sensor:
            phi = self.ducros_sensor(prim)
            mu_av = mu_av * phi

        # Compute AV stress divergence (simplified: Laplacian form)
        # d/dx_j(mu_av * du_i/dx_j) ≈ div(mu_av * grad(u_i))
        # Use product rule: mu_av * lap(u_i) + grad(mu_av) . grad(u_i)
        drhou_av = _av_diffusion(spec, mu_av, u)
        drhov_av = _av_diffusion(spec, mu_av, v)
        drhow_av = _av_diffusion(spec, mu_av, w)

        # Energy: AV heating ≈ mu_av * |S|^2 (simplified)
        S11, S12, S13, S22, S23, S33 = spec.strain_rate_tensor(u, v, w)
        dissipation = 2.0 * mu_av * (
            S11**2 + S22**2 + S33**2 + 2.0 * (S12**2 + S13**2 + S23**2)
        )

        zero = torch.zeros_like(cons.rho)
        return ConservativeFields(
            rho=zero,
            rhou=drhou_av,
            rhov=drhov_av,
            rhow=drhow_av,
            E=dissipation,
        )


def _av_diffusion(
    spec: SpectralOperator, mu: torch.Tensor, f: torch.Tensor
) -> torch.Tensor:
    """Compute div(mu * grad(f)) for artificial viscosity."""
    dfdx, dfdy, dfdz = spec.gradient(f)
    return (
        spec.ddx(mu * dfdx)
        + spec.ddy(mu * dfdy)
        + spec.ddz(mu * dfdz)
    )
