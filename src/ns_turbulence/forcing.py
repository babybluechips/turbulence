"""
Turbulence forcing schemes for sustained compressible turbulence DNS.

Provides spectral-band forcing that injects energy at large scales
to maintain a statistically stationary turbulent state. The forcing
is purely solenoidal (divergence-free) by default, ensuring it does
not inject acoustic energy.
"""

from __future__ import annotations

import torch

from .config import ForcingConfig
from .spectral import SpectralOperator


class SpectralForcing:
    """Low-wavenumber spectral band forcing.

    Injects energy at wavenumbers k_min <= |k| <= k_max by rescaling
    the velocity field in those shells to maintain a target energy
    injection rate.
    """

    def __init__(
        self,
        config: ForcingConfig,
        spec: SpectralOperator,
    ) -> None:
        self.config = config
        self.spec = spec

        # Build forcing mask: 1 in [k_min, k_max], 0 elsewhere
        k_mag = spec.k_mag
        self.force_mask = (
            (k_mag >= config.k_min) & (k_mag <= config.k_max)
        ).to(spec.dtype)

        # Count modes in the forcing band for normalization
        self.n_forced_modes = max(float(self.force_mask.sum()), 1.0)

    def compute_forcing(
        self,
        rhou: torch.Tensor,
        rhov: torch.Tensor,
        rhow: torch.Tensor,
        rho: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute solenoidal forcing term for momentum equations.

        Returns (F_x, F_y, F_z) to be added to the momentum RHS.

        Method: project forcing onto solenoidal modes using Leray projector
        P_ij = delta_ij - k_i*k_j / |k|^2
        """
        if not self.config.enabled:
            zero = torch.zeros_like(rhou)
            return zero, zero, zero

        spec = self.spec
        eps = self.config.energy_injection_rate

        # Current velocity in forced band
        u = rhou / rho
        v = rhov / rho
        w = rhow / rho

        u_hat = spec.fft(u)
        v_hat = spec.fft(v)
        w_hat = spec.fft(w)

        # Mask to forcing band
        u_f = u_hat * self.force_mask
        v_f = v_hat * self.force_mask
        w_f = w_hat * self.force_mask

        # Current energy in forced band
        n_total = spec.nx * spec.ny * spec.nz
        E_f = float(
            (torch.abs(u_f)**2 + torch.abs(v_f)**2 + torch.abs(w_f)**2).sum()
        ) / n_total**2

        if E_f < 1e-30:
            # No energy in band yet — seed with random solenoidal field
            return self._seed_forcing(rho, dt)

        # Scaling factor to inject eps*dt of energy
        # New energy: E_f * alpha^2 = E_f + eps * dt
        alpha = (1.0 + eps * dt / E_f) ** 0.5

        # Force = rho * (alpha - 1) * u_forced / dt
        du_hat = (alpha - 1.0) * u_f
        dv_hat = (alpha - 1.0) * v_f
        dw_hat = (alpha - 1.0) * w_f

        # Leray projection (ensure solenoidality)
        du_hat, dv_hat, dw_hat = self._leray_project(du_hat, dv_hat, dw_hat)

        Fx = rho * spec.ifft(du_hat) / max(dt, 1e-30)
        Fy = rho * spec.ifft(dv_hat) / max(dt, 1e-30)
        Fz = rho * spec.ifft(dw_hat) / max(dt, 1e-30)

        return Fx, Fy, Fz

    def _leray_project(
        self,
        fx_hat: torch.Tensor,
        fy_hat: torch.Tensor,
        fz_hat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Leray (Helmholtz) projection to remove divergent component.

        P_ij f_j = f_i - k_i (k_j f_j) / |k|^2
        """
        spec = self.spec
        k_sq = spec.k_sq.clone()
        k_sq[0, 0, 0] = 1.0  # avoid division by zero at k=0

        k_dot_f = spec.kx * fx_hat + spec.ky * fy_hat + spec.kz * fz_hat
        proj = k_dot_f / k_sq

        fx_sol = fx_hat - spec.kx * proj
        fy_sol = fy_hat - spec.ky * proj
        fz_sol = fz_hat - spec.kz * proj

        return fx_sol, fy_sol, fz_sol

    def _seed_forcing(
        self, rho: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Seed forcing band with random solenoidal perturbations."""
        spec = self.spec
        shape = spec.kx.shape

        # Random complex amplitudes in forcing band
        fx_hat = torch.randn(shape, device=spec.device, dtype=spec.dtype) + \
            1j * torch.randn(shape, device=spec.device, dtype=spec.dtype)
        fy_hat = torch.randn(shape, device=spec.device, dtype=spec.dtype) + \
            1j * torch.randn(shape, device=spec.device, dtype=spec.dtype)
        fz_hat = torch.randn(shape, device=spec.device, dtype=spec.dtype) + \
            1j * torch.randn(shape, device=spec.device, dtype=spec.dtype)

        fx_hat *= self.force_mask
        fy_hat *= self.force_mask
        fz_hat *= self.force_mask

        # Leray project
        fx_hat, fy_hat, fz_hat = self._leray_project(fx_hat, fy_hat, fz_hat)

        # Normalize to target injection rate
        n_total = spec.nx * spec.ny * spec.nz
        E_seed = float(
            (torch.abs(fx_hat)**2 + torch.abs(fy_hat)**2 + torch.abs(fz_hat)**2).sum()
        ) / n_total**2

        if E_seed > 1e-30:
            scale = (self.config.energy_injection_rate * dt / E_seed) ** 0.5
            fx_hat *= scale
            fy_hat *= scale
            fz_hat *= scale

        Fx = rho * spec.ifft(fx_hat) / max(dt, 1e-30)
        Fy = rho * spec.ifft(fy_hat) / max(dt, 1e-30)
        Fz = rho * spec.ifft(fz_hat) / max(dt, 1e-30)

        return Fx, Fy, Fz
