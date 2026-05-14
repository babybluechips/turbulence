"""
Pseudo-spectral operators on GPU via torch.fft.

Provides FFT-based spatial derivatives, Laplacian, spectral filtering,
and 2/3-rule dealiasing — all operating on 3D periodic domains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class SpectralOperator:
    """Cached wavenumber arrays and spectral tools for a 3D periodic grid.

    All internal tensors live on `device` and are created once at construction.
    Derivatives and filters then run entirely on GPU with zero host round-trips.
    """

    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self) -> None:
        # Wavenumber vectors
        kx = torch.fft.fftfreq(self.nx, d=self.lx / (2 * math.pi * self.nx),
                                device=self.device).to(self.dtype)
        ky = torch.fft.fftfreq(self.ny, d=self.ly / (2 * math.pi * self.ny),
                                device=self.device).to(self.dtype)
        kz = torch.fft.rfftfreq(self.nz, d=self.lz / (2 * math.pi * self.nz),
                                  device=self.device).to(self.dtype)

        # 3D wavenumber grids: (nx, ny, nzh)
        self.kx, self.ky, self.kz = torch.meshgrid(kx, ky, kz, indexing="ij")

        # |k|^2 for Laplacian and filtering
        self.k_sq = self.kx**2 + self.ky**2 + self.kz**2
        self.k_mag = torch.sqrt(self.k_sq)

        # Maximum wavenumber for normalization
        self.k_max = max(self.nx // 2, self.ny // 2, self.nz // 2)

        # 2/3 dealiasing mask
        kx_max = self.nx // 2
        ky_max = self.ny // 2
        kz_max = self.nz // 2
        self.dealias_mask = (
            (torch.abs(self.kx) <= 2 * kx_max / 3)
            & (torch.abs(self.ky) <= 2 * ky_max / 3)
            & (torch.abs(self.kz) <= 2 * kz_max / 3)
        ).to(self.dtype)

    @classmethod
    def from_config(cls, config) -> SpectralOperator:
        """Construct from a SimConfig."""
        return cls(
            nx=config.grid.nx,
            ny=config.grid.ny,
            nz=config.grid.nz,
            lx=config.grid.lx,
            ly=config.grid.ly,
            lz=config.grid.lz,
            device=config.get_device(),
            dtype=config.get_dtype(),
        )

    # ---- Forward / inverse FFT wrappers ----

    def fft(self, f: torch.Tensor) -> torch.Tensor:
        """Real-to-complex 3D FFT."""
        return torch.fft.rfftn(f, dim=(-3, -2, -1))

    def ifft(self, f_hat: torch.Tensor) -> torch.Tensor:
        """Complex-to-real 3D inverse FFT."""
        return torch.fft.irfftn(f_hat, s=(self.nx, self.ny, self.nz), dim=(-3, -2, -1))

    # ---- Spectral derivatives ----

    def ddx(self, f: torch.Tensor) -> torch.Tensor:
        """Partial derivative df/dx via spectral method."""
        f_hat = self.fft(f)
        return self.ifft(1j * self.kx * f_hat)

    def ddy(self, f: torch.Tensor) -> torch.Tensor:
        """Partial derivative df/dy via spectral method."""
        f_hat = self.fft(f)
        return self.ifft(1j * self.ky * f_hat)

    def ddz(self, f: torch.Tensor) -> torch.Tensor:
        """Partial derivative df/dz via spectral method."""
        f_hat = self.fft(f)
        return self.ifft(1j * self.kz * f_hat)

    def gradient(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gradient (df/dx, df/dy, df/dz) with a single FFT."""
        f_hat = self.fft(f)
        dfdx = self.ifft(1j * self.kx * f_hat)
        dfdy = self.ifft(1j * self.ky * f_hat)
        dfdz = self.ifft(1j * self.kz * f_hat)
        return dfdx, dfdy, dfdz

    def divergence(
        self, fx: torch.Tensor, fy: torch.Tensor, fz: torch.Tensor
    ) -> torch.Tensor:
        """Compute divergence div(F) = dFx/dx + dFy/dy + dFz/dz."""
        return self.ddx(fx) + self.ddy(fy) + self.ddz(fz)

    def curl(
        self, fx: torch.Tensor, fy: torch.Tensor, fz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute curl of vector field (Fx, Fy, Fz)."""
        # Pre-compute all needed hats
        fx_hat = self.fft(fx)
        fy_hat = self.fft(fy)
        fz_hat = self.fft(fz)

        # curl_x = dFz/dy - dFy/dz
        cx = self.ifft(1j * self.ky * fz_hat - 1j * self.kz * fy_hat)
        # curl_y = dFx/dz - dFz/dx
        cy = self.ifft(1j * self.kz * fx_hat - 1j * self.kx * fz_hat)
        # curl_z = dFy/dx - dFx/dy
        cz = self.ifft(1j * self.kx * fy_hat - 1j * self.ky * fx_hat)
        return cx, cy, cz

    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """Laplacian via -|k|^2 multiplication in spectral space."""
        f_hat = self.fft(f)
        return self.ifft(-self.k_sq * f_hat)

    # ---- Filtering and dealiasing ----

    def dealias(self, f: torch.Tensor) -> torch.Tensor:
        """Apply 2/3-rule dealiasing filter."""
        f_hat = self.fft(f)
        return self.ifft(f_hat * self.dealias_mask)

    def exponential_filter(
        self, f: torch.Tensor, order: int = 16, cutoff: float = 0.65
    ) -> torch.Tensor:
        """Apply exponential spectral filter for shock capturing.

        sigma(eta) = exp(-alpha * ((eta - cutoff)/(1 - cutoff))^order)  for eta > cutoff
        sigma(eta) = 1  for eta <= cutoff

        where eta = |k| / k_max.
        """
        eta = self.k_mag / max(self.k_max, 1)
        alpha = -math.log(1e-16)

        sigma = torch.ones_like(eta)
        mask = eta > cutoff
        ratio = (eta[mask] - cutoff) / (1.0 - cutoff + 1e-30)
        sigma[mask] = torch.exp(-alpha * ratio**order)

        f_hat = self.fft(f)
        return self.ifft(f_hat * sigma)

    def energy_spectrum(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> tuple:
        """Compute 3D kinetic energy spectrum E(k).

        Returns (k_bins, E_k) where E_k[i] is the energy in shell [k_i, k_{i+1}).
        """
        u_hat = self.fft(u)
        v_hat = self.fft(v)
        w_hat = self.fft(w)

        # Energy density in spectral space (factor of 0.5 for KE)
        n_total = self.nx * self.ny * self.nz
        e_hat = 0.5 * (torch.abs(u_hat)**2 + torch.abs(v_hat)**2 + torch.abs(w_hat)**2) / n_total**2

        # Account for rfft: modes except k_z=0 and k_z=N/2 appear twice
        weight = 2.0 * torch.ones_like(e_hat)
        weight[:, :, 0] = 1.0
        if self.nz % 2 == 0:
            weight[:, :, -1] = 1.0
        e_hat = e_hat * weight

        # Bin by |k|
        k_int = torch.round(self.k_mag).long()
        k_max_bin = int(k_int.max().item()) + 1
        E_k = torch.zeros(k_max_bin, device=self.device, dtype=self.dtype)
        E_k.scatter_add_(0, k_int.reshape(-1), e_hat.reshape(-1))

        k_bins = torch.arange(k_max_bin, device=self.device, dtype=self.dtype)
        return k_bins, E_k

    def strain_rate_tensor(
        self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Compute strain rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/dx_i).

        Returns (S11, S12, S13, S22, S23, S33) — the 6 independent components.
        """
        u_hat = self.fft(u)
        v_hat = self.fft(v)
        w_hat = self.fft(w)

        dudx = self.ifft(1j * self.kx * u_hat)
        dudy = self.ifft(1j * self.ky * u_hat)
        dudz = self.ifft(1j * self.kz * u_hat)
        dvdx = self.ifft(1j * self.kx * v_hat)
        dvdy = self.ifft(1j * self.ky * v_hat)
        dvdz = self.ifft(1j * self.kz * v_hat)
        dwdx = self.ifft(1j * self.kx * w_hat)
        dwdy = self.ifft(1j * self.ky * w_hat)
        dwdz = self.ifft(1j * self.kz * w_hat)

        S11 = dudx
        S22 = dvdy
        S33 = dwdz
        S12 = 0.5 * (dudy + dvdx)
        S13 = 0.5 * (dudz + dwdx)
        S23 = 0.5 * (dvdz + dwdy)

        return S11, S12, S13, S22, S23, S33
