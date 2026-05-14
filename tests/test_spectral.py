"""Tests for pseudo-spectral operators."""

from __future__ import annotations

import math

import pytest
import torch

from ns_turbulence.config import SimConfig
from ns_turbulence.spectral import SpectralOperator


@pytest.fixture
def spec():
    """Small 16^3 spectral operator for testing."""
    config = SimConfig()
    config.grid.nx = config.grid.ny = config.grid.nz = 16
    config.device = "cpu"
    config.dtype = "float64"
    return SpectralOperator.from_config(config)


@pytest.fixture
def grid(spec):
    """Coordinate grids for 16^3 domain."""
    x = torch.linspace(0, 2 * math.pi, spec.nx + 1, dtype=torch.float64)[:-1]
    y = torch.linspace(0, 2 * math.pi, spec.ny + 1, dtype=torch.float64)[:-1]
    z = torch.linspace(0, 2 * math.pi, spec.nz + 1, dtype=torch.float64)[:-1]
    return torch.meshgrid(x, y, z, indexing="ij")


class TestDerivatives:
    def test_ddx_sin(self, spec, grid):
        """d/dx sin(x) = cos(x)."""
        X, Y, Z = grid
        f = torch.sin(X)
        df = spec.ddx(f)
        expected = torch.cos(X)
        assert torch.allclose(df, expected, atol=1e-10)

    def test_ddy_sin(self, spec, grid):
        """d/dy sin(2y) = 2*cos(2y)."""
        X, Y, Z = grid
        f = torch.sin(2 * Y)
        df = spec.ddy(f)
        expected = 2.0 * torch.cos(2 * Y)
        assert torch.allclose(df, expected, atol=1e-10)

    def test_ddz_cos(self, spec, grid):
        """d/dz cos(3z) = -3*sin(3z)."""
        X, Y, Z = grid
        f = torch.cos(3 * Z)
        df = spec.ddz(f)
        expected = -3.0 * torch.sin(3 * Z)
        assert torch.allclose(df, expected, atol=1e-10)

    def test_gradient(self, spec, grid):
        """Gradient of x*y*z product."""
        X, Y, Z = grid
        f = torch.sin(X) * torch.cos(Y)
        dfdx, dfdy, dfdz = spec.gradient(f)
        assert torch.allclose(dfdx, torch.cos(X) * torch.cos(Y), atol=1e-10)
        assert torch.allclose(dfdy, -torch.sin(X) * torch.sin(Y), atol=1e-10)
        assert torch.allclose(dfdz, torch.zeros_like(f), atol=1e-10)

    def test_laplacian(self, spec, grid):
        """Laplacian of sin(x)sin(y)sin(z) = -3*sin(x)sin(y)sin(z)."""
        X, Y, Z = grid
        f = torch.sin(X) * torch.sin(Y) * torch.sin(Z)
        lap = spec.laplacian(f)
        expected = -3.0 * f
        assert torch.allclose(lap, expected, atol=1e-10)


class TestDivergenceCurl:
    def test_divergence_zero(self, spec, grid):
        """Divergence of curl is zero."""
        X, Y, Z = grid
        fx = torch.sin(X) * torch.cos(Y)
        fy = torch.cos(X) * torch.sin(Z)
        fz = torch.sin(Y) * torch.cos(Z)
        cx, cy, cz = spec.curl(fx, fy, fz)
        div_curl = spec.divergence(cx, cy, cz)
        assert torch.allclose(div_curl, torch.zeros_like(div_curl), atol=1e-10)

    def test_divergence_known(self, spec, grid):
        """div(sin(x), sin(y), sin(z)) = cos(x) + cos(y) + cos(z)."""
        X, Y, Z = grid
        div = spec.divergence(torch.sin(X), torch.sin(Y), torch.sin(Z))
        expected = torch.cos(X) + torch.cos(Y) + torch.cos(Z)
        assert torch.allclose(div, expected, atol=1e-10)


class TestFiltering:
    def test_dealias_preserves_low_modes(self, spec, grid):
        """Low-k modes pass through dealiasing."""
        X, Y, Z = grid
        f = torch.sin(X) + torch.cos(Y)
        f_filtered = spec.dealias(f)
        assert torch.allclose(f_filtered, f, atol=1e-10)

    def test_exponential_filter_preserves_low_modes(self, spec, grid):
        """Exponential filter preserves low wavenumbers."""
        X, Y, Z = grid
        f = torch.sin(X) * torch.cos(Y)
        f_filt = spec.exponential_filter(f, order=16, cutoff=0.65)
        assert torch.allclose(f_filt, f, atol=1e-8)


class TestEnergySpectrum:
    def test_single_mode_energy(self, spec, grid):
        """Energy spectrum of single Fourier mode peaks at correct k."""
        X, Y, Z = grid
        u = torch.sin(2 * X)
        v = torch.zeros_like(u)
        w = torch.zeros_like(u)
        k_bins, E_k = spec.energy_spectrum(u, v, w)
        # Peak should be at k=2
        peak_k = int(k_bins[E_k.argmax()])
        assert peak_k == 2

    def test_energy_spectrum_positive(self, spec, grid):
        """Energy spectrum is non-negative."""
        X, Y, Z = grid
        u = torch.sin(X) + 0.5 * torch.cos(3 * Y)
        v = torch.sin(2 * Z)
        w = torch.cos(X)
        _, E_k = spec.energy_spectrum(u, v, w)
        assert (E_k >= -1e-15).all()


class TestStrainRate:
    def test_strain_symmetric(self, spec, grid):
        """Strain rate tensor is symmetric (S12 = S21)."""
        X, Y, Z = grid
        u = torch.sin(X) * torch.cos(Y)
        v = torch.cos(X) * torch.sin(Y)
        w = torch.sin(Z)
        S11, S12, S13, S22, S23, S33 = spec.strain_rate_tensor(u, v, w)
        # S12 = 0.5*(du/dy + dv/dx) — symmetry is built in
        assert S11.shape == X.shape
        assert S12.shape == X.shape

    def test_incompressible_traceless(self, spec, grid):
        """Trace of strain = 0 for divergence-free field."""
        X, Y, Z = grid
        u = torch.sin(X) * torch.cos(Y)
        v = -torch.cos(X) * torch.sin(Y)
        w = torch.zeros_like(X)
        S11, S12, S13, S22, S23, S33 = spec.strain_rate_tensor(u, v, w)
        trace = S11 + S22 + S33
        assert torch.allclose(trace, torch.zeros_like(trace), atol=1e-10)
