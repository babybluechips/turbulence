"""Tests for shock-capturing methods."""

from __future__ import annotations

import pytest
import torch

from ns_turbulence.config import SimConfig
from ns_turbulence.fields import (
    PrimitiveFields,
    conservative_to_primitive,
    create_grid,
    create_uniform_field,
)
from ns_turbulence.shock import ShockCapture
from ns_turbulence.spectral import SpectralOperator


@pytest.fixture
def setup():
    config = SimConfig()
    config.grid.nx = config.grid.ny = config.grid.nz = 16
    config.device = "cpu"
    config.dtype = "float64"
    spec = SpectralOperator.from_config(config)
    shock = ShockCapture(config.shock, spec, config.physics.gamma)
    X, Y, Z = create_grid(config)
    return config, spec, shock, X, Y, Z


class TestSpectralFilter:
    def test_preserves_uniform(self, setup):
        """Filter preserves uniform fields."""
        config, spec, shock, X, Y, Z = setup
        U = create_uniform_field(config)
        U_filt = shock.apply_spectral_filter(U)
        for a, b in zip(U.as_tuple(), U_filt.as_tuple()):
            assert torch.allclose(a, b, atol=1e-10)

    def test_preserves_low_modes(self, setup):
        """Low-k modes survive filtering."""
        config, spec, shock, X, Y, Z = setup
        U = create_uniform_field(config)
        U.rho = U.rho + 0.1 * torch.sin(X)
        U_filt = shock.apply_spectral_filter(U)
        assert torch.allclose(U_filt.rho, U.rho, atol=1e-6)


class TestDucrosSensor:
    def test_sensor_range(self, setup):
        """Ducros sensor in [0, 1]."""
        config, spec, shock, X, Y, Z = setup
        rho = torch.ones_like(X)
        u = torch.sin(X)
        v = torch.sin(Y)
        w = torch.sin(Z)
        p = torch.ones_like(X)
        T = torch.ones_like(X)
        prim = PrimitiveFields(rho=rho, u=u, v=v, w=w, p=p, T=T)
        phi = shock.ducros_sensor(prim)
        assert (phi >= -1e-10).all()
        assert (phi <= 1.0 + 1e-10).all()

    def test_sensor_low_for_solenoidal(self, setup):
        """Ducros sensor has low mean for solenoidal flow."""
        config, spec, shock, X, Y, Z = setup
        u = torch.sin(X) * torch.cos(Y)
        v = -torch.cos(X) * torch.sin(Y)
        w = torch.zeros_like(X)
        rho = torch.ones_like(X)
        p = torch.ones_like(X)
        T = torch.ones_like(X)
        prim = PrimitiveFields(rho=rho, u=u, v=v, w=w, p=p, T=T)
        phi = shock.ducros_sensor(prim)
        # Mean should be low even if some points have high values
        # (at stagnation points, both div and curl vanish)
        assert float(phi.mean()) < 0.5


class TestArtificialViscosity:
    def test_av_zero_for_uniform(self, setup):
        """AV contribution is zero for uniform flow."""
        config, spec, shock, X, Y, Z = setup
        U = create_uniform_field(config)
        prim = conservative_to_primitive(U, config.physics.gamma)
        av = shock.artificial_viscosity(U, prim)
        for field in av.as_tuple():
            assert torch.allclose(field, torch.zeros_like(field), atol=1e-10)
