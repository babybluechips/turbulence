"""Tests for compressible Navier-Stokes equations."""

from __future__ import annotations

import pytest
import torch

from ns_turbulence.config import SimConfig
from ns_turbulence.equations import CompressibleNS
from ns_turbulence.fields import (
    conservative_to_primitive,
    create_grid,
    create_uniform_field,
)
from ns_turbulence.spectral import SpectralOperator


@pytest.fixture
def setup():
    config = SimConfig()
    config.grid.nx = config.grid.ny = config.grid.nz = 16
    config.device = "cpu"
    config.dtype = "float64"
    config.physics.reynolds = 100.0
    config.physics.mach = 0.3
    spec = SpectralOperator.from_config(config)
    ns = CompressibleNS(config, spec)
    X, Y, Z = create_grid(config)
    return config, spec, ns, X, Y, Z


class TestInviscidRHS:
    def test_uniform_rhs_zero(self, setup):
        """Uniform flow has zero RHS."""
        config, spec, ns, X, Y, Z = setup
        U = create_uniform_field(config, rho0=1.0, u0=0.0, v0=0.0, w0=0.0, p0=1.0)
        dUdt = ns.rhs(U)
        for field in dUdt.as_tuple():
            assert torch.allclose(field, torch.zeros_like(field), atol=1e-10)

    def test_rhs_finite(self, setup):
        """RHS of non-trivial field is finite."""
        config, spec, ns, X, Y, Z = setup
        from ns_turbulence.initial_conditions import taylor_green_vortex
        U = taylor_green_vortex(X, Y, Z, config)
        dUdt = ns.rhs(U)
        for field in dUdt.as_tuple():
            assert torch.isfinite(field).all()


class TestViscousRHS:
    def test_viscous_zero_for_uniform(self, setup):
        """Viscous contribution is zero for uniform flow."""
        config, spec, ns, X, Y, Z = setup
        U = create_uniform_field(config)
        prim = conservative_to_primitive(U, config.physics.gamma)
        vis = ns.viscous_rhs(prim)
        for field in vis.as_tuple():
            assert torch.allclose(field, torch.zeros_like(field), atol=1e-10)


class TestViscosityModels:
    def test_constant_viscosity(self, setup):
        config, spec, ns, X, Y, Z = setup
        T = torch.ones(config.grid.shape, dtype=torch.float64)
        mu = ns.compute_viscosity(T)
        expected = 1.0 / config.physics.reynolds
        assert torch.allclose(mu, torch.full_like(mu, expected))

    def test_sutherland_viscosity(self, setup):
        config, spec, ns, X, Y, Z = setup
        config.physics.viscosity_model = "sutherland"
        ns2 = CompressibleNS(config, spec)
        T = torch.ones(config.grid.shape, dtype=torch.float64)
        mu = ns2.compute_viscosity(T)
        assert (mu > 0).all()
        assert torch.isfinite(mu).all()


class TestCFL:
    def test_dt_positive(self, setup):
        config, spec, ns, X, Y, Z = setup
        U = create_uniform_field(config, u0=0.5, p0=1.0)
        prim = conservative_to_primitive(U, config.physics.gamma)
        dt = ns.compute_dt(prim)
        assert dt > 0

    def test_dt_decreases_with_speed(self, setup):
        config, spec, ns, X, Y, Z = setup
        U_slow = create_uniform_field(config, u0=0.1, p0=1.0)
        U_fast = create_uniform_field(config, u0=1.0, p0=1.0)
        prim_slow = conservative_to_primitive(U_slow, config.physics.gamma)
        prim_fast = conservative_to_primitive(U_fast, config.physics.gamma)
        dt_slow = ns.compute_dt(prim_slow)
        dt_fast = ns.compute_dt(prim_fast)
        assert dt_fast < dt_slow
