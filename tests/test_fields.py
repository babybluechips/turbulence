"""Tests for field containers and variable conversion."""

from __future__ import annotations

import math

import pytest
import torch

from ns_turbulence.config import SimConfig
from ns_turbulence.fields import (
    conservative_to_primitive,
    create_grid,
    create_uniform_field,
    primitive_to_conservative,
)


@pytest.fixture
def config():
    cfg = SimConfig()
    cfg.grid.nx = cfg.grid.ny = cfg.grid.nz = 8
    cfg.device = "cpu"
    cfg.dtype = "float64"
    return cfg


class TestConservativeFields:
    def test_clone(self, config):
        U = create_uniform_field(config)
        V = U.clone()
        V.rho.fill_(999.0)
        assert float(U.rho.mean()) != 999.0

    def test_add(self, config):
        U = create_uniform_field(config)
        V = U + U
        assert torch.allclose(V.rho, 2.0 * U.rho)

    def test_scalar_mul(self, config):
        U = create_uniform_field(config)
        V = 3.0 * U
        assert torch.allclose(V.rho, 3.0 * U.rho)

    def test_shape(self, config):
        U = create_uniform_field(config)
        assert U.shape == (8, 8, 8)


class TestVariableConversion:
    def test_roundtrip(self, config):
        """cons -> prim -> cons roundtrip preserves values."""
        gamma = config.physics.gamma
        U = create_uniform_field(config, rho0=1.2, u0=0.5, v0=-0.3, w0=0.1, p0=2.0)
        prim = conservative_to_primitive(U, gamma)
        U2 = primitive_to_conservative(prim, gamma)
        for a, b in zip(U.as_tuple(), U2.as_tuple()):
            assert torch.allclose(a, b, atol=1e-12)

    def test_uniform_pressure(self, config):
        """Uniform field has correct pressure."""
        gamma = config.physics.gamma
        p0 = 2.5
        U = create_uniform_field(config, p0=p0)
        prim = conservative_to_primitive(U, gamma)
        assert torch.allclose(prim.p, torch.full_like(prim.p, p0), atol=1e-12)

    def test_kinetic_energy(self, config):
        """KE = 0.5 * rho * |u|^2."""
        gamma = config.physics.gamma
        U = create_uniform_field(config, rho0=2.0, u0=3.0, v0=4.0, w0=0.0, p0=1.0)
        prim = conservative_to_primitive(U, gamma)
        ke = prim.kinetic_energy()
        expected = 0.5 * 2.0 * (9.0 + 16.0)
        assert torch.allclose(ke, torch.full_like(ke, expected), atol=1e-12)

    def test_speed_of_sound(self, config):
        """c = sqrt(gamma * p / rho)."""
        gamma = config.physics.gamma
        U = create_uniform_field(config, rho0=1.0, p0=1.0)
        prim = conservative_to_primitive(U, gamma)
        c = prim.speed_of_sound(gamma)
        expected = math.sqrt(gamma)
        assert torch.allclose(c, torch.full_like(c, expected), atol=1e-12)


class TestGrid:
    def test_grid_shape(self, config):
        X, Y, Z = create_grid(config)
        assert X.shape == (8, 8, 8)

    def test_grid_range(self, config):
        X, Y, Z = create_grid(config)
        assert float(X.min()) == pytest.approx(0.0)
        assert float(X.max()) < config.grid.lx
