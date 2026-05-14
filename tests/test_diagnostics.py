"""Tests for RCL diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from ns_turbulence.config import SimConfig
from ns_turbulence.diagnostics import (
    ALPHA_INCOMP,
    INCOMP_MARGIN,
    T_CRITICAL,
    cfm_from_vorticity,
    compute_statistics,
    depletion_factor_field,
    f_bound,
    q_criterion,
)
from ns_turbulence.fields import create_grid
from ns_turbulence.spectral import SpectralOperator


@pytest.fixture
def setup():
    config = SimConfig()
    config.grid.nx = config.grid.ny = config.grid.nz = 16
    config.device = "cpu"
    config.dtype = "float64"
    spec = SpectralOperator.from_config(config)
    X, Y, Z = create_grid(config)
    return config, spec, X, Y, Z


class TestConstants:
    def test_alpha_incomp(self):
        assert ALPHA_INCOMP == pytest.approx(math.sqrt(2.0 / 3.0), abs=1e-10)

    def test_t_critical(self):
        assert T_CRITICAL == pytest.approx(math.sqrt(2.0), abs=1e-10)

    def test_margin(self):
        assert INCOMP_MARGIN == pytest.approx(1.0 - math.sqrt(2.0 / 3.0), abs=1e-10)


class TestFBound:
    def test_f_at_zero(self):
        t = torch.tensor(0.0, dtype=torch.float64)
        assert float(f_bound(t)) == pytest.approx(1.0 / math.sqrt(3), abs=1e-10)

    def test_f_at_critical(self):
        t = torch.tensor(math.sqrt(2.0), dtype=torch.float64)
        assert float(f_bound(t)) == pytest.approx(1.0, abs=1e-10)

    def test_f_at_infinity(self):
        t = torch.tensor(1000.0, dtype=torch.float64)
        assert float(f_bound(t)) == pytest.approx(ALPHA_INCOMP, abs=0.01)

    def test_f_monotone_after_critical(self):
        t = torch.linspace(T_CRITICAL, 100.0, 100, dtype=torch.float64)
        f = f_bound(t)
        # f should be monotonically decreasing after t=sqrt(2)
        assert (f[1:] <= f[:-1] + 1e-10).all()


class TestDepletionFactor:
    def test_incompressible_mean_bounded(self, setup):
        """Mean depletion factor of divergence-free field is sub-critical."""
        config, spec, X, Y, Z = setup
        # 3D solenoidal field for a proper test
        u = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
        v = -torch.cos(X) * torch.sin(Y) * torch.cos(Z)
        w = torch.zeros_like(X)  # div-free since d(sin*cos)/dx + d(-cos*sin)/dy = 0
        alpha = depletion_factor_field(spec, u, v, w)
        # Mean alpha should be well below 1
        assert float(alpha.mean()) < 1.0

    def test_alpha_nonnegative(self, setup):
        config, spec, X, Y, Z = setup
        u = torch.sin(X) * torch.cos(Y)
        v = -torch.cos(X) * torch.sin(Y)
        w = torch.zeros_like(X)
        alpha = depletion_factor_field(spec, u, v, w)
        assert (alpha >= -1e-10).all()


class TestCFM:
    def test_cfm_range(self, setup):
        """CFM is in [0, 1]."""
        config, spec, X, Y, Z = setup
        u = torch.sin(X) * torch.cos(Y)
        v = -torch.cos(X) * torch.sin(Y)
        w = torch.sin(Z)
        cfm = cfm_from_vorticity(spec, u, v, w)
        assert 0.0 <= cfm <= 1.0

    def test_cfm_high_for_isotropic(self, setup):
        """CFM is high for multi-directional vorticity (near-isotropic)."""
        config, spec, X, Y, Z = setup
        # Three-component vorticity → high CFM (incoherent)
        u = torch.sin(Y) * torch.cos(Z)
        v = torch.sin(Z) * torch.cos(X)
        w = torch.sin(X) * torch.cos(Y)
        cfm = cfm_from_vorticity(spec, u, v, w)
        assert cfm > 0.5  # multi-directional → high CFM


class TestQCriterion:
    def test_q_shape(self, setup):
        config, spec, X, Y, Z = setup
        u = torch.sin(X) * torch.cos(Y)
        v = -torch.cos(X) * torch.sin(Y)
        w = torch.zeros_like(X)
        Q = q_criterion(spec, u, v, w)
        assert Q.shape == X.shape

    def test_q_finite(self, setup):
        config, spec, X, Y, Z = setup
        u = torch.sin(X)
        v = torch.sin(Y)
        w = torch.sin(Z)
        Q = q_criterion(spec, u, v, w)
        assert torch.isfinite(Q).all()


class TestStatistics:
    def test_compute_statistics(self, setup):
        config, spec, X, Y, Z = setup
        from ns_turbulence.initial_conditions import taylor_green_vortex
        U = taylor_green_vortex(X, Y, Z, config)
        stats = compute_statistics(U, spec, config.physics.gamma, config.physics.reynolds, 0.0)
        assert stats.kinetic_energy > 0
        assert stats.enstrophy >= 0
        assert 0 <= stats.depletion_alpha <= 1.5
        assert stats.min_density > 0
