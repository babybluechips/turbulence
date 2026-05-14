"""Tests for time integration schemes."""

from __future__ import annotations

import pytest
import torch

from ns_turbulence.config import SimConfig
from ns_turbulence.fields import ConservativeFields, create_uniform_field
from ns_turbulence.timestepping import get_integrator, rk4_step, ssprk3_step


@pytest.fixture
def config():
    cfg = SimConfig()
    cfg.grid.nx = cfg.grid.ny = cfg.grid.nz = 8
    cfg.device = "cpu"
    cfg.dtype = "float64"
    return cfg


def _zero_rhs(U: ConservativeFields) -> ConservativeFields:
    """Zero RHS: dU/dt = 0 → constant solution."""
    return ConservativeFields(
        rho=torch.zeros_like(U.rho),
        rhou=torch.zeros_like(U.rhou),
        rhov=torch.zeros_like(U.rhov),
        rhow=torch.zeros_like(U.rhow),
        E=torch.zeros_like(U.E),
    )


def _linear_rhs(U: ConservativeFields) -> ConservativeFields:
    """Linear RHS: dU/dt = -U → exponential decay."""
    return -1.0 * U


class TestRK4:
    def test_constant_solution(self, config):
        """Zero RHS preserves initial state."""
        U = create_uniform_field(config)
        U_new = rk4_step(U, _zero_rhs, dt=0.1)
        for a, b in zip(U.as_tuple(), U_new.as_tuple()):
            assert torch.allclose(a, b, atol=1e-14)

    def test_convergence_order(self, config):
        """RK4 has 4th-order convergence."""
        U0 = create_uniform_field(config, rho0=1.0)

        dt1 = 0.1
        dt2 = 0.05
        U1 = rk4_step(U0, _linear_rhs, dt1)
        U2 = rk4_step(U0, _linear_rhs, dt2)
        U2 = rk4_step(U2, _linear_rhs, dt2)  # two steps of dt/2

        # Exact: rho(dt) = exp(-dt) * rho(0)
        import math
        exact = math.exp(-dt1)
        err1 = abs(float(U1.rho.mean()) - exact)
        err2 = abs(float(U2.rho.mean()) - exact)

        # Error ratio should be ~16 for 4th order
        if err2 > 1e-15:
            ratio = err1 / err2
            assert ratio > 10  # some tolerance for floating point


class TestSSPRK3:
    def test_constant_solution(self, config):
        U = create_uniform_field(config)
        U_new = ssprk3_step(U, _zero_rhs, dt=0.1)
        for a, b in zip(U.as_tuple(), U_new.as_tuple()):
            assert torch.allclose(a, b, atol=1e-14)

    def test_stability(self, config):
        """SSPRK3 is stable for small enough dt with linear decay."""
        U = create_uniform_field(config, rho0=1.0)
        for _ in range(100):
            U = ssprk3_step(U, _linear_rhs, dt=0.01)
        # After decay, rho should be smaller
        assert float(U.rho.mean()) < 1.0
        assert float(U.rho.mean()) > 0.0


class TestGetIntegrator:
    def test_valid_names(self):
        assert get_integrator("rk4") is rk4_step
        assert get_integrator("ssprk3") is ssprk3_step

    def test_invalid_name(self):
        with pytest.raises(ValueError):
            get_integrator("euler_forward")
