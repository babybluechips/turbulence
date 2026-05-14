"""Tests for thermochemistry models."""

from __future__ import annotations

import pytest
import torch

from ns_turbulence.thermo import (
    blottner_viscosity,
    dilatation_fraction,
    gamma_effective,
    post_shock_density_ratio,
    post_shock_pressure_ratio,
    post_shock_temperature,
    sutherland_viscosity,
)


class TestGammaEffective:
    def test_low_temperature(self):
        """gamma ≈ 1.4 at low T."""
        T = torch.tensor(1.0, dtype=torch.float64)
        g = gamma_effective(T, T_ref=300.0)
        assert float(g) == pytest.approx(1.4, abs=0.01)

    def test_decreasing_with_T(self):
        """gamma decreases with temperature."""
        T_vals = torch.tensor([1.0, 5.0, 10.0, 20.0], dtype=torch.float64)
        g_vals = gamma_effective(T_vals, T_ref=300.0)
        for i in range(len(g_vals) - 1):
            assert g_vals[i] >= g_vals[i + 1] - 0.01

    def test_bounded(self):
        """gamma stays in physical range [1.1, 1.4]."""
        T = torch.linspace(0.5, 50.0, 100, dtype=torch.float64)
        g = gamma_effective(T, T_ref=300.0)
        assert (g >= 1.1).all()
        assert (g <= 1.41).all()


class TestViscosity:
    def test_sutherland_reference(self):
        """Sutherland = 1 at reference temperature."""
        T = torch.tensor(1.0, dtype=torch.float64)
        mu = sutherland_viscosity(T, T_ref=300.0)
        assert float(mu) == pytest.approx(1.0, abs=0.01)

    def test_sutherland_increases(self):
        """Viscosity increases with temperature."""
        T = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
        mu = sutherland_viscosity(T, T_ref=300.0)
        for i in range(len(mu) - 1):
            assert mu[i] < mu[i + 1]

    def test_blottner_positive(self):
        T = torch.tensor([1.0, 5.0, 10.0], dtype=torch.float64)
        mu = blottner_viscosity(T, "N2", T_ref=300.0)
        assert (mu > 0).all()

    def test_blottner_species(self):
        T = torch.tensor(5.0, dtype=torch.float64)
        for species in ["N2", "O2", "N", "O", "NO", "air"]:
            mu = blottner_viscosity(T, species, T_ref=300.0)
            assert float(mu) > 0


class TestShockRelations:
    def test_normal_shock_Ma1(self):
        """At Ma=1, shock relations give ratio=1."""
        assert post_shock_density_ratio(1.0) == pytest.approx(1.0, abs=1e-10)
        assert post_shock_pressure_ratio(1.0) == pytest.approx(1.0, abs=1e-10)
        assert post_shock_temperature(1.0) == pytest.approx(1.0, abs=1e-10)

    def test_strong_shock(self):
        """Strong shock: rho ratio → (gamma+1)/(gamma-1) = 6 for gamma=1.4."""
        rho_r = post_shock_density_ratio(100.0, gamma=1.4)
        assert rho_r == pytest.approx(6.0, abs=0.1)

    def test_pressure_increases(self):
        """Post-shock pressure > pre-shock pressure."""
        for Ma in [1.5, 2.0, 5.0, 10.0]:
            assert post_shock_pressure_ratio(Ma) > 1.0

    def test_temperature_increases(self):
        """Post-shock temperature > pre-shock temperature."""
        for Ma in [1.5, 2.0, 5.0, 10.0]:
            assert post_shock_temperature(Ma) > 1.0


class TestDilatation:
    def test_zero_mach(self):
        M_t = torch.tensor(0.0, dtype=torch.float64)
        chi = dilatation_fraction(M_t)
        assert float(chi) == pytest.approx(0.0, abs=1e-10)

    def test_bounded(self):
        M_t = torch.linspace(0, 10, 50, dtype=torch.float64)
        chi = dilatation_fraction(M_t)
        assert (chi >= -1e-10).all()
        assert (chi <= 1.0 + 1e-10).all()

    def test_monotone(self):
        M_t = torch.linspace(0, 10, 50, dtype=torch.float64)
        chi = dilatation_fraction(M_t)
        assert (chi[1:] >= chi[:-1] - 1e-10).all()
