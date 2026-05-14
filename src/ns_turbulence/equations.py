"""
Compressible Navier-Stokes equations in conservative form.

Computes the RHS dU/dt for the conservative variable vector
U = (rho, rho*u, rho*v, rho*w, E) using spectral derivatives.

Euler (inviscid) fluxes:
    F_x = (rho*u, rho*u^2+p, rho*u*v, rho*u*w, (E+p)*u)
    F_y = (rho*v, rho*u*v, rho*v^2+p, rho*v*w, (E+p)*v)
    F_z = (rho*w, rho*u*w, rho*v*w, rho*w^2+p, (E+p)*w)

Viscous stresses:
    tau_ij = mu * (du_i/dx_j + du_j/dx_i - 2/3 delta_ij div(u))
    Heat flux: q_i = -kappa * dT/dx_i  (kappa = mu * cp / Pr)
"""

from __future__ import annotations

import torch

from .config import SimConfig
from .fields import ConservativeFields, PrimitiveFields, conservative_to_primitive
from .spectral import SpectralOperator


class CompressibleNS:
    """Compressible Navier-Stokes RHS evaluator.

    Computes dU/dt = -div(F_inviscid) + div(F_viscous) using
    pseudo-spectral differentiation on a periodic domain.
    """

    def __init__(self, config: SimConfig, spec: SpectralOperator) -> None:
        self.config = config
        self.spec = spec
        self.gamma = config.physics.gamma
        self.Re = config.physics.reynolds
        self.Pr = config.physics.prandtl
        self.Ma = config.physics.mach

        # Non-dimensional reference viscosity: mu_ref = 1/Re
        self.mu_ref = 1.0 / self.Re
        # Thermal conductivity: kappa = mu * gamma / ((gamma-1) * Pr * Re * Ma^2)
        # In non-dim form with our scaling: kappa = mu * gamma / ((gamma-1) * Pr)
        self.kappa_factor = self.gamma / ((self.gamma - 1.0) * self.Pr)

    def compute_viscosity(self, T: torch.Tensor) -> torch.Tensor:
        """Compute dynamic viscosity field.

        Supports constant, Sutherland, or Blottner models.
        """
        model = self.config.physics.viscosity_model
        if model == "constant":
            return torch.full_like(T, self.mu_ref)
        elif model == "sutherland":
            T_ref = 1.0  # non-dimensional reference
            S_const = 110.4 / self.config.physics.reference_temperature
            mu = self.mu_ref * (T / T_ref) ** 1.5 * (T_ref + S_const) / (T + S_const)
            return mu
        elif model == "blottner":
            # Blottner curve fit: ln(mu) = (A ln(T) + B) ln(T) + C
            # Using N2 coefficients in non-dimensional form
            T_dim = T * self.config.physics.reference_temperature
            A, B, C = 0.0268142, 0.3177838, -11.3155513
            ln_T = torch.log(torch.clamp(T_dim, min=100.0))
            ln_mu = (A * ln_T + B) * ln_T + C
            mu_dim = torch.exp(ln_mu)
            # Non-dimensionalize
            T_ref = self.config.physics.reference_temperature
            mu_ref_dim = float(torch.exp(torch.tensor(
                (A * torch.log(torch.tensor(T_ref)) + B) * torch.log(torch.tensor(T_ref)) + C
            )))
            return self.mu_ref * mu_dim / mu_ref_dim
        else:
            raise ValueError(f"Unknown viscosity model: {model}")

    def inviscid_rhs(
        self, cons: ConservativeFields, prim: PrimitiveFields
    ) -> ConservativeFields:
        """Compute inviscid (Euler) contribution: -div(F).

        Uses the skew-symmetric form for improved stability:
        d(rho*u_i*u_j)/dx_j ≈ 0.5*(d(rho*u_i*u_j)/dx_j + u_j*d(rho*u_i)/dx_j
                                    + rho*u_i*du_j/dx_j)
        """
        rho, u, v, w, p = prim.rho, prim.u, prim.v, prim.w, prim.p
        E = cons.E
        spec = self.spec

        # --- Continuity: d(rho)/dt = -div(rho * u) ---
        drho_dt = -(
            spec.ddx(rho * u) + spec.ddy(rho * v) + spec.ddz(rho * w)
        )

        # --- Momentum: d(rho*u_i)/dt = -d(rho*u_i*u_j)/dx_j - dp/dx_i ---
        # Convective terms in divergence form
        drhou_dt = -(
            spec.ddx(cons.rhou * u + p)
            + spec.ddy(cons.rhou * v)
            + spec.ddz(cons.rhou * w)
        )
        drhov_dt = -(
            spec.ddx(cons.rhov * u)
            + spec.ddy(cons.rhov * v + p)
            + spec.ddz(cons.rhov * w)
        )
        drhow_dt = -(
            spec.ddx(cons.rhow * u)
            + spec.ddy(cons.rhow * v)
            + spec.ddz(cons.rhow * w + p)
        )

        # --- Energy: d(E)/dt = -div((E + p) * u) ---
        Ep = E + p
        dE_dt = -(
            spec.ddx(Ep * u) + spec.ddy(Ep * v) + spec.ddz(Ep * w)
        )

        return ConservativeFields(
            rho=drho_dt,
            rhou=drhou_dt,
            rhov=drhov_dt,
            rhow=drhow_dt,
            E=dE_dt,
        )

    def viscous_rhs(
        self, prim: PrimitiveFields
    ) -> ConservativeFields:
        """Compute viscous contribution: div(tau) and heat conduction.

        tau_ij = mu * (du_i/dx_j + du_j/dx_i - 2/3 delta_ij * div(u))
        q_i = -kappa * dT/dx_i
        """
        u, v, w, T = prim.u, prim.v, prim.w, prim.T
        spec = self.spec

        mu = self.compute_viscosity(T)
        kappa = mu * self.kappa_factor

        # Velocity gradients (9 components)
        dudx, dudy, dudz = spec.gradient(u)
        dvdx, dvdy, dvdz = spec.gradient(v)
        dwdx, dwdy, dwdz = spec.gradient(w)

        # Divergence of velocity
        div_u = dudx + dvdy + dwdz

        # Stress tensor (Stokes hypothesis: bulk viscosity = 0)
        two_thirds = 2.0 / 3.0
        tau11 = mu * (2.0 * dudx - two_thirds * div_u)
        tau22 = mu * (2.0 * dvdy - two_thirds * div_u)
        tau33 = mu * (2.0 * dwdz - two_thirds * div_u)
        tau12 = mu * (dudy + dvdx)
        tau13 = mu * (dudz + dwdx)
        tau23 = mu * (dvdz + dwdy)

        # Temperature gradients
        dTdx, dTdy, dTdz = spec.gradient(T)

        # Viscous contribution to momentum
        drhou_dt = spec.ddx(tau11) + spec.ddy(tau12) + spec.ddz(tau13)
        drhov_dt = spec.ddx(tau12) + spec.ddy(tau22) + spec.ddz(tau23)
        drhow_dt = spec.ddx(tau13) + spec.ddy(tau23) + spec.ddz(tau33)

        # Viscous contribution to energy
        # d/dx_j(u_i * tau_ij) - d/dx_j(q_j)
        dE_dt = (
            spec.ddx(u * tau11 + v * tau12 + w * tau13 + kappa * dTdx)
            + spec.ddy(u * tau12 + v * tau22 + w * tau23 + kappa * dTdy)
            + spec.ddz(u * tau13 + v * tau23 + w * tau33 + kappa * dTdz)
        )

        # No viscous contribution to continuity
        zero = torch.zeros_like(u)

        return ConservativeFields(
            rho=zero,
            rhou=drhou_dt,
            rhov=drhov_dt,
            rhow=drhow_dt,
            E=dE_dt,
        )

    def rhs(self, cons: ConservativeFields) -> ConservativeFields:
        """Full RHS: dU/dt = inviscid + viscous.

        This is the main entry point called by the time integrator.
        """
        prim = conservative_to_primitive(cons, self.gamma)

        inv = self.inviscid_rhs(cons, prim)
        vis = self.viscous_rhs(prim)

        return inv + vis

    def compute_dt(self, prim: PrimitiveFields) -> float:
        """Compute CFL-limited time step.

        dt = CFL * min(dx, dy, dz) / max(|u| + c)
        """
        c = prim.speed_of_sound(self.gamma)
        speed = torch.sqrt(prim.u**2 + prim.v**2 + prim.w**2)
        max_wave_speed = float((speed + c).max())

        dx = min(self.config.grid.dx, self.config.grid.dy, self.config.grid.dz)
        dt = self.config.time.cfl * dx / max(max_wave_speed, 1e-30)

        # Also limit by viscous CFL: dt < 0.5 * dx^2 * Re
        dt_visc = 0.5 * dx**2 * self.Re
        return min(dt, dt_visc)
