"""
Explicit time integrators for the compressible Navier-Stokes system.

Provides:
    - Classical 4th-order Runge-Kutta (RK4)
    - Strong Stability Preserving RK3 (SSPRK3) — preferred for shock-containing flows

All integrators operate on ConservativeFields and call a generic RHS function.
"""

from __future__ import annotations

from typing import Callable

from .fields import ConservativeFields

RHSFunction = Callable[[ConservativeFields], ConservativeFields]


def rk4_step(
    U: ConservativeFields,
    rhs_fn: RHSFunction,
    dt: float,
) -> ConservativeFields:
    """Classical 4th-order Runge-Kutta time step.

    k1 = dt * f(U)
    k2 = dt * f(U + k1/2)
    k3 = dt * f(U + k2/2)
    k4 = dt * f(U + k3)
    U_new = U + (k1 + 2*k2 + 2*k3 + k4) / 6
    """
    k1 = dt * rhs_fn(U)
    k2 = dt * rhs_fn(U + 0.5 * k1)
    k3 = dt * rhs_fn(U + 0.5 * k2)
    k4 = dt * rhs_fn(U + k3)

    return U + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def ssprk3_step(
    U: ConservativeFields,
    rhs_fn: RHSFunction,
    dt: float,
) -> ConservativeFields:
    """Strong Stability Preserving 3rd-order Runge-Kutta (Shu-Osher).

    Optimal SSP property makes this the standard choice for flows
    with shocks or steep gradients.

    Stage 1:  U^(1) = U^n + dt * f(U^n)
    Stage 2:  U^(2) = 3/4 * U^n + 1/4 * (U^(1) + dt * f(U^(1)))
    Stage 3:  U^{n+1} = 1/3 * U^n + 2/3 * (U^(2) + dt * f(U^(2)))
    """
    # Stage 1
    U1 = U + dt * rhs_fn(U)

    # Stage 2
    U2 = 0.75 * U + 0.25 * (U1 + dt * rhs_fn(U1))

    # Stage 3
    U_new = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * rhs_fn(U2))

    return U_new


def get_integrator(name: str) -> Callable:
    """Get time integrator by name."""
    integrators = {
        "rk4": rk4_step,
        "ssprk3": ssprk3_step,
    }
    if name not in integrators:
        raise ValueError(f"Unknown integrator '{name}'. Choose from: {list(integrators.keys())}")
    return integrators[name]
