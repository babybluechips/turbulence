"""
Standard initial conditions for compressible turbulence simulations.

Each function has signature:
    f(X, Y, Z, config) -> ConservativeFields

where X, Y, Z are 3D coordinate grids on the configured device.
"""

from __future__ import annotations

import math

import torch

from .config import SimConfig
from .fields import ConservativeFields, PrimitiveFields, primitive_to_conservative


def taylor_green_vortex(
    X: torch.Tensor,
    Y: torch.Tensor,
    Z: torch.Tensor,
    config: SimConfig,
) -> ConservativeFields:
    """Compressible Taylor-Green vortex.

    The canonical 3D benchmark for transition to turbulence.
    Initial condition:
        u = sin(x) cos(y) cos(z)
        v = -cos(x) sin(y) cos(z)
        w = 0
        p = p0 + (1/16)(cos(2x) + cos(2y))(cos(2z) + 2)
        rho = p / T0  (from ideal gas)

    Velocity scaled by Mach number.
    """
    gamma = config.physics.gamma
    Ma = config.physics.mach

    # Velocity field (scaled by Ma for compressible)
    u = Ma * torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    v = -Ma * torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    w = torch.zeros_like(X)

    # Pressure: base + perturbation
    p0 = 1.0 / (gamma * Ma**2)
    p = p0 + (Ma**2 / 16.0) * (
        (torch.cos(2 * X) + torch.cos(2 * Y)) * (torch.cos(2 * Z) + 2.0)
    )

    # Density from ideal gas
    T0 = 1.0
    rho = gamma * Ma**2 * p  # p = rho * T / (gamma * Ma^2) in non-dim

    # Actually in our non-dim: p = rho * T, so rho = p / T0
    rho = p / T0
    T = T0 * torch.ones_like(X)

    prim = PrimitiveFields(rho=rho, u=u, v=v, w=w, p=p, T=T)
    return primitive_to_conservative(prim, gamma)


def compressible_hit(
    X: torch.Tensor,
    Y: torch.Tensor,
    Z: torch.Tensor,
    config: SimConfig,
) -> ConservativeFields:
    """Compressible homogeneous isotropic turbulence (HIT).

    Broadband initial velocity field with prescribed energy spectrum
    E(k) ~ k^4 * exp(-2*(k/k0)^2), plus uniform density and pressure.
    """
    gamma = config.physics.gamma
    Ma = config.physics.mach

    device = X.device
    dtype = X.dtype
    shape = X.shape
    nx, ny, nz = shape

    # Generate random solenoidal velocity field in spectral space
    torch.manual_seed(config.seed)

    # Wavenumbers
    kx = torch.fft.fftfreq(nx, d=1.0 / (2 * math.pi), device=device).to(dtype)
    ky = torch.fft.fftfreq(ny, d=1.0 / (2 * math.pi), device=device).to(dtype)
    kz = torch.fft.rfftfreq(nz, d=1.0 / (2 * math.pi), device=device).to(dtype)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
    K = torch.sqrt(KX**2 + KY**2 + KZ**2)

    # Target spectrum: E(k) ~ k^4 exp(-2(k/k0)^2)
    k0 = 4.0  # peak wavenumber
    E_target = K**4 * torch.exp(-2.0 * (K / k0) ** 2)
    E_target[0, 0, 0] = 0.0  # no mean flow

    # Random phases
    nzh = nz // 2 + 1
    phase_u = 2 * math.pi * torch.rand(nx, ny, nzh, device=device, dtype=dtype)
    phase_v = 2 * math.pi * torch.rand(nx, ny, nzh, device=device, dtype=dtype)
    phase_w = 2 * math.pi * torch.rand(nx, ny, nzh, device=device, dtype=dtype)

    # Amplitude from target spectrum
    amp = torch.sqrt(E_target / (4 * math.pi * K**2 + 1e-30))
    amp[0, 0, 0] = 0.0

    u_hat = amp * torch.exp(1j * phase_u)
    v_hat = amp * torch.exp(1j * phase_v)
    w_hat = amp * torch.exp(1j * phase_w)

    # Leray projection (make solenoidal)
    K_sq = K**2
    K_sq_safe = K_sq.clone()
    K_sq_safe[0, 0, 0] = 1.0
    k_dot_u = KX * u_hat + KY * v_hat + KZ * w_hat
    u_hat -= KX * k_dot_u / K_sq_safe
    v_hat -= KY * k_dot_u / K_sq_safe
    w_hat -= KZ * k_dot_u / K_sq_safe

    u = torch.fft.irfftn(u_hat, s=(nx, ny, nz))
    v = torch.fft.irfftn(v_hat, s=(nx, ny, nz))
    w = torch.fft.irfftn(w_hat, s=(nx, ny, nz))

    # Scale to target Mach number
    u_rms = torch.sqrt((u**2 + v**2 + w**2).mean() / 3.0)
    scale = Ma / (u_rms + 1e-30)
    u *= scale
    v *= scale
    w *= scale

    # Uniform thermodynamic state
    rho = torch.ones(shape, device=device, dtype=dtype)
    T = torch.ones(shape, device=device, dtype=dtype)
    p = rho * T  # p = rho * T in non-dim

    prim = PrimitiveFields(rho=rho, u=u, v=v, w=w, p=p, T=T)
    return primitive_to_conservative(prim, gamma)


def shock_turbulence_interaction(
    X: torch.Tensor,
    Y: torch.Tensor,
    Z: torch.Tensor,
    config: SimConfig,
) -> ConservativeFields:
    """Shock-turbulence interaction initial condition.

    A planar normal shock at x = L/4 with upstream turbulence.
    Uses Rankine-Hugoniot for the post-shock state.
    """
    gamma = config.physics.gamma
    Ma = config.physics.mach

    from .thermo import post_shock_density_ratio, post_shock_pressure_ratio

    # Shock location
    x_shock = config.grid.lx / 4.0

    # Pre-shock state (upstream)
    rho_pre = 1.0
    p_pre = 1.0 / (gamma * Ma**2)

    # Post-shock state
    rho_ratio = post_shock_density_ratio(Ma, gamma)
    p_ratio = post_shock_pressure_ratio(Ma, gamma)
    rho_post = rho_pre * rho_ratio
    p_post = p_pre * p_ratio

    # Velocity: post-shock velocity from mass conservation
    u_pre = Ma  # in non-dim where c = 1
    u_post = u_pre / rho_ratio

    # Smooth transition (tanh profile instead of sharp shock)
    delta = config.grid.dx * 4  # shock thickness
    transition = 0.5 * (1.0 + torch.tanh((X - x_shock) / delta))

    rho = rho_pre + (rho_post - rho_pre) * transition
    u_mean = u_pre + (u_post - u_pre) * transition
    p = p_pre + (p_post - p_pre) * transition
    T = p / rho

    # Add upstream turbulent perturbations
    torch.manual_seed(config.seed + 1)
    u_turb = 0.05 * Ma * torch.randn_like(X) * (1.0 - transition)
    v_turb = 0.05 * Ma * torch.randn_like(X) * (1.0 - transition)
    w_turb = 0.05 * Ma * torch.randn_like(X) * (1.0 - transition)

    u = u_mean + u_turb
    v = v_turb
    w = w_turb

    prim = PrimitiveFields(rho=rho, u=u, v=v, w=w, p=p, T=T)
    return primitive_to_conservative(prim, gamma)
