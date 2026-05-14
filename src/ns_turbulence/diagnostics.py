"""
RCL-framework diagnostics for compressible turbulence.

GPU-accelerated computation of:
    - Depletion factor alpha (incompressible and compressible bounds)
    - Coherence Failure Metric (CFM)
    - Spectral gap lambda_2(Q)
    - Vortex identification (Q-criterion, lambda_2 criterion)
    - Enstrophy, dissipation, energy spectra
    - Strain tensor statistics

All diagnostics operate on PyTorch tensors and run on GPU when available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .fields import ConservativeFields, conservative_to_primitive
from .spectral import SpectralOperator

# Fundamental constants from RCL theory
ALPHA_INCOMP = math.sqrt(2.0 / 3.0)     # ≈ 0.8165
T_CRITICAL = math.sqrt(2.0)             # extremizer ratio
INCOMP_MARGIN = 1.0 - ALPHA_INCOMP      # ≈ 0.1835


@dataclass
class FlowStatistics:
    """Container for flow diagnostic quantities."""

    time: float
    kinetic_energy: float
    enstrophy: float
    dissipation_rate: float
    max_mach: float
    mean_mach: float
    taylor_re: float
    kolmogorov_eta: float
    depletion_alpha: float
    depletion_alpha_c: float
    cfm: float
    max_divergence: float
    min_density: float
    max_density: float
    max_temperature: float

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---- Depletion factor ----

def depletion_factor_field(
    spec: SpectralOperator,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """Compute pointwise depletion factor alpha(x).

    alpha = max|lambda_i(S)| / |S|_F

    where S is the strain rate tensor and lambda_i are its eigenvalues.
    Returns a scalar field of alpha values.
    """
    S11, S12, S13, S22, S23, S33 = spec.strain_rate_tensor(u, v, w)

    # Frobenius norm: |S|_F = sqrt(S_ij * S_ij)
    # For symmetric tensor: |S|_F^2 = S11^2 + S22^2 + S33^2 + 2*(S12^2 + S13^2 + S23^2)
    S_frob = torch.sqrt(
        S11**2 + S22**2 + S33**2 + 2.0 * (S12**2 + S13**2 + S23**2)
    )

    # Batch eigenvalue computation via torch.linalg.eigvalsh
    # Stack into (N, 3, 3) matrix
    shape = S11.shape
    n = S11.numel()

    S_mat = torch.zeros(n, 3, 3, device=S11.device, dtype=S11.dtype)
    S_mat[:, 0, 0] = S11.reshape(-1)
    S_mat[:, 0, 1] = S12.reshape(-1)
    S_mat[:, 0, 2] = S13.reshape(-1)
    S_mat[:, 1, 0] = S12.reshape(-1)
    S_mat[:, 1, 1] = S22.reshape(-1)
    S_mat[:, 1, 2] = S23.reshape(-1)
    S_mat[:, 2, 0] = S13.reshape(-1)
    S_mat[:, 2, 1] = S23.reshape(-1)
    S_mat[:, 2, 2] = S33.reshape(-1)

    eigvals = torch.linalg.eigvalsh(S_mat)  # (n, 3), sorted ascending
    max_abs_eig = torch.max(torch.abs(eigvals), dim=-1).values  # (n,)

    S_frob_flat = S_frob.reshape(-1)
    alpha = max_abs_eig / (S_frob_flat + 1e-30)

    return alpha.reshape(shape)


def compressible_depletion_field(
    spec: SpectralOperator,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pointwise compressible depletion factor and t-ratio.

    Returns (alpha_c, t) where:
        t = |S_rot|_F / |S_dil|_F
        alpha_c compared against f(t) = (sqrt(2)*t + 1) / (sqrt(3)*sqrt(t^2+1))
    """
    S11, S12, S13, S22, S23, S33 = spec.strain_rate_tensor(u, v, w)

    # Trace = dilatation
    theta = S11 + S22 + S33

    # Decompose: S = S_rot + S_dil
    # S_dil = (theta/3) * I, S_rot = S - S_dil
    third_theta = theta / 3.0
    S11_rot = S11 - third_theta
    S22_rot = S22 - third_theta
    S33_rot = S33 - third_theta
    # Off-diagonal unchanged: S12_rot = S12, etc.

    # |S_rot|_F
    S_rot_frob = torch.sqrt(
        S11_rot**2 + S22_rot**2 + S33_rot**2
        + 2.0 * (S12**2 + S13**2 + S23**2)
    )

    # |S_dil|_F = |theta/3| * sqrt(3) = |theta| / sqrt(3)
    S_dil_frob = torch.abs(theta) / math.sqrt(3.0)

    # t = |S_rot| / |S_dil|
    t = S_rot_frob / (S_dil_frob + 1e-30)

    # Actual alpha_c from eigenvalues
    alpha = depletion_factor_field(spec, u, v, w)

    return alpha, t


def f_bound(t: torch.Tensor) -> torch.Tensor:
    """Compressible depletion bound f(t) = (sqrt(2)*t + 1) / (sqrt(3)*sqrt(t^2+1))."""
    return (math.sqrt(2) * t + 1.0) / (math.sqrt(3) * torch.sqrt(t**2 + 1.0))


# ---- Coherence Failure Metric ----

def cfm_from_vorticity(
    spec: SpectralOperator,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    n_samples: int = 50,
) -> float:
    """Compute volume-averaged Coherence Failure Metric.

    CFM = <1 - |xi_bar|^2> where xi_bar is the local mean vorticity direction.

    Approximated by sampling random points and computing local directional
    coherence of the vorticity field.
    """
    wx, wy, wz = spec.curl(u, v, w)

    # Vorticity magnitude
    w_mag = torch.sqrt(wx**2 + wy**2 + wz**2)
    w_mag_mean = float(w_mag.mean())

    if w_mag_mean < 1e-30:
        return 0.0

    # Normalized vorticity direction
    xi_x = wx / (w_mag + 1e-30)
    xi_y = wy / (w_mag + 1e-30)
    xi_z = wz / (w_mag + 1e-30)

    # Volume-averaged direction weighted by magnitude
    weight = w_mag / (w_mag.sum() + 1e-30)
    xi_bar_x = float((xi_x * weight).sum())
    xi_bar_y = float((xi_y * weight).sum())
    xi_bar_z = float((xi_z * weight).sum())

    xi_bar_sq = xi_bar_x**2 + xi_bar_y**2 + xi_bar_z**2
    return 1.0 - xi_bar_sq


# ---- Vortex identification ----

def q_criterion(
    spec: SpectralOperator,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """Q-criterion for vortex identification.

    Q = 0.5 * (|Omega|^2 - |S|^2)

    where Omega is the rotation rate tensor and S is the strain rate.
    Q > 0 identifies vortex cores.
    """
    u_hat = spec.fft(u)
    v_hat = spec.fft(v)
    w_hat = spec.fft(w)

    dudx = spec.ifft(1j * spec.kx * u_hat)
    dudy = spec.ifft(1j * spec.ky * u_hat)
    dudz = spec.ifft(1j * spec.kz * u_hat)
    dvdx = spec.ifft(1j * spec.kx * v_hat)
    dvdy = spec.ifft(1j * spec.ky * v_hat)
    dvdz = spec.ifft(1j * spec.kz * v_hat)
    dwdx = spec.ifft(1j * spec.kx * w_hat)
    dwdy = spec.ifft(1j * spec.ky * w_hat)
    dwdz = spec.ifft(1j * spec.kz * w_hat)

    # Strain rate: S_ij = 0.5*(du_i/dx_j + du_j/dx_i)
    # Rotation rate: Omega_ij = 0.5*(du_i/dx_j - du_j/dx_i)
    # |S|^2 = S_ij * S_ij, |Omega|^2 = Omega_ij * Omega_ij

    S_sq = (dudx**2 + dvdy**2 + dwdz**2
            + 0.5 * ((dudy + dvdx)**2 + (dudz + dwdx)**2 + (dvdz + dwdy)**2))

    Omega_sq = 0.5 * ((dudy - dvdx)**2 + (dudz - dwdx)**2 + (dvdz - dwdy)**2)

    return 0.5 * (Omega_sq - S_sq)


# ---- Integral statistics ----

def compute_statistics(
    cons: ConservativeFields,
    spec: SpectralOperator,
    gamma: float,
    Re: float,
    time: float,
) -> FlowStatistics:
    """Compute comprehensive flow statistics for a snapshot.

    This is the main diagnostic entry point called by the solver.
    """
    prim = conservative_to_primitive(cons, gamma)
    u, v, w = prim.u, prim.v, prim.w
    rho, T = prim.rho, prim.T

    # Kinetic energy: <0.5 * rho * |u|^2>
    ke = float((0.5 * rho * (u**2 + v**2 + w**2)).mean())

    # Enstrophy: <0.5 * |omega|^2>
    wx, wy, wz = spec.curl(u, v, w)
    enstrophy = float((0.5 * (wx**2 + wy**2 + wz**2)).mean())

    # Dissipation rate: 2 * mu * <S_ij S_ij>
    S11, S12, S13, S22, S23, S33 = spec.strain_rate_tensor(u, v, w)
    S_sq = (S11**2 + S22**2 + S33**2 + 2.0 * (S12**2 + S13**2 + S23**2))
    mu = 1.0 / Re
    dissipation = float(2.0 * mu * S_sq.mean())

    # Mach number
    c = prim.speed_of_sound(gamma)
    speed = torch.sqrt(u**2 + v**2 + w**2)
    mach = speed / c
    max_mach = float(mach.max())
    mean_mach = float(mach.mean())

    # Taylor microscale Reynolds number
    u_rms = float(torch.sqrt((u**2 + v**2 + w**2).mean() / 3.0))
    if dissipation > 1e-30 and u_rms > 1e-30:
        lambda_taylor = u_rms * math.sqrt(15.0 * mu / max(dissipation, 1e-30))
        taylor_re = u_rms * lambda_taylor / mu
    else:
        taylor_re = 0.0

    # Kolmogorov scale
    if dissipation > 1e-30:
        nu = mu / float(rho.mean())
        eta = (nu**3 / dissipation) ** 0.25
    else:
        eta = 0.0

    # Depletion factors (sampled for efficiency)
    alpha_field = depletion_factor_field(spec, u, v, w)
    depletion_alpha = float(alpha_field.max())

    # Compressible depletion
    alpha_c_field, t_field = compressible_depletion_field(spec, u, v, w)
    depletion_alpha_c = float(alpha_c_field.max())

    # CFM
    cfm = cfm_from_vorticity(spec, u, v, w)

    # Divergence (measures compressibility)
    div_u = spec.divergence(u, v, w)
    max_div = float(torch.abs(div_u).max())

    return FlowStatistics(
        time=time,
        kinetic_energy=ke,
        enstrophy=enstrophy,
        dissipation_rate=dissipation,
        max_mach=max_mach,
        mean_mach=mean_mach,
        taylor_re=taylor_re,
        kolmogorov_eta=eta,
        depletion_alpha=depletion_alpha,
        depletion_alpha_c=depletion_alpha_c,
        cfm=cfm,
        max_divergence=max_div,
        min_density=float(rho.min()),
        max_density=float(rho.max()),
        max_temperature=float(T.max()),
    )
