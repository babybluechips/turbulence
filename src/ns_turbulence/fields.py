"""
GPU tensor field containers for compressible Navier-Stokes.

Manages conservative variables (rho, rho*u, rho*v, rho*w, E)
and primitive variables (rho, u, v, w, p, T) on PyTorch tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import SimConfig


@dataclass
class ConservativeFields:
    """Conservative variable set: (rho, rho*u, rho*v, rho*w, E).

    All tensors have shape (nx, ny, nz) on the configured device.
    """

    rho: torch.Tensor       # density
    rhou: torch.Tensor      # x-momentum
    rhov: torch.Tensor      # y-momentum
    rhow: torch.Tensor      # z-momentum
    E: torch.Tensor         # total energy per unit volume

    def clone(self) -> ConservativeFields:
        return ConservativeFields(
            rho=self.rho.clone(),
            rhou=self.rhou.clone(),
            rhov=self.rhov.clone(),
            rhow=self.rhow.clone(),
            E=self.E.clone(),
        )

    def __add__(self, other: ConservativeFields) -> ConservativeFields:
        return ConservativeFields(
            rho=self.rho + other.rho,
            rhou=self.rhou + other.rhou,
            rhov=self.rhov + other.rhov,
            rhow=self.rhow + other.rhow,
            E=self.E + other.E,
        )

    def __mul__(self, scalar: float) -> ConservativeFields:
        return ConservativeFields(
            rho=self.rho * scalar,
            rhou=self.rhou * scalar,
            rhov=self.rhov * scalar,
            rhow=self.rhow * scalar,
            E=self.E * scalar,
        )

    def __rmul__(self, scalar: float) -> ConservativeFields:
        return self.__mul__(scalar)

    @property
    def device(self) -> torch.device:
        return self.rho.device

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.rho.shape)

    def max_density(self) -> float:
        return float(self.rho.max())

    def min_density(self) -> float:
        return float(self.rho.min())

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return (self.rho, self.rhou, self.rhov, self.rhow, self.E)

    def to(self, device: torch.device) -> ConservativeFields:
        return ConservativeFields(
            rho=self.rho.to(device),
            rhou=self.rhou.to(device),
            rhov=self.rhov.to(device),
            rhow=self.rhow.to(device),
            E=self.E.to(device),
        )


@dataclass
class PrimitiveFields:
    """Primitive variable set: (rho, u, v, w, p, T).

    All tensors have shape (nx, ny, nz) on the configured device.
    """

    rho: torch.Tensor    # density
    u: torch.Tensor      # x-velocity
    v: torch.Tensor      # y-velocity
    w: torch.Tensor      # z-velocity
    p: torch.Tensor      # pressure
    T: torch.Tensor      # temperature

    @property
    def device(self) -> torch.device:
        return self.rho.device

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.rho.shape)

    def kinetic_energy(self) -> torch.Tensor:
        """Kinetic energy per unit volume: 0.5 * rho * |u|^2."""
        return 0.5 * self.rho * (self.u**2 + self.v**2 + self.w**2)

    def speed_of_sound(self, gamma: float) -> torch.Tensor:
        """Local speed of sound: c = sqrt(gamma * p / rho)."""
        return torch.sqrt(gamma * self.p / self.rho)

    def local_mach(self, gamma: float) -> torch.Tensor:
        """Local Mach number: |u| / c."""
        speed = torch.sqrt(self.u**2 + self.v**2 + self.w**2)
        c = self.speed_of_sound(gamma)
        return speed / c

    def to(self, device: torch.device) -> PrimitiveFields:
        return PrimitiveFields(
            rho=self.rho.to(device),
            u=self.u.to(device),
            v=self.v.to(device),
            w=self.w.to(device),
            p=self.p.to(device),
            T=self.T.to(device),
        )


def conservative_to_primitive(
    cons: ConservativeFields, gamma: float
) -> PrimitiveFields:
    """Convert conservative to primitive variables.

    rho = rho
    u = rhou / rho
    v = rhov / rho
    w = rhow / rho
    p = (gamma - 1) * (E - 0.5 * rho * (u^2 + v^2 + w^2))
    T = p / (rho * R)  where R = 1/(gamma-1) in non-dim units → T = gamma * p / rho
    """
    rho = cons.rho
    u = cons.rhou / rho
    v = cons.rhov / rho
    w = cons.rhow / rho
    ke = 0.5 * rho * (u**2 + v**2 + w**2)
    p = (gamma - 1.0) * (cons.E - ke)
    # Non-dimensional temperature: T = p / rho (with R=1 normalization)
    T = p / rho
    return PrimitiveFields(rho=rho, u=u, v=v, w=w, p=p, T=T)


def primitive_to_conservative(
    prim: PrimitiveFields, gamma: float
) -> ConservativeFields:
    """Convert primitive to conservative variables.

    rho = rho
    rhou = rho * u
    rhov = rho * v
    rhow = rho * w
    E = p / (gamma - 1) + 0.5 * rho * (u^2 + v^2 + w^2)
    """
    rho = prim.rho
    rhou = rho * prim.u
    rhov = rho * prim.v
    rhow = rho * prim.w
    ke = 0.5 * rho * (prim.u**2 + prim.v**2 + prim.w**2)
    E = prim.p / (gamma - 1.0) + ke
    return ConservativeFields(rho=rho, rhou=rhou, rhov=rhov, rhow=rhow, E=E)


def create_uniform_field(
    config: SimConfig,
    rho0: float = 1.0,
    u0: float = 0.0,
    v0: float = 0.0,
    w0: float = 0.0,
    p0: float = 1.0,
) -> ConservativeFields:
    """Create a uniform flow field in conservative variables."""
    device = config.get_device()
    dtype = config.get_dtype()
    shape = config.grid.shape

    rho = torch.full(shape, rho0, device=device, dtype=dtype)
    u = torch.full(shape, u0, device=device, dtype=dtype)
    v = torch.full(shape, v0, device=device, dtype=dtype)
    w = torch.full(shape, w0, device=device, dtype=dtype)
    p = torch.full(shape, p0, device=device, dtype=dtype)
    T = p / rho

    prim = PrimitiveFields(rho=rho, u=u, v=v, w=w, p=p, T=T)
    return primitive_to_conservative(prim, config.physics.gamma)


def create_grid(config: SimConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create 3D coordinate grids on the configured device."""
    device = config.get_device()
    dtype = config.get_dtype()
    g = config.grid

    x = torch.linspace(0, g.lx, g.nx + 1, device=device, dtype=dtype)[:-1]
    y = torch.linspace(0, g.ly, g.ny + 1, device=device, dtype=dtype)[:-1]
    z = torch.linspace(0, g.lz, g.nz + 1, device=device, dtype=dtype)[:-1]

    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z
