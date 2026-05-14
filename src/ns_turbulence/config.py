"""
Simulation configuration for the compressible Navier-Stokes solver.

Dataclass-based configuration with validation, serialization,
and sensible defaults for compressible turbulence DNS.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import yaml


@dataclass
class GridConfig:
    """Computational grid parameters."""

    nx: int = 64
    ny: int = 64
    nz: int = 64
    lx: float = 2 * math.pi
    ly: float = 2 * math.pi
    lz: float = 2 * math.pi

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def dz(self) -> float:
        return self.lz / self.nz

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    def validate(self) -> None:
        for n in (self.nx, self.ny, self.nz):
            if n < 4:
                raise ValueError(f"Grid dimension {n} too small (min 4)")
            if n & (n - 1) != 0:
                raise ValueError(f"Grid dimension {n} must be a power of 2 for FFT efficiency")


@dataclass
class PhysicsConfig:
    """Physical parameters for compressible flow."""

    gamma: float = 1.4
    reynolds: float = 1600.0
    mach: float = 0.5
    prandtl: float = 0.71
    reference_temperature: float = 300.0
    use_real_gas: bool = False
    viscosity_model: str = "constant"  # "constant", "sutherland", "blottner"

    def validate(self) -> None:
        if self.gamma <= 1.0:
            raise ValueError(f"gamma must be > 1, got {self.gamma}")
        if self.reynolds <= 0:
            raise ValueError(f"Reynolds must be positive, got {self.reynolds}")
        if self.mach <= 0:
            raise ValueError(f"Mach must be positive, got {self.mach}")
        if self.viscosity_model not in ("constant", "sutherland", "blottner"):
            raise ValueError(f"Unknown viscosity model: {self.viscosity_model}")


@dataclass
class TimeConfig:
    """Time integration parameters."""

    dt: float = 1e-3
    t_end: float = 10.0
    cfl: float = 0.5
    adaptive_dt: bool = True
    integrator: str = "ssprk3"  # "rk4", "ssprk3"

    def validate(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.integrator not in ("rk4", "ssprk3"):
            raise ValueError(f"Unknown integrator: {self.integrator}")


@dataclass
class ShockConfig:
    """Shock-capturing parameters."""

    enabled: bool = True
    filter_order: int = 16
    filter_cutoff: float = 0.65
    artificial_viscosity: bool = True
    av_coefficient: float = 1.0
    ducros_sensor: bool = True

    def validate(self) -> None:
        if self.filter_order < 2:
            raise ValueError(f"Filter order must be >= 2, got {self.filter_order}")


@dataclass
class ForcingConfig:
    """Turbulence forcing parameters."""

    enabled: bool = False
    method: str = "spectral_band"  # "spectral_band", "linear"
    k_min: float = 1.0
    k_max: float = 2.5
    energy_injection_rate: float = 0.1


@dataclass
class DiagnosticsConfig:
    """Diagnostic output parameters."""

    enabled: bool = True
    interval: int = 100
    compute_depletion: bool = True
    compute_cfm: bool = True
    compute_spectral_gap: bool = True
    compute_energy_spectrum: bool = True
    n_sample_points: int = 50


@dataclass
class IOConfig:
    """Input/output configuration."""

    output_dir: str = "outputs"
    checkpoint_interval: int = 1000
    stats_interval: int = 10
    field_output_interval: int = 500
    save_format: str = "hdf5"  # "hdf5", "numpy"


@dataclass
class SimConfig:
    """Master simulation configuration."""

    grid: GridConfig = field(default_factory=GridConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    shock: ShockConfig = field(default_factory=ShockConfig)
    forcing: ForcingConfig = field(default_factory=ForcingConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    io: IOConfig = field(default_factory=IOConfig)
    device: str = "auto"
    dtype: str = "float64"
    seed: int = 42

    def validate(self) -> None:
        """Validate all sub-configs."""
        self.grid.validate()
        self.physics.validate()
        self.time.validate()
        self.shock.validate()

    def get_device(self) -> torch.device:
        """Resolve device string to torch.device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def get_dtype(self) -> torch.dtype:
        """Resolve dtype string to torch.dtype."""
        return {"float32": torch.float32, "float64": torch.float64}[self.dtype]

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> SimConfig:
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            grid=GridConfig(**data.get("grid", {})),
            physics=PhysicsConfig(**data.get("physics", {})),
            time=TimeConfig(**data.get("time", {})),
            shock=ShockConfig(**data.get("shock", {})),
            forcing=ForcingConfig(**data.get("forcing", {})),
            diagnostics=DiagnosticsConfig(**data.get("diagnostics", {})),
            io=IOConfig(**data.get("io", {})),
            device=data.get("device", "auto"),
            dtype=data.get("dtype", "float64"),
            seed=data.get("seed", 42),
        )
