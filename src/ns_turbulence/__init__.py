"""
ns-turbulence: GPU-accelerated pseudo-spectral compressible Navier-Stokes solver.

Built on PyTorch for seamless GPU acceleration. Implements the full
compressible NS equations with spectral derivatives, shock capturing,
and RCL-framework diagnostics for vortex stretching depletion analysis.
"""

from .config import GridConfig, PhysicsConfig, SimConfig, TimeConfig
from .diagnostics import (
    ALPHA_INCOMP,
    INCOMP_MARGIN,
    T_CRITICAL,
    FlowStatistics,
    cfm_from_vorticity,
    compressible_depletion_field,
    compute_statistics,
    depletion_factor_field,
    f_bound,
    q_criterion,
)
from .equations import CompressibleNS
from .fields import (
    ConservativeFields,
    PrimitiveFields,
    conservative_to_primitive,
    create_grid,
    create_uniform_field,
    primitive_to_conservative,
)
from .forcing import SpectralForcing
from .initial_conditions import (
    compressible_hit,
    shock_turbulence_interaction,
    taylor_green_vortex,
)
from .shock import ShockCapture
from .solver import Solver
from .spectral import SpectralOperator
from .thermo import (
    blottner_viscosity,
    dilatation_fraction,
    gamma_effective,
    post_shock_density_ratio,
    post_shock_pressure_ratio,
    post_shock_temperature,
    sutherland_viscosity,
    turbulent_mach_number,
)
from .timestepping import get_integrator, rk4_step, ssprk3_step

__version__ = "0.1.0"
__all__ = [
    "SimConfig",
    "GridConfig",
    "PhysicsConfig",
    "TimeConfig",
    "ConservativeFields",
    "PrimitiveFields",
    "conservative_to_primitive",
    "primitive_to_conservative",
    "create_uniform_field",
    "create_grid",
    "SpectralOperator",
    "CompressibleNS",
    "rk4_step",
    "ssprk3_step",
    "get_integrator",
    "ShockCapture",
    "SpectralForcing",
    "ALPHA_INCOMP",
    "T_CRITICAL",
    "INCOMP_MARGIN",
    "depletion_factor_field",
    "compressible_depletion_field",
    "f_bound",
    "cfm_from_vorticity",
    "q_criterion",
    "compute_statistics",
    "FlowStatistics",
    "gamma_effective",
    "sutherland_viscosity",
    "blottner_viscosity",
    "post_shock_temperature",
    "post_shock_density_ratio",
    "post_shock_pressure_ratio",
    "turbulent_mach_number",
    "dilatation_fraction",
    "Solver",
    "taylor_green_vortex",
    "compressible_hit",
    "shock_turbulence_interaction",
]
