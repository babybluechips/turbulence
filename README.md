# ns-turbulence

**GPU-accelerated pseudo-spectral compressible Navier-Stokes solver with RCL diagnostics**

[![CI](https://github.com/arkgart/ns-turbulence/actions/workflows/ci.yml/badge.svg)](https://github.com/arkgart/ns-turbulence/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

A production-grade compressible Navier-Stokes solver built on PyTorch for seamless GPU acceleration. Implements pseudo-spectral spatial discretization with FFT-based derivatives on triply periodic domains, targeting direct numerical simulation (DNS) of compressible turbulence from Mach 0.1 to 50+.

The solver integrates the Root Coupling Laplacian (RCL) diagnostic framework from the companion [ns-regularity](https://github.com/arkgart/ns-regularity) package, providing real-time tracking of vortex stretching depletion, coherence failure, and spectral gap evolution during simulation.

### Key features

- **Full compressible NS** — Conservative formulation with continuity, momentum, and energy equations
- **Pseudo-spectral method** — FFT-based derivatives via `torch.fft` with 2/3-rule dealiasing
- **GPU-native** — All operations on PyTorch tensors; zero-copy CPU↔GPU, automatic device selection
- **Shock capturing** — Exponential spectral filter + Cook-Cabot artificial viscosity with Ducros sensor
- **Real-gas thermochemistry** — Temperature-dependent γ(T), Sutherland/Blottner viscosity
- **RCL diagnostics** — Depletion factor α, compressible bound f(t), CFM, Q-criterion, energy spectra
- **Time integration** — SSPRK3 (shock-safe) and classical RK4 with adaptive CFL
- **Solenoidal forcing** — Spectral-band energy injection with Leray projection
- **HDF5 checkpointing** — Full restart capability with statistics logging

### Mathematical formulation

The solver advances the conservative variables **U** = (ρ, ρ**u**, E) via:

$$\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F}(\mathbf{U}) = \nabla \cdot \mathbf{F}_v(\mathbf{U}) + \mathbf{S}$$

where **F** are the Euler fluxes, **F**_v the viscous fluxes (Newtonian stress + Fourier heat conduction), and **S** includes forcing and artificial viscosity.

The RCL depletion diagnostic tracks the bound:

$$\alpha_c \leq f(t) = \frac{\sqrt{2}\,t + 1}{\sqrt{3}\,\sqrt{t^2 + 1}}$$

where t = |S_rot|/|S_dil|. The incompressible limit gives α ≤ √(2/3) ≈ 0.817.

## Installation

```bash
git clone https://github.com/arkgart/ns-turbulence.git
cd ns-turbulence
pip install -e ".[dev]"
```

For GPU support, install PyTorch with CUDA first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Quick start

```python
from ns_turbulence import SimConfig, Solver, taylor_green_vortex

config = SimConfig()
config.grid.nx = config.grid.ny = config.grid.nz = 128
config.physics.reynolds = 1600.0
config.physics.mach = 0.1
config.time.t_end = 10.0

solver = Solver(config)
solver.initialize(taylor_green_vortex)
stats = solver.run()

# Access diagnostics
print(f"Final KE: {stats[-1].kinetic_energy:.6e}")
print(f"Depletion α: {stats[-1].depletion_alpha:.4f}")
```

## Examples

```bash
python examples/taylor_green.py           # Taylor-Green vortex (Re=1600, Ma=0.1)
python examples/compressible_hit.py       # Compressible HIT (Re=400, Ma=0.5)
python examples/shock_turbulence.py       # Shock-turbulence interaction (Ma=2.0)
```

## Configuration

All parameters are controlled via a `SimConfig` dataclass or YAML file:

```yaml
grid:
  nx: 128
  ny: 128
  nz: 128
physics:
  gamma: 1.4
  reynolds: 1600.0
  mach: 0.5
  prandtl: 0.71
  viscosity_model: sutherland
time:
  integrator: ssprk3
  cfl: 0.5
  t_end: 10.0
shock:
  enabled: true
  filter_order: 16
  artificial_viscosity: true
  ducros_sensor: true
device: auto
dtype: float64
```

```bash
ns-turbulence run config.yaml
ns-turbulence taylor-green --nx 128 --re 1600 --mach 0.1
ns-turbulence info config.yaml
```

## Repository structure

```
ns-turbulence/
├── src/ns_turbulence/
│   ├── config.py               Configuration dataclasses
│   ├── fields.py               Conservative/primitive field containers
│   ├── spectral.py             FFT spectral operators (derivatives, filter)
│   ├── equations.py            Compressible NS RHS (inviscid + viscous)
│   ├── timestepping.py         RK4, SSPRK3 integrators
│   ├── shock.py                Spectral filter, AV, Ducros sensor
│   ├── thermo.py               Real-gas thermochemistry
│   ├── forcing.py              Solenoidal spectral-band forcing
│   ├── diagnostics.py          RCL depletion, CFM, Q-criterion, spectra
│   ├── initial_conditions.py   TGV, compressible HIT, shock-turbulence
│   ├── solver.py               Main simulation driver
│   ├── io.py                   HDF5 checkpointing, stats CSV
│   └── cli.py                  Command-line interface
├── examples/                   Ready-to-run simulations
├── benchmarks/                 GPU scaling benchmarks
├── tests/                      Pytest test suite (70+ tests)
├── .github/workflows/ci.yml   GitHub Actions CI
├── pyproject.toml              Package configuration
└── Makefile                    Common operations
```

## Running the tests

```bash
make test          # unit + integration tests
make lint          # ruff linting
make all           # install + test + lint
```

## GPU performance

The solver is designed for GPU execution. All field operations, FFTs, and diagnostics run on PyTorch tensors and automatically use CUDA when available.

```bash
python benchmarks/gpu_scaling.py
```

Typical throughput on an A100:

| Grid | DOFs | RHS time | DOFs/s |
|------|------|----------|--------|
| 64³ | 262K | ~5 ms | 5.2×10⁷ |
| 128³ | 2.1M | ~25 ms | 8.4×10⁷ |
| 256³ | 16.8M | ~180 ms | 9.3×10⁷ |

## Citation

```bibtex
@unpublished{gulati2026turbulence,
  author = {Gulati, Rick},
  title  = {GPU-Accelerated Pseudo-Spectral Compressible {Navier--Stokes}
            with {RCL} Diagnostics},
  year   = {2026},
  note   = {University of Pennsylvania}
}
```

## License

MIT — see [LICENSE](LICENSE).
# turbulence
