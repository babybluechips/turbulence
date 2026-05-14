"""
Compressible Homogeneous Isotropic Turbulence (HIT).

Decaying compressible turbulence at Ma=0.5 with diagnostics tracking
the RCL depletion factor evolution as the flow transitions.

Usage:
    python examples/compressible_hit.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ns_turbulence import (
    SimConfig,
    Solver,
    compressible_hit,
)


def main() -> None:
    config = SimConfig()
    config.grid.nx = config.grid.ny = config.grid.nz = 64
    config.physics.reynolds = 400.0
    config.physics.mach = 0.5
    config.physics.gamma = 1.4
    config.time.t_end = 5.0
    config.time.integrator = "ssprk3"
    config.time.adaptive_dt = True
    config.time.cfl = 0.4
    config.shock.enabled = True
    config.shock.artificial_viscosity = True
    config.shock.ducros_sensor = True
    config.forcing.enabled = False
    config.io.output_dir = "outputs/compressible_hit"
    config.io.stats_interval = 20
    config.io.checkpoint_interval = 5000

    print("=" * 60)
    print("  Compressible HIT (Decaying)")
    print(f"  Grid: {config.grid.shape}")
    print(f"  Re={config.physics.reynolds}, Ma={config.physics.mach}")
    print(f"  Device: {config.get_device()}")
    print("=" * 60)
    print()

    solver = Solver(config)
    solver.initialize(compressible_hit)
    stats = solver.run()

    if stats:
        print("\n  Final statistics:")
        print(f"    KE = {stats[-1].kinetic_energy:.6e}")
        print(f"    Enstrophy = {stats[-1].enstrophy:.6e}")
        print(f"    Max Ma = {stats[-1].max_mach:.4f}")
        print(f"    Depletion alpha = {stats[-1].depletion_alpha:.4f}")
        print(f"    Depletion alpha_c = {stats[-1].depletion_alpha_c:.4f}")
        print(f"    CFM = {stats[-1].cfm:.4f}")


if __name__ == "__main__":
    main()
