"""
Taylor-Green Vortex — canonical 3D turbulence benchmark.

Compressible Taylor-Green at Re=1600, Ma=0.1 on a 64^3 grid.
Demonstrates solver setup, execution, and diagnostic output.

Usage:
    python examples/taylor_green.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ns_turbulence import (
    SimConfig,
    Solver,
    taylor_green_vortex,
)


def main() -> None:
    config = SimConfig()
    config.grid.nx = config.grid.ny = config.grid.nz = 64
    config.physics.reynolds = 1600.0
    config.physics.mach = 0.1
    config.physics.gamma = 1.4
    config.time.t_end = 5.0
    config.time.integrator = "ssprk3"
    config.time.adaptive_dt = True
    config.time.cfl = 0.5
    config.shock.enabled = True
    config.shock.artificial_viscosity = False  # low Ma, no shocks
    config.forcing.enabled = False
    config.diagnostics.enabled = True
    config.io.output_dir = "outputs/taylor_green"
    config.io.stats_interval = 50
    config.io.checkpoint_interval = 5000

    print("=" * 60)
    print("  Taylor-Green Vortex")
    print(f"  Grid: {config.grid.shape}")
    print(f"  Re={config.physics.reynolds}, Ma={config.physics.mach}")
    print(f"  Device: {config.get_device()}")
    print("=" * 60)
    print()

    solver = Solver(config)
    solver.initialize(taylor_green_vortex)
    stats = solver.run()

    if stats:
        print("\n  Final statistics:")
        print(f"    KE = {stats[-1].kinetic_energy:.6e}")
        print(f"    Enstrophy = {stats[-1].enstrophy:.6e}")
        print(f"    Dissipation = {stats[-1].dissipation_rate:.6e}")
        print(f"    Max depletion alpha = {stats[-1].depletion_alpha:.4f}")
        print(f"    Max depletion alpha_c = {stats[-1].depletion_alpha_c:.4f}")


if __name__ == "__main__":
    main()
