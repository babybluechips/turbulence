"""
Shock-Turbulence Interaction.

Normal shock interacting with upstream isotropic turbulence.
Demonstrates shock-capturing capabilities and the compressible
depletion bound f(t) approaching criticality near the shock.

Usage:
    python examples/shock_turbulence.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ns_turbulence import (
    SimConfig,
    Solver,
    shock_turbulence_interaction,
)


def main() -> None:
    config = SimConfig()
    config.grid.nx = config.grid.ny = config.grid.nz = 64
    config.physics.reynolds = 200.0
    config.physics.mach = 2.0
    config.physics.gamma = 1.4
    config.time.t_end = 2.0
    config.time.integrator = "ssprk3"
    config.time.adaptive_dt = True
    config.time.cfl = 0.3
    config.shock.enabled = True
    config.shock.filter_order = 16
    config.shock.filter_cutoff = 0.6
    config.shock.artificial_viscosity = True
    config.shock.av_coefficient = 2.0
    config.shock.ducros_sensor = True
    config.forcing.enabled = False
    config.io.output_dir = "outputs/shock_turbulence"
    config.io.stats_interval = 20
    config.io.checkpoint_interval = 5000

    print("=" * 60)
    print("  Shock-Turbulence Interaction")
    print(f"  Grid: {config.grid.shape}")
    print(f"  Re={config.physics.reynolds}, Ma={config.physics.mach}")
    print(f"  Device: {config.get_device()}")
    print("=" * 60)
    print()

    solver = Solver(config)
    solver.initialize(shock_turbulence_interaction)
    stats = solver.run()

    if stats:
        print("\n  Final statistics:")
        print(f"    KE = {stats[-1].kinetic_energy:.6e}")
        print(f"    Max Ma = {stats[-1].max_mach:.4f}")
        print(f"    rho range = [{stats[-1].min_density:.3f}, {stats[-1].max_density:.3f}]")
        print(f"    Depletion alpha = {stats[-1].depletion_alpha:.4f}")
        print(f"    Depletion alpha_c = {stats[-1].depletion_alpha_c:.4f}")


if __name__ == "__main__":
    main()
