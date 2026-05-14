"""Integration tests: short simulation runs."""

from __future__ import annotations

from ns_turbulence import (
    SimConfig,
    Solver,
    compressible_hit,
    taylor_green_vortex,
)


def _small_config():
    config = SimConfig()
    config.grid.nx = config.grid.ny = config.grid.nz = 8
    config.device = "cpu"
    config.dtype = "float64"
    config.physics.reynolds = 100.0
    config.physics.mach = 0.1
    config.time.t_end = 0.01
    config.time.cfl = 0.3
    config.time.adaptive_dt = True
    config.shock.enabled = False
    config.forcing.enabled = False
    config.io.stats_interval = 1
    config.io.checkpoint_interval = 100000
    config.io.output_dir = "/tmp/ns_test_output"
    return config


class TestTaylorGreen:
    def test_runs_without_error(self):
        config = _small_config()
        solver = Solver(config)
        solver.initialize(taylor_green_vortex)
        stats = solver.run()
        assert len(stats) > 0

    def test_energy_decreases(self):
        """Kinetic energy should decrease for decaying TGV."""
        config = _small_config()
        config.time.t_end = 0.05
        solver = Solver(config)
        solver.initialize(taylor_green_vortex)
        stats = solver.run()
        if len(stats) >= 2:
            assert stats[-1].kinetic_energy <= stats[0].kinetic_energy * 1.01

    def test_density_stays_positive(self):
        config = _small_config()
        solver = Solver(config)
        solver.initialize(taylor_green_vortex)
        stats = solver.run()
        for s in stats:
            assert s.min_density > 0


class TestCompressibleHIT:
    def test_runs_without_error(self):
        config = _small_config()
        config.physics.mach = 0.3
        solver = Solver(config)
        solver.initialize(compressible_hit)
        stats = solver.run()
        assert len(stats) > 0
