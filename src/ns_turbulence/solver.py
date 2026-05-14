"""
Main simulation driver: orchestrates time integration, shock capturing,
forcing, diagnostics, and I/O into a single solver loop.

Usage:
    config = SimConfig(...)
    solver = Solver(config)
    solver.initialize(initial_condition_fn)
    solver.run()
"""

from __future__ import annotations

import os
import time as wall_time
from typing import Callable, Optional

import torch

from .config import SimConfig
from .diagnostics import FlowStatistics, compute_statistics
from .equations import CompressibleNS
from .fields import (
    ConservativeFields,
    conservative_to_primitive,
    create_grid,
)
from .forcing import SpectralForcing
from .io import StatsLogger, ensure_dir, save_checkpoint
from .shock import ShockCapture
from .spectral import SpectralOperator
from .timestepping import get_integrator


class Solver:
    """Main compressible Navier-Stokes solver.

    Pseudo-spectral method on periodic domain with:
        - SSPRK3 or RK4 time integration
        - Exponential spectral filtering for shock capture
        - Cook-Cabot artificial viscosity with Ducros sensor
        - Solenoidal spectral-band forcing
        - Full RCL diagnostic suite
    """

    def __init__(self, config: SimConfig) -> None:
        config.validate()
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_dtype()

        # Set random seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        # Build operators
        self.spec = SpectralOperator.from_config(config)
        self.ns = CompressibleNS(config, self.spec)
        self.shock = ShockCapture(config.shock, self.spec, config.physics.gamma)
        self.forcing = SpectralForcing(config.forcing, self.spec)
        self.integrator = get_integrator(config.time.integrator)

        # Grid
        self.X, self.Y, self.Z = create_grid(config)

        # State
        self.U: Optional[ConservativeFields] = None
        self.step = 0
        self.time = 0.0

        # I/O
        self.stats_logger: Optional[StatsLogger] = None
        self.stats_history: list[FlowStatistics] = []

    def initialize(
        self,
        init_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, SimConfig],
            ConservativeFields,
        ],
    ) -> None:
        """Initialize fields from an initial condition function.

        init_fn(X, Y, Z, config) -> ConservativeFields
        """
        self.U = init_fn(self.X, self.Y, self.Z, self.config)
        self.step = 0
        self.time = 0.0

    def _rhs_with_forcing(self, U: ConservativeFields) -> ConservativeFields:
        """RHS including forcing terms."""
        dUdt = self.ns.rhs(U)

        if self.config.forcing.enabled:
            Fx, Fy, Fz = self.forcing.compute_forcing(
                U.rhou, U.rhov, U.rhow, U.rho, self.config.time.dt
            )
            dUdt = ConservativeFields(
                rho=dUdt.rho,
                rhou=dUdt.rhou + Fx,
                rhov=dUdt.rhov + Fy,
                rhow=dUdt.rhow + Fz,
                E=dUdt.E,
            )

        # Add artificial viscosity
        if self.config.shock.artificial_viscosity:
            prim = conservative_to_primitive(U, self.config.physics.gamma)
            av = self.shock.artificial_viscosity(U, prim)
            dUdt = dUdt + av

        return dUdt

    def advance(self, dt: float) -> None:
        """Advance one time step."""
        # Time integration
        self.U = self.integrator(self.U, self._rhs_with_forcing, dt)

        # Spectral filtering (shock capture)
        if self.config.shock.enabled:
            self.U = self.shock.apply_spectral_filter(self.U)

        # Dealiasing
        self.U = ConservativeFields(
            rho=self.spec.dealias(self.U.rho),
            rhou=self.spec.dealias(self.U.rhou),
            rhov=self.spec.dealias(self.U.rhov),
            rhow=self.spec.dealias(self.U.rhow),
            E=self.spec.dealias(self.U.E),
        )

        self.step += 1
        self.time += dt

    def run(self, callback: Optional[Callable] = None) -> list[FlowStatistics]:
        """Run simulation from current state to t_end.

        Args:
            callback: Optional function called each stats interval with
                      (solver, stats) arguments.

        Returns:
            List of FlowStatistics collected during the run.
        """
        if self.U is None:
            raise RuntimeError("Solver not initialized. Call initialize() first.")

        config = self.config
        output_dir = config.io.output_dir
        ensure_dir(output_dir)

        # Stats logger
        self.stats_logger = StatsLogger(os.path.join(output_dir, "stats.csv"))
        self.stats_history = []

        t0 = wall_time.time()
        prim = conservative_to_primitive(self.U, config.physics.gamma)

        while self.time < config.time.t_end:
            # Adaptive time step
            if config.time.adaptive_dt:
                dt = self.ns.compute_dt(prim)
                dt = min(dt, config.time.t_end - self.time)
            else:
                dt = min(config.time.dt, config.time.t_end - self.time)

            if dt <= 0:
                break

            # Advance
            self.advance(dt)

            # Update primitives
            prim = conservative_to_primitive(self.U, config.physics.gamma)

            # Statistics
            if self.step % config.io.stats_interval == 0:
                stats = compute_statistics(
                    self.U, self.spec, config.physics.gamma,
                    config.physics.reynolds, self.time,
                )
                self.stats_history.append(stats)
                self.stats_logger.log(stats)

                if callback is not None:
                    callback(self, stats)

            # Checkpoint
            if self.step % config.io.checkpoint_interval == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint_{self.step:08d}.pt")
                save_checkpoint(self.U, config, self.step, self.time, ckpt_path)

            # Print progress
            if self.step % config.io.stats_interval == 0:
                elapsed = wall_time.time() - t0
                print(
                    f"  step {self.step:6d}  t={self.time:.4e}  dt={dt:.2e}"
                    f"  Ma_max={float(prim.local_mach(config.physics.gamma).max()):.3f}"
                    f"  rho=[{float(prim.rho.min()):.3f},{float(prim.rho.max()):.3f}]"
                    f"  wall={elapsed:.1f}s"
                )

        # Final checkpoint
        save_checkpoint(
            self.U, config, self.step, self.time,
            os.path.join(output_dir, "checkpoint_final.pt"),
        )

        self.stats_logger.close()

        total_time = wall_time.time() - t0
        print(
            f"\n  Simulation complete: {self.step} steps,"
            f" t={self.time:.4e}, wall={total_time:.1f}s"
        )

        return self.stats_history
