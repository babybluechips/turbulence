"""
Command-line interface for the compressible Navier-Stokes solver.

Usage:
    ns-turbulence run config.yaml
    ns-turbulence taylor-green --nx 64 --re 1600 --mach 0.5 --t-end 10
    ns-turbulence info config.yaml
"""

from __future__ import annotations

import argparse
import sys

import torch

from .config import SimConfig


def cmd_run(args: argparse.Namespace) -> None:
    """Run simulation from YAML config file."""
    from .solver import Solver

    config = SimConfig.from_yaml(args.config)
    if args.device:
        config.device = args.device
    config.validate()

    print(f"ns-turbulence: loading config from {args.config}")
    print(f"  Grid: {config.grid.shape}")
    print(f"  Re={config.physics.reynolds}, Ma={config.physics.mach}")
    print(f"  Device: {config.get_device()}")
    print(f"  Integrator: {config.time.integrator}")
    print()

    solver = Solver(config)

    # Use Taylor-Green as default IC
    from .initial_conditions import taylor_green_vortex
    solver.initialize(taylor_green_vortex)
    solver.run()


def cmd_taylor_green(args: argparse.Namespace) -> None:
    """Run Taylor-Green vortex benchmark."""
    from .initial_conditions import taylor_green_vortex
    from .solver import Solver

    config = SimConfig()
    config.grid.nx = config.grid.ny = config.grid.nz = args.nx
    config.physics.reynolds = args.re
    config.physics.mach = args.mach
    config.time.t_end = args.t_end
    config.device = args.device or "auto"
    config.validate()

    print("ns-turbulence: Taylor-Green Vortex")
    print(f"  Grid: {config.grid.shape}, Re={args.re}, Ma={args.mach}")
    print(f"  Device: {config.get_device()}")
    print()

    solver = Solver(config)
    solver.initialize(taylor_green_vortex)
    solver.run()


def cmd_info(args: argparse.Namespace) -> None:
    """Print config info without running."""
    config = SimConfig.from_yaml(args.config)
    print(f"Configuration: {args.config}")
    print(f"  Grid: {config.grid.shape}")
    print(f"  Domain: [{config.grid.lx}, {config.grid.ly}, {config.grid.lz}]")
    print(f"  Re={config.physics.reynolds}, Ma={config.physics.mach}, gamma={config.physics.gamma}")
    print(f"  Pr={config.physics.prandtl}, viscosity={config.physics.viscosity_model}")
    print(f"  Integrator: {config.time.integrator}, CFL={config.time.cfl}")
    print(f"  t_end={config.time.t_end}")
    print(f"  Shock capture: {config.shock.enabled}")
    print(f"  Forcing: {config.forcing.enabled}")
    print(f"  Diagnostics: {config.diagnostics.enabled}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ns-turbulence",
        description="GPU-accelerated compressible Navier-Stokes solver",
    )
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run simulation from YAML config")
    p_run.add_argument("config", help="Path to YAML config file")
    p_run.add_argument("--device", help="Device override (cpu/cuda)")

    # taylor-green
    p_tg = sub.add_parser("taylor-green", help="Taylor-Green vortex benchmark")
    p_tg.add_argument("--nx", type=int, default=64, help="Grid resolution")
    p_tg.add_argument("--re", type=float, default=1600, help="Reynolds number")
    p_tg.add_argument("--mach", type=float, default=0.5, help="Mach number")
    p_tg.add_argument("--t-end", type=float, default=10.0, help="End time")
    p_tg.add_argument("--device", help="Device override (cpu/cuda)")

    # info
    p_info = sub.add_parser("info", help="Print config info")
    p_info.add_argument("config", help="Path to YAML config file")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "taylor-green":
        cmd_taylor_green(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
