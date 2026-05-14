"""
GPU performance benchmark: measures throughput vs grid size.

Reports:
    - Time per RHS evaluation
    - Effective GFLOPS
    - Memory usage
    - Scaling with N^3

Usage:
    python benchmarks/gpu_scaling.py
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ns_turbulence import (
    SimConfig,
    SpectralOperator,
    CompressibleNS,
    create_grid,
    taylor_green_vortex,
)


def benchmark_rhs(nx: int, device_str: str = "auto", n_warmup: int = 5, n_iters: int = 20):
    """Benchmark RHS evaluation at given resolution."""
    config = SimConfig()
    config.grid.nx = config.grid.ny = config.grid.nz = nx
    config.physics.reynolds = 1600.0
    config.physics.mach = 0.1
    config.device = device_str
    config.dtype = "float64"

    device = config.get_device()
    spec = SpectralOperator.from_config(config)
    ns = CompressibleNS(config, spec)

    X, Y, Z = create_grid(config)
    U = taylor_green_vortex(X, Y, Z, config)

    # Warmup
    for _ in range(n_warmup):
        _ = ns.rhs(U)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed iterations
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = ns.rhs(U)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    time_per_rhs = elapsed / n_iters
    dofs = nx ** 3
    dofs_per_sec = dofs / time_per_rhs

    # Memory
    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        mem_mb = 0.0

    return {
        "nx": nx,
        "dofs": dofs,
        "time_per_rhs_ms": time_per_rhs * 1e3,
        "dofs_per_sec": dofs_per_sec,
        "mem_mb": mem_mb,
        "device": str(device),
    }


def main():
    resolutions = [16, 32, 64]

    device = "auto"
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        resolutions.append(128)
    else:
        print("Running on CPU (no GPU detected)")

    print(f"\n{'N':>6} {'DOFs':>10} {'RHS (ms)':>10} {'DOFs/s':>12} {'Mem (MB)':>10}")
    print("-" * 55)

    for nx in resolutions:
        try:
            result = benchmark_rhs(nx, device)
            print(
                f"{result['nx']:>6} {result['dofs']:>10} "
                f"{result['time_per_rhs_ms']:>10.2f} "
                f"{result['dofs_per_sec']:>12.2e} "
                f"{result['mem_mb']:>10.1f}"
            )
        except RuntimeError as e:
            print(f"{nx:>6}  FAILED: {e}")
            break


if __name__ == "__main__":
    main()
