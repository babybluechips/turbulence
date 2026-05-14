"""
I/O utilities: HDF5 checkpointing, statistics logging, field output.

Supports saving and loading full simulation state (conservative fields +
metadata) for restart capability, and streaming statistics to CSV/HDF5.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch

from .config import SimConfig
from .diagnostics import FlowStatistics
from .fields import ConservativeFields


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ---- Checkpointing ----

def save_checkpoint(
    cons: ConservativeFields,
    config: SimConfig,
    step: int,
    time: float,
    path: str,
) -> None:
    """Save simulation state to file.

    Saves conservative variables and metadata using torch.save
    (or HDF5 if h5py is available and configured).
    """
    ensure_dir(os.path.dirname(path) or ".")

    state = {
        "rho": cons.rho.cpu(),
        "rhou": cons.rhou.cpu(),
        "rhov": cons.rhov.cpu(),
        "rhow": cons.rhow.cpu(),
        "E": cons.E.cpu(),
        "step": step,
        "time": time,
        "nx": config.grid.nx,
        "ny": config.grid.ny,
        "nz": config.grid.nz,
        "gamma": config.physics.gamma,
        "reynolds": config.physics.reynolds,
        "mach": config.physics.mach,
    }

    if config.io.save_format == "hdf5":
        try:
            _save_hdf5(state, path)
        except ImportError:
            torch.save(state, path.replace(".h5", ".pt").replace(".hdf5", ".pt"))
    else:
        torch.save(state, path)


def load_checkpoint(
    path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[ConservativeFields, int, float]:
    """Load simulation state from file.

    Returns (conservative_fields, step, time).
    """
    if path.endswith(".h5") or path.endswith(".hdf5"):
        state = _load_hdf5(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)

    cons = ConservativeFields(
        rho=state["rho"].to(device=device, dtype=dtype),
        rhou=state["rhou"].to(device=device, dtype=dtype),
        rhov=state["rhov"].to(device=device, dtype=dtype),
        rhow=state["rhow"].to(device=device, dtype=dtype),
        E=state["E"].to(device=device, dtype=dtype),
    )
    return cons, int(state["step"]), float(state["time"])


def _save_hdf5(state: dict, path: str) -> None:
    """Save state dict to HDF5 file."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 output. Install with: pip install h5py")

    if not path.endswith((".h5", ".hdf5")):
        path = path + ".h5"

    with h5py.File(path, "w") as f:
        for key, val in state.items():
            if isinstance(val, torch.Tensor):
                f.create_dataset(key, data=val.numpy(), compression="gzip")
            else:
                f.attrs[key] = val


def _load_hdf5(path: str) -> dict:
    """Load state dict from HDF5 file."""
    import h5py

    state = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            state[key] = torch.from_numpy(np.array(f[key]))
        for key, val in f.attrs.items():
            state[key] = val
    return state


# ---- Statistics logging ----

class StatsLogger:
    """CSV logger for time-series flow statistics."""

    def __init__(self, path: str) -> None:
        self.path = path
        ensure_dir(os.path.dirname(path) or ".")
        self._header_written = False
        self._file = None
        self._writer = None

    def log(self, stats: FlowStatistics) -> None:
        """Append one row of statistics."""
        data = stats.as_dict()

        if not self._header_written:
            self._file = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=data.keys())
            self._writer.writeheader()
            self._header_written = True

        self._writer.writerow({k: f"{v:.8e}" if isinstance(v, float) else v
                                for k, v in data.items()})
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
