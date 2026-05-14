"""Tests for configuration management."""

from __future__ import annotations

import os
import tempfile

import pytest

from ns_turbulence.config import GridConfig, PhysicsConfig, SimConfig


class TestGridConfig:
    def test_defaults(self):
        g = GridConfig()
        assert g.shape == (64, 64, 64)

    def test_validate_power_of_two(self):
        g = GridConfig(nx=48)
        with pytest.raises(ValueError, match="power of 2"):
            g.validate()

    def test_validate_too_small(self):
        g = GridConfig(nx=2)
        with pytest.raises(ValueError, match="too small"):
            g.validate()


class TestPhysicsConfig:
    def test_validate_gamma(self):
        p = PhysicsConfig(gamma=0.5)
        with pytest.raises(ValueError, match="gamma"):
            p.validate()

    def test_validate_reynolds(self):
        p = PhysicsConfig(reynolds=-1.0)
        with pytest.raises(ValueError, match="Reynolds"):
            p.validate()


class TestSimConfig:
    def test_default_creation(self):
        config = SimConfig()
        config.validate()

    def test_yaml_roundtrip(self):
        config = SimConfig()
        config.physics.reynolds = 3200.0
        config.grid.nx = 128

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            path = f.name

        try:
            config.to_yaml(path)
            loaded = SimConfig.from_yaml(path)
            assert loaded.physics.reynolds == 3200.0
            assert loaded.grid.nx == 128
        finally:
            os.unlink(path)

    def test_device_auto(self):
        config = SimConfig(device="auto")
        dev = config.get_device()
        assert dev.type in ("cpu", "cuda")

    def test_dtype_mapping(self):
        import torch
        config = SimConfig(dtype="float64")
        assert config.get_dtype() == torch.float64
        config.dtype = "float32"
        assert config.get_dtype() == torch.float32
