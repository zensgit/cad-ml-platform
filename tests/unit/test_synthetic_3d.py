"""Tests for the synthetic 3D point cloud generator."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from scripts.generate_3d_training_data import Synthetic3DGenerator


@pytest.fixture
def gen() -> Synthetic3DGenerator:
    return Synthetic3DGenerator(num_points=2048, seed=0)


# -- shape tests --------------------------------------------------------------


def test_generate_flange_shape(gen: Synthetic3DGenerator) -> None:
    pc = gen.generate_flange()
    assert pc.shape == (2048, 3)


def test_generate_shaft_shape(gen: Synthetic3DGenerator) -> None:
    pc = gen.generate_shaft()
    assert pc.shape == (2048, 3)


# -- validity tests -----------------------------------------------------------


def test_all_generators_produce_valid_points(gen: Synthetic3DGenerator) -> None:
    """Every generator must return finite points within reasonable bounds."""
    for name in gen._GENERATORS:
        pc = getattr(gen, name)()
        assert pc.shape == (2048, 3), f"{name}: wrong shape {pc.shape}"
        assert not np.any(np.isnan(pc)), f"{name}: contains NaN"
        assert not np.any(np.isinf(pc)), f"{name}: contains Inf"
        # After normalization, all points should be within unit sphere (+ small eps)
        radii = np.linalg.norm(pc, axis=1)
        assert np.all(radii <= 1.0 + 1e-6), (
            f"{name}: points outside unit sphere, max radius={radii.max():.6f}"
        )


# -- dataset tests ------------------------------------------------------------


def test_generate_dataset_balanced() -> None:
    gen = Synthetic3DGenerator(num_points=128, seed=7)
    points, labels, names = gen.generate_dataset(samples_per_class=5)

    assert points.shape == (40, 128, 3)  # 8 classes * 5 samples
    assert labels.shape == (40,)
    assert len(names) == 8

    # Each class should have exactly 5 samples
    for cls in range(8):
        assert np.sum(labels == cls) == 5


# -- normalization test -------------------------------------------------------


def test_normalize_unit_sphere(gen: Synthetic3DGenerator) -> None:
    pc = gen.generate_flange()
    radii = np.linalg.norm(pc, axis=1)
    assert radii.max() <= 1.0 + 1e-6


# -- save / load tests --------------------------------------------------------


def test_save_dataset_creates_files() -> None:
    gen = Synthetic3DGenerator(num_points=64, seed=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        gen.save_dataset(tmpdir, samples_per_class=2)

        assert os.path.isfile(os.path.join(tmpdir, "train.npz"))
        assert os.path.isfile(os.path.join(tmpdir, "val.npz"))
        assert os.path.isfile(os.path.join(tmpdir, "test.npz"))
        assert os.path.isfile(os.path.join(tmpdir, "manifest.json"))


def test_manifest_has_metadata() -> None:
    gen = Synthetic3DGenerator(num_points=64, seed=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        gen.save_dataset(tmpdir, samples_per_class=2)
        with open(os.path.join(tmpdir, "manifest.json")) as f:
            manifest = json.load(f)

        assert "class_names" in manifest
        assert len(manifest["class_names"]) == 8
        assert "split_sizes" in manifest
        assert "total_samples" in manifest
        assert manifest["total_samples"] == 16  # 8 * 2
