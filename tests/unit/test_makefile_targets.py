"""Verify that Makefile contains the new module testing targets."""

import pathlib

MAKEFILE = pathlib.Path(__file__).resolve().parents[2] / "Makefile"


def _read_makefile() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def test_makefile_has_new_targets():
    """Makefile must declare the test-new-modules target."""
    content = _read_makefile()
    assert "test-new-modules:" in content, (
        "Makefile is missing the 'test-new-modules' target"
    )


def test_makefile_has_smoke_target():
    """Makefile must declare the smoke-new-modules target."""
    content = _read_makefile()
    assert "smoke-new-modules:" in content, (
        "Makefile is missing the 'smoke-new-modules' target"
    )


def test_makefile_has_category_targets():
    """Makefile must declare the per-category convenience targets."""
    content = _read_makefile()
    expected = [
        "test-cost:",
        "test-ai-intelligence:",
        "test-diff:",
        "test-pointcloud:",
        "test-knowledge:",
        "test-embeddings:",
        "test-copilot:",
        "test-training-scripts:",
    ]
    for target in expected:
        assert target in content, f"Makefile is missing the '{target}' target"
