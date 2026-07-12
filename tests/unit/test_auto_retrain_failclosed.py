"""L3: prove `scripts/auto_retrain.sh` is fail-closed at the evaluation-integrity gate.

The security-critical property is now stronger than the earlier draft's: the gate has
**no pass path**, so `auto_retrain.sh` must exit NON-ZERO **before any manifest write,
preprocessing, training, or model-file touch** — under *every* environment, including the
env var and artifact path the previous (bypassable) draft honoured.

Hermetic: runs in a temp cwd containing only the two scripts under test, so the fail path
needs no ML stack and no side-effect can touch the real tree.

See PRODUCT_STRATEGY.md §5.2 (evaluation integrity is not release-grade), §8.1 (Track E).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTO_RETRAIN = REPO_ROOT / "scripts" / "auto_retrain.sh"
GATE = REPO_ROOT / "scripts" / "eval_integrity_gate.py"


def _sandbox(tmp_path: Path) -> Path:
    """A temp tree holding only the scripts under test — any write is detectable."""
    (tmp_path / "scripts").mkdir(parents=True)
    shutil.copy2(AUTO_RETRAIN, tmp_path / "scripts" / "auto_retrain.sh")
    shutil.copy2(GATE, tmp_path / "scripts" / "eval_integrity_gate.py")
    os.chmod(tmp_path / "scripts" / "auto_retrain.sh", 0o755)
    return tmp_path


def _run(cwd: Path, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(cwd)}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", "scripts/auto_retrain.sh"],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        timeout=120,
    )


def _mutations(root: Path) -> set[str]:
    """Anything created beyond the two scripts we seeded is a side-effect."""
    seeded = {"scripts", "scripts/auto_retrain.sh", "scripts/eval_integrity_gate.py"}
    found = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if not str(p.relative_to(root)).startswith(".")
    }
    return found - seeded


def test_blocks_and_mutates_nothing(tmp_path):
    root = _sandbox(tmp_path)
    proc = _run(root)

    assert proc.returncode != 0, "auto_retrain must be fail-closed"
    assert "Step 0" in proc.stdout
    assert "BLOCKED" in proc.stdout
    # Never reached the first mutating step.
    assert "Step 1" not in proc.stdout
    assert "Step 2" not in proc.stdout
    assert _mutations(root) == set(), "the gate must abort before ANY side-effect"


@pytest.mark.parametrize(
    "env_extra",
    [
        {"EVAL_INTEGRITY_ARTIFACT": "/tmp/anything.json"},
        {"EVAL_INTEGRITY_SKIP": "1"},
        {"SKIP_EVAL_INTEGRITY": "true"},
        {"FORCE_RETRAIN": "1"},
        {"ALLOW_PROMOTION": "yes"},
    ],
)
def test_no_env_toggle_opens_the_path(tmp_path, env_extra):
    """The previous draft honoured EVAL_INTEGRITY_ARTIFACT. Nothing may open it now."""
    root = _sandbox(tmp_path)
    proc = _run(root, env_extra)
    assert proc.returncode != 0
    assert _mutations(root) == set()


def test_shell_still_blocks_even_if_the_gate_itself_is_subverted(tmp_path):
    """Defence in depth: replace the gate with a stub that EXITS 0 — the shell must still stop.

    The gate has no pass path by construction, so a zero exit can only mean someone edited
    `check()` (or swapped the file). `auto_retrain.sh` must not trust that: it treats a
    successful gate as an invariant breach and refuses to mutate anything anyway.
    """
    root = _sandbox(tmp_path)
    # Subvert the gate: make it succeed.
    (root / "scripts" / "eval_integrity_gate.py").write_text(
        "import sys\nsys.exit(0)\n", encoding="utf-8"
    )

    proc = _run(root)

    assert proc.returncode != 0, "the shell must not trust a gate that returns success"
    assert "invariant breach" in (proc.stdout + proc.stderr).lower()
    # Still never reached the first mutating step, and wrote nothing.
    assert "Step 1" not in proc.stdout
    assert _mutations(root) == set()


def test_even_a_valid_looking_artifact_on_disk_does_not_open_the_path(tmp_path):
    """Place the exact artifact the old gate accepted; the path must still be closed."""
    root = _sandbox(tmp_path)
    art_dir = root / "data" / "eval_integrity"
    art_dir.mkdir(parents=True)
    (art_dir / "evaluation_integrity_v2.json").write_text(
        json.dumps(
            {
                "schema_version": "evaluation-integrity-v2",
                "reproducible": True,
                "holdout": {"fraction": 0.2, "holdout_rows": 180},
                "split_digest": "0" * 64,
                "metrics": {
                    "per_class": {"法兰": {"f1": 0.97}},
                    "macro_f1": 0.93,
                    "calibration_ece": 0.02,
                    "false_duplicate_rate": 0.01,
                    "missed_reuse_rate": 0.03,
                },
            }
        ),
        encoding="utf-8",
    )

    proc = _run(root)
    assert proc.returncode != 0, "a well-formed artifact must NOT open the gate any more"
    assert "BLOCKED" in proc.stdout
    # The artifact we seeded is the only thing under data/; nothing new was written.
    assert _mutations(root) == {
        "data",
        "data/eval_integrity",
        "data/eval_integrity/evaluation_integrity_v2.json",
    }
