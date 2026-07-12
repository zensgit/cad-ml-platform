"""Unit tests for the unconditional fail-closed evaluation-integrity gate (L3).

The property under test is NOT "the gate fails on bad input". It is stronger:

    **the gate has no success path at all.**

An earlier draft accepted a JSON artifact as a "you may proceed" token. That token was
unbound (it named neither the validation manifest actually used nor the candidate model
actually promoted) and it did not even need forging — the sanctioned producer emitted a
*passing* artifact with zero holdout rows and all-zero metrics.

So these tests assert the bypass is *absent*, behaviourally (any argv, any env, even a
perfectly-shaped artifact still blocks) and structurally (no `return 0` / `sys.exit(0)`
reachable from `main`). See PRODUCT_STRATEGY.md §5.2, §8.1.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import eval_integrity_gate as gate

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "scripts" / "eval_integrity_gate.py"


# --------------------------------------------------------------------------------------
# Behavioural: no argv opens it
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--artifact", "/tmp/whatever.json"],
        ["--require-version", "evaluation-integrity-v2"],
        ["--force"],
        ["--allow"],
        ["--skip-gate"],
        ["--artifact", "/tmp/x.json", "--require-version", "evaluation-integrity-v2"],
        ["totally", "unknown", "positional", "args"],
    ],
)
def test_no_argv_opens_the_gate(argv):
    assert gate.main(argv) != 0


# --------------------------------------------------------------------------------------
# Behavioural: no env var opens it
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "var",
    [
        "EVAL_INTEGRITY_ARTIFACT",
        "EVAL_INTEGRITY_SKIP",
        "SKIP_EVAL_INTEGRITY",
        "FORCE_RETRAIN",
        "ALLOW_PROMOTION",
        "CI",
        "DEBUG",
    ],
)
def test_no_env_var_opens_the_gate(monkeypatch, var):
    monkeypatch.setenv(var, "1")
    assert gate.main([]) != 0
    monkeypatch.setenv(var, "true")
    assert gate.main([]) != 0


# --------------------------------------------------------------------------------------
# Behavioural: even a perfectly-shaped artifact does not open it (the old bypass is gone)
# --------------------------------------------------------------------------------------
def test_a_perfectly_valid_looking_artifact_still_blocks(tmp_path):
    """The exact shape the previous gate accepted must now be inert."""
    artifact = {
        "schema_version": "evaluation-integrity-v2",
        "reproducible": True,
        "holdout": {"type": "content-hash+family", "fraction": 0.2, "holdout_rows": 180},
        "split_digest": "0" * 64,
        "metrics": {
            "per_class": {"法兰": {"f1": 0.97}},
            "macro_f1": 0.93,
            "calibration_ece": 0.02,
            "false_duplicate_rate": 0.01,
            "missed_reuse_rate": 0.03,
        },
    }
    path = tmp_path / "evaluation_integrity_v2.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    assert gate.main(["--artifact", str(path)]) != 0


# --------------------------------------------------------------------------------------
# Behavioural: check() always raises
# --------------------------------------------------------------------------------------
def test_check_always_raises():
    with pytest.raises(gate.GateBlocked):
        gate.check()


# --------------------------------------------------------------------------------------
# Structural: there is NO success path reachable from main()
# --------------------------------------------------------------------------------------
def test_main_has_no_zero_return_path():
    """Assert the *absence of a bypass*, not merely that bad input fails.

    Any `return 0` or `sys.exit(0)` inside `main` would be a pass path. There must be none.
    """
    tree = ast.parse(GATE_PATH.read_text(encoding="utf-8"), filename=str(GATE_PATH))
    main_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )

    zero_returns = [
        n
        for n in ast.walk(main_fn)
        if isinstance(n, ast.Return)
        and isinstance(n.value, ast.Constant)
        and n.value.value == 0
    ]
    assert not zero_returns, "main() must have no `return 0` — that would be a bypass"

    zero_exits = [
        n
        for n in ast.walk(main_fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "exit"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == 0
    ]
    assert not zero_exits, "main() must have no `sys.exit(0)` — that would be a bypass"

    # Every return in main must be a non-zero constant.
    for node in ast.walk(main_fn):
        if isinstance(node, ast.Return):
            assert isinstance(node.value, ast.Constant) and node.value.value != 0, (
                "every return from main() must be a non-zero constant"
            )


def test_check_body_is_an_unconditional_raise():
    """`check()` must not acquire a conditional that could let it return."""
    tree = ast.parse(GATE_PATH.read_text(encoding="utf-8"), filename=str(GATE_PATH))
    check_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check"
    )
    # Strip the docstring, then the body must be exactly one `raise`.
    body = [n for n in check_fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert len(body) == 1 and isinstance(body[0], ast.Raise), (
        "check() must be an unconditional raise; a branch here would be a bypass"
    )


# --------------------------------------------------------------------------------------
# End-to-end: invoking the script as a subprocess exits non-zero and says why
# --------------------------------------------------------------------------------------
def test_subprocess_exits_nonzero_with_strategy_pointer():
    proc = subprocess.run(
        [sys.executable, str(GATE_PATH), "--artifact", "/nonexistent.json"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert proc.returncode != 0
    assert "fail-closed" in proc.stderr
    assert "PRODUCT_STRATEGY.md" in proc.stderr
    assert "no bypass" in proc.stderr
