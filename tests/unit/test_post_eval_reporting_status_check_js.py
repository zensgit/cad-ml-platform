from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_BIN = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE_BIN is None, reason="node is not available")


def _run_node_inline(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [NODE_BIN, "-e", script],  # type: ignore[list-item]
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_map_readiness_to_state_ready() -> None:
    script = r"""
const mod = require("./scripts/ci/post_eval_reporting_status_check.js");
const state = mod.mapReadinessToState("ready");
if (state !== "success") throw new Error("expected success, got " + state);
console.log("ok:ready-success");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:ready-success" in result.stdout


def test_map_readiness_to_state_degraded() -> None:
    script = r"""
const mod = require("./scripts/ci/post_eval_reporting_status_check.js");
const state = mod.mapReadinessToState("degraded");
if (state !== "success") throw new Error("expected success, got " + state);
console.log("ok:degraded-success");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:degraded-success" in result.stdout


def test_map_readiness_to_state_unavailable() -> None:
    script = r"""
const mod = require("./scripts/ci/post_eval_reporting_status_check.js");
const state = mod.mapReadinessToState("unavailable");
if (state !== "failure") throw new Error("expected failure, got " + state);
console.log("ok:unavailable-failure");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:unavailable-failure" in result.stdout


def test_map_readiness_to_description_degraded() -> None:
    script = r"""
const mod = require("./scripts/ci/post_eval_reporting_status_check.js");
const desc = mod.mapReadinessToDescription("degraded", { missing_count: 1, stale_count: 2, mismatch_count: 0 });
if (!desc.includes("Degraded")) throw new Error("expected Degraded in desc: " + desc);
if (!desc.includes("missing=1")) throw new Error("expected missing=1 in desc: " + desc);
if (!desc.includes("stale=2")) throw new Error("expected stale=2 in desc: " + desc);
console.log("ok:degraded-desc");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:degraded-desc" in result.stdout


def test_load_release_summary_from_file(tmp_path: Path) -> None:
    summary_path = tmp_path / "release_summary.json"
    summary_path.write_text(
        json.dumps({"release_readiness": "ready", "status": "ready"}),
        encoding="utf-8",
    )

    script = f"""
const mod = require("./scripts/ci/post_eval_reporting_status_check.js");
const summary = mod.loadReleaseSummary("{summary_path}");
if (!summary) throw new Error("expected non-null summary");
if (summary.release_readiness !== "ready") throw new Error("expected ready, got " + summary.release_readiness);
console.log("ok:load-summary");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:load-summary" in result.stdout


def test_load_release_summary_returns_null_for_missing() -> None:
    script = r"""
const mod = require("./scripts/ci/post_eval_reporting_status_check.js");
const summary = mod.loadReleaseSummary("/tmp/nonexistent_release_summary_12345.json");
if (summary !== null) throw new Error("expected null for missing file");
console.log("ok:null-for-missing");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:null-for-missing" in result.stdout


def test_status_check_module_does_not_own_content_logic() -> None:
    """The JS module must not define summary/metrics/trend/render functions."""
    script_content = (REPO_ROOT / "scripts" / "ci" / "post_eval_reporting_status_check.js").read_text(
        encoding="utf-8"
    )
    forbidden = [
        "buildEvaluationReportCommentBody",
        "generateHtml",
        "buildWeekly",
        "plotTrend",
        "materializeBundle",
    ]
    for name in forbidden:
        assert name not in script_content, (
            f"Status check module must not own content logic, but references {name}"
        )
