from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_BIN = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE_BIN is None, reason="node is not available")


def _run_node_inline(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [NODE_BIN, "-e", script, *args],  # type: ignore[list-item]
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_soft_mode_comment_body_matches_between_js_and_python(tmp_path: Path) -> None:
    from scripts.ci import post_soft_mode_smoke_pr_comment as py_mod

    summary = {
        "overall_exit_code": 0,
        "dispatch_exit_code": 0,
        "soft_marker_ok": True,
        "restore_ok": True,
        "dispatch": {
            "run_id": 123456,
            "run_url": "https://example.invalid/run/123456",
        },
        "attempts": [
            {
                "attempt": 1,
                "dispatch_exit_code": 0,
                "soft_marker_ok": True,
                "soft_marker_message": "marker restored",
            }
        ],
    }
    summary_path = tmp_path / "soft_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")

    node_script = r"""
const fs = require("fs");
const mod = require("./scripts/ci/comment_soft_mode_smoke_pr.js");
const summaryPath = process.argv[1];
const summary = JSON.parse(fs.readFileSync(summaryPath, "utf8"));
const body = mod.buildSoftModeSmokeCommentBody({
  summary,
  title: "CAD ML Platform - Soft Mode Smoke",
  commitSha: "abcdef123456",
  updatedAt: "2026-03-17 10:00:00",
});
process.stdout.write(body);
"""

    node_result = _run_node_inline(node_script, str(summary_path))
    assert node_result.returncode == 0, node_result.stderr or node_result.stdout

    py_body = py_mod.build_comment_body(
        summary=summary,
        title="CAD ML Platform - Soft Mode Smoke",
        commit_sha="abcdef123456",
        updated_at="2026-03-17 10:00:00",
    )

    assert node_result.stdout == py_body
