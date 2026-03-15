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


def test_comment_soft_mode_smoke_pr_js_creates_comment_for_valid_pr(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "soft_smoke_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "overall_exit_code": 0,
                "soft_marker_ok": True,
                "restore_ok": True,
                "dispatch": {
                    "run_id": 123456,
                    "run_url": "https://example.invalid/run/123456",
                },
                "attempts": [{"attempt": 1, "dispatch_exit_code": 0, "soft_marker_ok": True}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    node_script = r"""
const mod = require("./scripts/ci/comment_soft_mode_smoke_pr.js");
const summaryPath = process.argv[1];
process.env.SOFT_SMOKE_SUMMARY_JSON = summaryPath;
process.env.SOFT_SMOKE_TRIGGER_PR = "321";

let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 1 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called for empty comment list");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 321 },
  sha: "abcdef1234567890",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentSoftModeSmokePR({ github, context, process: mockProcess });
  if (!createdPayload) {
    throw new Error("createComment was not called");
  }
  if (!String(createdPayload.body || "").includes("CAD ML Platform - Soft Mode Smoke")) {
    throw new Error("comment body missing soft mode smoke heading");
  }
  console.log("ok:create-comment");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script, str(summary_path))
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create-comment" in result.stdout


def test_comment_soft_mode_smoke_pr_js_missing_summary_file_skips_safely(
    tmp_path: Path,
) -> None:
    missing_summary = tmp_path / "not_found_summary.json"

    node_script = r"""
const mod = require("./scripts/ci/comment_soft_mode_smoke_pr.js");
const summaryPath = process.argv[1];
process.env.SOFT_SMOKE_SUMMARY_JSON = summaryPath;
process.env.SOFT_SMOKE_TRIGGER_PR = "654";

let createCalled = false;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async () => {
        createCalled = true;
        return { data: { id: 1 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 654 },
  sha: "fedcba9876543210",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentSoftModeSmokePR({ github, context, process: mockProcess });
  if (createCalled) {
    throw new Error("createComment should not be called when summary file is missing");
  }
  console.log("ok:skip-missing-summary");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script, str(missing_summary))
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:skip-missing-summary" in result.stdout
