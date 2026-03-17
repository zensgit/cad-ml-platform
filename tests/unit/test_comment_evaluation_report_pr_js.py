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


def test_comment_evaluation_report_pr_js_creates_comment_with_workflow_inventory(
    tmp_path: Path,
) -> None:
    ci_watch_summary = tmp_path / "ci_watch_summary.json"
    ci_watch_summary.write_text(
        json.dumps(
            {
                "reason": "passed",
                "counts": {"failed": 0},
                "failure_details": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    workflow_file_health = tmp_path / "workflow_file_health.json"
    workflow_file_health.write_text(
        json.dumps(
            {
                "count": 33,
                "failed_count": 0,
                "mode_used": "auto",
                "fallback_reason": "none",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    workflow_inventory = tmp_path / "workflow_inventory.json"
    workflow_inventory.write_text(
        json.dumps(
            {
                "workflow_count": 33,
                "duplicate_name_count": 0,
                "missing_required_count": 0,
                "non_unique_required_count": 0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");
const ciWatchPath = process.argv[1];
const workflowFileHealthPath = process.argv[2];
const workflowInventoryPath = process.argv[3];

process.env.EVAL_COMBINED_SCORE = "0.920";
process.env.EVAL_VISION_SCORE = "0.910";
process.env.EVAL_OCR_SCORE = "0.930";
process.env.EVAL_MIN_COMBINED = "0.800";
process.env.EVAL_MIN_VISION = "0.650";
process.env.EVAL_MIN_OCR = "0.900";
process.env.SECURITY_STATUS = "pass";
process.env.EVALUATION_STRICT_FAIL_MODE = "soft";
process.env.EVALUATION_STRICT_FAIL_MODE_RESOLVED = "soft";
process.env.EVALUATION_STRICT_FAIL_MODE_RAW = "soft";
process.env.CI_WATCH_SUMMARY_JSON_FOR_COMMENT = ciWatchPath;
process.env.WORKFLOW_FILE_HEALTH_SUMMARY_JSON_FOR_COMMENT = workflowFileHealthPath;
process.env.WORKFLOW_INVENTORY_REPORT_JSON_FOR_COMMENT = workflowInventoryPath;

let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 42 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called when no bot comment exists");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 123 },
  sha: "abcdef1234567890",
  runId: "111222333",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentEvaluationReportPR({ github, context, process: mockProcess });
  if (!createdPayload) {
    throw new Error("createComment was not called");
  }
  const body = String(createdPayload.body || "");
  if (!body.includes("CAD ML Platform - Evaluation Results")) {
    throw new Error("comment body missing evaluation heading");
  }
  if (!body.includes("Workflow Inventory Audit")) {
    throw new Error("comment body missing workflow inventory section");
  }
  if (!body.includes("workflows=33, duplicate=0, missing_required=0, non_unique_required=0")) {
    throw new Error("comment body missing workflow inventory summary");
  }
  if (!body.includes("failed=0/33, mode=auto, fallback=none")) {
    throw new Error("comment body missing workflow file health summary");
  }
  console.log("ok:create-comment-with-workflow-inventory");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(
        node_script,
        str(ci_watch_summary),
        str(workflow_file_health),
        str(workflow_inventory),
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create-comment-with-workflow-inventory" in result.stdout


def test_comment_evaluation_report_pr_js_updates_comment_when_inventory_missing(
    tmp_path: Path,
) -> None:
    missing_workflow_inventory = tmp_path / "workflow_inventory_missing.json"

    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");
const workflowInventoryPath = process.argv[1];

process.env.EVAL_COMBINED_SCORE = "0.810";
process.env.EVAL_VISION_SCORE = "0.820";
process.env.EVAL_OCR_SCORE = "0.910";
process.env.EVAL_MIN_COMBINED = "0.800";
process.env.EVAL_MIN_VISION = "0.650";
process.env.EVAL_MIN_OCR = "0.900";
process.env.WORKFLOW_INVENTORY_REPORT_JSON_FOR_COMMENT = workflowInventoryPath;

let createdPayload = null;
let updatedPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({
        data: [
          {
            id: 77,
            user: { type: "Bot" },
            body: "## 📊 CAD ML Platform - Evaluation Results\nold",
          },
        ],
      }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 43 } };
      },
      updateComment: async (payload) => {
        updatedPayload = payload;
        return { data: { id: 77 } };
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 456 },
  sha: "fedcba9876543210",
  runId: "444555666",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentEvaluationReportPR({ github, context, process: mockProcess });
  if (createdPayload) {
    throw new Error("createComment should not be called when bot comment exists");
  }
  if (!updatedPayload) {
    throw new Error("updateComment was not called");
  }
  const body = String(updatedPayload.body || "");
  if (!body.includes("Workflow Inventory Audit")) {
    throw new Error("updated comment missing workflow inventory section");
  }
  if (!body.includes(`summary missing at ${workflowInventoryPath}`)) {
    throw new Error("updated comment missing workflow inventory warning");
  }
  console.log("ok:update-comment-with-missing-workflow-inventory");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script, str(missing_workflow_inventory))
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:update-comment-with-missing-workflow-inventory" in result.stdout
