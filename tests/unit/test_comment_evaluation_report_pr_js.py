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


def test_comment_evaluation_report_pr_js_builds_body_from_view_model() -> None:
    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");

const body = mod.buildEvaluationReportCommentBody({
  overallStatus: "✅ **All checks passed!**",
  combined: 0.92,
  minCombined: 0.8,
  combinedStatus: "✅ Pass",
  vision: 0.91,
  minVision: 0.65,
  visionStatus: "✅ Pass",
  ocr: 0.93,
  minOcr: 0.9,
  ocrStatus: "✅ Pass",
  hasAnomalies: false,
  securityStatus: "pass",
  reviewPackStatus: "⏭️ skipped",
  reviewPackInsights: "⏭️ skipped",
  reviewGateStatus: "passed (exit=0, headline=n/a)",
  reviewGateStrictStatus: "strict=false, should_fail=false, reason=n/a",
  trainSweepStatus: "⏭️ skipped",
  hybridBlindEvalStatus: "ok (source=real, exit=0, coverage=1.0, hybrid_acc=0.81, graph2d_acc=0.73, gain=0.08)",
  hybridBlindGateStatus: "passed (exit=0, headline=n/a)",
  hybridBlindStrictStatus: "strict=true, require_real=true, should_fail=false, reason=n/a",
  hybridBlindDriftStatus: "ok (exit=0, headline=stable)",
  blindGainSummary: "0.0800",
  hybridCalibrationStatus: "ok (exit=0, n_samples=32, ece=0.02, brier=0.11, mce=0.04)",
  hybridCalibrationGateStatus: "passed (exit=0, headline=n/a)",
  hybridCalibrationStrictStatus: "strict=false, should_fail=false, reason=n/a",
  hybridSuperpassStrictStatus: "strict=false, should_fail=false, reason=n/a",
  hybridSuperpassValidationStrictStatus: "strict=false, exit=0, status=ok",
  hybridCalibrationBaselineStatus: "ok (exit=0, path=models/calibration.json)",
  evaluationStrictMode: "soft",
  evaluationStrictModeRawValue: "soft",
  strictDecisionResult: "downgraded_to_warning",
  strictPlaybookSummary: "[strict-gate-playbook](https://example.invalid/playbook)",
  ciWatchFailureSummary: "failed=0, reason=passed",
  ciWatchValidationReportSummary: "verdict=PASS, reason=all_workflows_success, failed=0, missing_required=0, workflow_guardrail=ok, ci_workflow_overview=ok",
  workflowFileHealthSummary: "failed=0/33, mode=auto, fallback=none",
  workflowInventorySummary: "workflows=33, duplicate=0, missing_required=0, non_unique_required=0",
  workflowPublishHelperSummary: "checked=33, failed=0, raw=0, missing_comment_helper=0, missing_issue_helper=0",
  workflowGuardrailSummary: "status=ok, workflow_health=ok, inventory=ok, publish_helper=ok",
  ciWorkflowGuardrailOverviewSummary: "status=ok, ci_watch=ok, workflow_guardrail=ok",
  reviewPackLight: "⚪",
  reviewGateLight: "🟢",
  trainSweepLight: "⚪",
  hybridBlindLight: "🟢",
  hybridCalibrationLight: "🟢",
  strictDecisionLight: "🟡",
  strictFailureRequestsCount: 2,
  ciWatchFailureLight: "🟢",
  ciWatchValidationReportLight: "🟢",
  workflowFileHealthLight: "🟢",
  workflowInventoryLight: "🟢",
  workflowPublishHelperLight: "🟢",
  workflowGuardrailLight: "🟢",
  ciWorkflowGuardrailOverviewLight: "🟢",
  evaluationStrictModeResolvedRaw: "soft",
  strictFailureRequestSummary: "hybrid_blind:gate_failed_under_strict_mode",
  strictActionItems: ["- ⚠️ soft gate downgraded"],
  strictActionChecklist: "- ⚠️ soft gate downgraded\n- 📚 follow playbook",
  runUrl: "https://github.com/zensgit/cad-ml-platform/actions/runs/111222333",
  updatedAt: "2026-03-17 12:34:56",
  commitSha: "abcdef1234567890",
});

if (!body.includes("CAD ML Platform - Evaluation Results")) {
  throw new Error("body missing heading");
}
if (!body.includes("### Scores")) {
  throw new Error("body missing scores section");
}
if (!body.includes("| **Combined** | 0.920 | 0.8 | ✅ Pass |")) {
  throw new Error("body missing combined score row");
}
if (!body.includes("Workflow Inventory Audit")) {
  throw new Error("body missing workflow inventory row");
}
if (!body.includes("Workflow Publish Helper Adoption")) {
  throw new Error("body missing workflow publish helper row");
}
if (!body.includes("Workflow Guardrail Summary")) {
  throw new Error("body missing workflow guardrail row");
}
if (!body.includes("CI Watch Validation Report")) {
  throw new Error("body missing ci watch validation row");
}
if (!body.includes("CI Workflow Guardrail Overview")) {
  throw new Error("body missing ci workflow guardrail overview row");
}
if (!body.includes("**CI Watch Validation**")) {
  throw new Error("body missing ci watch validation signal");
}
if (!body.includes("strict_requests=2")) {
  throw new Error("body missing strict request count detail");
}
if (!body.includes("*Updated: 2026-03-17 12:34:56 UTC*")) {
  throw new Error("body missing footer timestamp");
}
if (!body.includes("*Commit: abcdef1*")) {
  throw new Error("body missing footer sha");
}
console.log("ok:build-evaluation-report-comment-body");
"""

    result = _run_node_inline(node_script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:build-evaluation-report-comment-body" in result.stdout


def test_comment_evaluation_report_pr_js_runtime_body_matches_builder_output() -> None:
    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");

const fixedIso = "2026-03-17T12:34:56Z";
const RealDate = Date;
class FakeDate extends RealDate {
  constructor(...args) {
    if (args.length > 0) {
      super(...args);
      return;
    }
    super(fixedIso);
  }
  static now() {
    return new RealDate(fixedIso).getTime();
  }
  static parse(value) {
    return RealDate.parse(value);
  }
  static UTC(...args) {
    return RealDate.UTC(...args);
  }
}
global.Date = FakeDate;

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

  const strictPlaybookSummary = "[strict-gate-playbook](https://github.com/zensgit/cad-ml-platform/blob/abcdef1234567890/docs/STRICT_GATE_PLAYBOOK.md)";
  const expectedBody = mod.buildEvaluationReportCommentBody({
    overallStatus: "✅ **All checks passed!**",
    combined: 0.92,
    minCombined: 0.8,
    combinedStatus: "✅ Pass",
    vision: 0.91,
    minVision: 0.65,
    visionStatus: "✅ Pass",
    ocr: 0.93,
    minOcr: 0.9,
    ocrStatus: "✅ Pass",
    hasAnomalies: false,
    securityStatus: "pass",
    reviewPackStatus: "⏭️ skipped",
    reviewPackInsights: "⏭️ skipped",
    reviewGateStatus: "⏭️ skipped",
    reviewGateStrictStatus: "⏭️ skipped",
    trainSweepStatus: "⏭️ skipped",
    hybridBlindEvalStatus: "⏭️ skipped",
    hybridBlindGateStatus: "⏭️ skipped",
    hybridBlindStrictStatus: "⏭️ skipped",
    hybridBlindDriftStatus: "⏭️ skipped",
    blindGainSummary: "n/a",
    hybridCalibrationStatus: "⏭️ skipped",
    hybridCalibrationGateStatus: "⏭️ skipped",
    hybridCalibrationStrictStatus: "⏭️ skipped",
    hybridSuperpassStrictStatus: "strict=false, should_fail=false, reason=n/a",
    hybridSuperpassValidationStrictStatus: "strict=false, exit=0, status=unknown",
    hybridCalibrationBaselineStatus: "⏭️ skipped",
    evaluationStrictMode: "soft",
    evaluationStrictModeRawValue: "soft",
    strictDecisionResult: "no_strict_fail_requests",
    strictPlaybookSummary,
    ciWatchFailureSummary: "⏭️ skipped (no summary path)",
    ciWatchValidationReportSummary: "⏭️ skipped (no summary path)",
    workflowFileHealthSummary: "⏭️ skipped (no summary path)",
    workflowInventorySummary: "⏭️ skipped (no summary path)",
    workflowPublishHelperSummary: "⏭️ skipped (no summary path)",
    workflowGuardrailSummary: "⏭️ skipped (no summary path)",
    ciWorkflowGuardrailOverviewSummary: "⏭️ skipped (no summary path)",
    reviewPackLight: "⚪",
    reviewGateLight: "⚪",
    trainSweepLight: "⚪",
    hybridBlindLight: "⚪",
    hybridCalibrationLight: "⚪",
    strictDecisionLight: "🟢",
    strictFailureRequestsCount: 0,
    ciWatchFailureLight: "⚪",
    ciWatchValidationReportLight: "⚪",
    workflowFileHealthLight: "⚪",
    workflowInventoryLight: "⚪",
    workflowPublishHelperLight: "⚪",
    workflowGuardrailLight: "⚪",
    ciWorkflowGuardrailOverviewLight: "⚪",
    evaluationStrictModeResolvedRaw: "soft",
    strictFailureRequestSummary: "none",
    strictActionItems: ["- ✅ No strict gate failure request detected."],
    strictActionChecklist: "- ✅ No strict gate failure request detected.",
    runUrl: "https://github.com/zensgit/cad-ml-platform/actions/runs/111222333",
    updatedAt: "2026-03-17 12:34:56",
    commitSha: "abcdef1234567890",
  });

  if (String(createdPayload.body || "") !== expectedBody) {
    throw new Error("runtime comment body drifted from builder output");
  }
  console.log("ok:runtime-body-matches-builder");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:runtime-body-matches-builder" in result.stdout


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


def test_comment_evaluation_report_pr_js_includes_inventory_problem_names(
    tmp_path: Path,
) -> None:
    workflow_inventory = tmp_path / "workflow_inventory_problematic.json"
    workflow_inventory.write_text(
        json.dumps(
            {
                "workflow_count": 35,
                "duplicate_name_count": 2,
                "missing_required_count": 1,
                "non_unique_required_count": 1,
                "duplicates": [
                    {"name": "Security Audit", "files": ["security-audit.yml", "security-copy.yml"]},
                    {"name": "CI", "files": ["ci.yml", "ci-copy.yml"]},
                ],
                "required_workflow_mapping": [
                    {"name": "Evaluation Report", "status": "ok", "files": ["evaluation-report.yml"]},
                    {"name": "Code Quality", "status": "missing", "files": []},
                    {"name": "Security Audit", "status": "non_unique", "files": ["security-audit.yml", "security-copy.yml"]},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");
const workflowInventoryPath = process.argv[1];

process.env.EVAL_COMBINED_SCORE = "0.830";
process.env.EVAL_VISION_SCORE = "0.840";
process.env.EVAL_OCR_SCORE = "0.930";
process.env.EVAL_MIN_COMBINED = "0.800";
process.env.EVAL_MIN_VISION = "0.650";
process.env.EVAL_MIN_OCR = "0.900";
process.env.WORKFLOW_INVENTORY_REPORT_JSON_FOR_COMMENT = workflowInventoryPath;

let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 99 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called for empty comment list");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 789 },
  sha: "0123456789abcdef",
  runId: "777888999",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentEvaluationReportPR({ github, context, process: mockProcess });
  if (!createdPayload) {
    throw new Error("createComment was not called");
  }
  const body = String(createdPayload.body || "");
  if (!body.includes("duplicate_names=Security Audit/CI")) {
    throw new Error("comment body missing duplicate workflow names");
  }
  if (!body.includes("missing_names=Code Quality")) {
    throw new Error("comment body missing missing-required workflow names");
  }
  if (!body.includes("non_unique_names=Security Audit")) {
    throw new Error("comment body missing non-unique workflow names");
  }
  console.log("ok:create-comment-with-inventory-problem-names");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script, str(workflow_inventory))
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create-comment-with-inventory-problem-names" in result.stdout


def test_comment_evaluation_report_pr_js_marks_inventory_parse_errors(
    tmp_path: Path,
) -> None:
    workflow_inventory = tmp_path / "workflow_inventory_invalid.json"
    workflow_inventory.write_text("{invalid json", encoding="utf-8")

    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");
const workflowInventoryPath = process.argv[1];

process.env.EVAL_COMBINED_SCORE = "0.830";
process.env.EVAL_VISION_SCORE = "0.840";
process.env.EVAL_OCR_SCORE = "0.930";
process.env.EVAL_MIN_COMBINED = "0.800";
process.env.EVAL_MIN_VISION = "0.650";
process.env.EVAL_MIN_OCR = "0.900";
process.env.WORKFLOW_INVENTORY_REPORT_JSON_FOR_COMMENT = workflowInventoryPath;

let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 100 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called for empty comment list");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 987 },
  sha: "1122334455667788",
  runId: "101010101",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentEvaluationReportPR({ github, context, process: mockProcess });
  if (!createdPayload) {
    throw new Error("createComment was not called");
  }
  const body = String(createdPayload.body || "");
  if (!body.includes("Workflow Inventory Audit")) {
    throw new Error("comment body missing workflow inventory section");
  }
  if (!body.includes("parse_error")) {
    throw new Error("comment body missing workflow inventory parse_error");
  }
  console.log("ok:create-comment-with-inventory-parse-error");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script, str(workflow_inventory))
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create-comment-with-inventory-parse-error" in result.stdout


def test_comment_evaluation_report_pr_js_includes_publish_helper_summary(
    tmp_path: Path,
) -> None:
    workflow_publish_helper = tmp_path / "workflow_publish_helper.json"
    workflow_publish_helper.write_text(
        json.dumps(
            {
                "checked_count": 33,
                "failed_count": 1,
                "raw_publish_violation_count": 0,
                "missing_comment_helper_import_count": 1,
                "missing_issue_helper_import_count": 0,
                "results": [
                    {"filename": "security-audit.yml", "ok": False},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");
const workflowPublishHelperPath = process.argv[1];

process.env.EVAL_COMBINED_SCORE = "0.830";
process.env.EVAL_VISION_SCORE = "0.840";
process.env.EVAL_OCR_SCORE = "0.930";
process.env.EVAL_MIN_COMBINED = "0.800";
process.env.EVAL_MIN_VISION = "0.650";
process.env.EVAL_MIN_OCR = "0.900";
process.env.WORKFLOW_PUBLISH_HELPER_SUMMARY_JSON_FOR_COMMENT = workflowPublishHelperPath;

let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 104 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called for empty comment list");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 991 },
  sha: "5566778899001122",
  runId: "505050505",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentEvaluationReportPR({ github, context, process: mockProcess });
  if (!createdPayload) {
    throw new Error("createComment was not called");
  }
  const body = String(createdPayload.body || "");
  if (!body.includes("Workflow Publish Helper Adoption")) {
    throw new Error("comment body missing workflow publish helper section");
  }
  if (!body.includes("checked=33, failed=1, raw=0, missing_comment_helper=1, missing_issue_helper=0")) {
    throw new Error("comment body missing workflow publish helper summary");
  }
  if (!body.includes("failing=security-audit.yml")) {
    throw new Error("comment body missing failing workflow filename");
  }
  console.log("ok:create-comment-with-workflow-publish-helper-summary");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script, str(workflow_publish_helper))
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create-comment-with-workflow-publish-helper-summary" in result.stdout


def test_comment_evaluation_report_pr_js_marks_publish_helper_parse_errors(
    tmp_path: Path,
) -> None:
    workflow_publish_helper = tmp_path / "workflow_publish_helper_invalid.json"
    workflow_publish_helper.write_text("{invalid json", encoding="utf-8")

    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");
const workflowPublishHelperPath = process.argv[1];

process.env.EVAL_COMBINED_SCORE = "0.830";
process.env.EVAL_VISION_SCORE = "0.840";
process.env.EVAL_OCR_SCORE = "0.930";
process.env.EVAL_MIN_COMBINED = "0.800";
process.env.EVAL_MIN_VISION = "0.650";
process.env.EVAL_MIN_OCR = "0.900";
process.env.WORKFLOW_PUBLISH_HELPER_SUMMARY_JSON_FOR_COMMENT = workflowPublishHelperPath;

let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 105 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called for empty comment list");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 992 },
  sha: "6677889900112233",
  runId: "606060606",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentEvaluationReportPR({ github, context, process: mockProcess });
  if (!createdPayload) {
    throw new Error("createComment was not called");
  }
  const body = String(createdPayload.body || "");
  if (!body.includes("Workflow Publish Helper Adoption")) {
    throw new Error("comment body missing workflow publish helper section");
  }
  if (!body.includes("parse_error")) {
    throw new Error("comment body missing workflow publish helper parse_error");
  }
  console.log("ok:create-comment-with-workflow-publish-helper-parse-error");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script, str(workflow_publish_helper))
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create-comment-with-workflow-publish-helper-parse-error" in result.stdout


def test_comment_evaluation_report_pr_js_includes_guardrail_summary(
    tmp_path: Path,
) -> None:
    workflow_guardrail = tmp_path / "workflow_guardrail.json"
    workflow_guardrail.write_text(
        json.dumps(
            {
                "overall_status": "error",
                "overall_light": "🔴",
                "summary": "status=error, workflow_health=ok, inventory=error, publish_helper=ok",
                "workflow_inventory": {
                    "status": "error",
                    "summary": "workflows=33, duplicate=1, missing_required=0, non_unique_required=0",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");
const workflowGuardrailPath = process.argv[1];

process.env.EVAL_COMBINED_SCORE = "0.830";
process.env.EVAL_VISION_SCORE = "0.840";
process.env.EVAL_OCR_SCORE = "0.930";
process.env.EVAL_MIN_COMBINED = "0.800";
process.env.EVAL_MIN_VISION = "0.650";
process.env.EVAL_MIN_OCR = "0.900";
process.env.WORKFLOW_GUARDRAIL_SUMMARY_JSON_FOR_COMMENT = workflowGuardrailPath;

let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 106 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called for empty comment list");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 993 },
  sha: "7788990011223344",
  runId: "707070707",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentEvaluationReportPR({ github, context, process: mockProcess });
  if (!createdPayload) {
    throw new Error("createComment was not called");
  }
  const body = String(createdPayload.body || "");
  if (!body.includes("Workflow Guardrail Summary")) {
    throw new Error("comment body missing workflow guardrail section");
  }
  if (!body.includes("status=error, workflow_health=ok, inventory=error, publish_helper=ok")) {
    throw new Error("comment body missing workflow guardrail summary");
  }
  if (!body.includes("workflow_inventory=error:workflows=33, duplicate=1, missing_required=0, non_unique_required=0")) {
    throw new Error("comment body missing workflow guardrail detail");
  }
  if (!body.includes("**Workflow Guardrails**")) {
    throw new Error("comment body missing workflow guardrails signal");
  }
  console.log("ok:create-comment-with-workflow-guardrail-summary");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script, str(workflow_guardrail))
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create-comment-with-workflow-guardrail-summary" in result.stdout


def test_comment_evaluation_report_pr_js_marks_guardrail_parse_errors(
    tmp_path: Path,
) -> None:
    workflow_guardrail = tmp_path / "workflow_guardrail_invalid.json"
    workflow_guardrail.write_text("{invalid json", encoding="utf-8")

    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");
const workflowGuardrailPath = process.argv[1];

process.env.EVAL_COMBINED_SCORE = "0.830";
process.env.EVAL_VISION_SCORE = "0.840";
process.env.EVAL_OCR_SCORE = "0.930";
process.env.EVAL_MIN_COMBINED = "0.800";
process.env.EVAL_MIN_VISION = "0.650";
process.env.EVAL_MIN_OCR = "0.900";
process.env.WORKFLOW_GUARDRAIL_SUMMARY_JSON_FOR_COMMENT = workflowGuardrailPath;

let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 107 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called for empty comment list");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 994 },
  sha: "8899001122334455",
  runId: "808080808",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentEvaluationReportPR({ github, context, process: mockProcess });
  if (!createdPayload) {
    throw new Error("createComment was not called");
  }
  const body = String(createdPayload.body || "");
  if (!body.includes("Workflow Guardrail Summary")) {
    throw new Error("comment body missing workflow guardrail section");
  }
  if (!body.includes("parse_error")) {
    throw new Error("comment body missing workflow guardrail parse_error");
  }
  console.log("ok:create-comment-with-workflow-guardrail-parse-error");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script, str(workflow_guardrail))
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create-comment-with-workflow-guardrail-parse-error" in result.stdout


def test_comment_evaluation_report_pr_js_includes_ci_workflow_guardrail_overview(
    tmp_path: Path,
) -> None:
    overview = tmp_path / "ci_workflow_guardrail_overview.json"
    overview.write_text(
        json.dumps(
            {
                "overall_status": "error",
                "overall_light": "🔴",
                "summary": "status=error, ci_watch=ok, workflow_guardrail=error",
                "workflow_guardrail": {
                    "status": "error",
                    "summary": "status=error, workflow_health=ok, inventory=error, publish_helper=ok",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");
const overviewPath = process.argv[1];

process.env.EVAL_COMBINED_SCORE = "0.830";
process.env.EVAL_VISION_SCORE = "0.840";
process.env.EVAL_OCR_SCORE = "0.930";
process.env.EVAL_MIN_COMBINED = "0.800";
process.env.EVAL_MIN_VISION = "0.650";
process.env.EVAL_MIN_OCR = "0.900";
process.env.CI_WORKFLOW_GUARDRAIL_OVERVIEW_JSON_FOR_COMMENT = overviewPath;

let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 108 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called for empty comment list");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 995 },
  sha: "9900112233445566",
  runId: "909090909",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentEvaluationReportPR({ github, context, process: mockProcess });
  if (!createdPayload) {
    throw new Error("createComment was not called");
  }
  const body = String(createdPayload.body || "");
  if (!body.includes("CI Workflow Guardrail Overview")) {
    throw new Error("comment body missing ci workflow guardrail overview section");
  }
  if (!body.includes("status=error, ci_watch=ok, workflow_guardrail=error")) {
    throw new Error("comment body missing ci workflow guardrail overview summary");
  }
  if (!body.includes("workflow_guardrail=error:status=error, workflow_health=ok, inventory=error, publish_helper=ok")) {
    throw new Error("comment body missing ci workflow guardrail overview detail");
  }
  if (!body.includes("**CI+Workflow Overview**")) {
    throw new Error("comment body missing ci+workflow overview signal");
  }
  console.log("ok:create-comment-with-ci-workflow-guardrail-overview");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script, str(overview))
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create-comment-with-ci-workflow-guardrail-overview" in result.stdout


def test_comment_evaluation_report_pr_js_marks_ci_workflow_guardrail_overview_parse_errors(
    tmp_path: Path,
) -> None:
    overview = tmp_path / "ci_workflow_guardrail_overview_invalid.json"
    overview.write_text("{invalid json", encoding="utf-8")

    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");
const overviewPath = process.argv[1];

process.env.EVAL_COMBINED_SCORE = "0.830";
process.env.EVAL_VISION_SCORE = "0.840";
process.env.EVAL_OCR_SCORE = "0.930";
process.env.EVAL_MIN_COMBINED = "0.800";
process.env.EVAL_MIN_VISION = "0.650";
process.env.EVAL_MIN_OCR = "0.900";
process.env.CI_WORKFLOW_GUARDRAIL_OVERVIEW_JSON_FOR_COMMENT = overviewPath;

let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 109 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called for empty comment list");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 996 },
  sha: "0011223344556677",
  runId: "919191919",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentEvaluationReportPR({ github, context, process: mockProcess });
  if (!createdPayload) {
    throw new Error("createComment was not called");
  }
  const body = String(createdPayload.body || "");
  if (!body.includes("CI Workflow Guardrail Overview")) {
    throw new Error("comment body missing ci workflow guardrail overview section");
  }
  if (!body.includes("parse_error")) {
    throw new Error("comment body missing ci workflow guardrail overview parse_error");
  }
  console.log("ok:create-comment-with-ci-workflow-guardrail-overview-parse-error");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script, str(overview))
    assert result.returncode == 0, result.stderr or result.stdout
    assert (
        "ok:create-comment-with-ci-workflow-guardrail-overview-parse-error"
        in result.stdout
    )


def test_comment_evaluation_report_pr_js_marks_ci_watch_parse_errors(
    tmp_path: Path,
) -> None:
    ci_watch_summary = tmp_path / "ci_watch_invalid.json"
    ci_watch_summary.write_text("{invalid json", encoding="utf-8")

    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");
const ciWatchPath = process.argv[1];

process.env.EVAL_COMBINED_SCORE = "0.830";
process.env.EVAL_VISION_SCORE = "0.840";
process.env.EVAL_OCR_SCORE = "0.930";
process.env.EVAL_MIN_COMBINED = "0.800";
process.env.EVAL_MIN_VISION = "0.650";
process.env.EVAL_MIN_OCR = "0.900";
process.env.CI_WATCH_SUMMARY_JSON_FOR_COMMENT = ciWatchPath;

let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 101 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called for empty comment list");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 988 },
  sha: "2233445566778899",
  runId: "202020202",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentEvaluationReportPR({ github, context, process: mockProcess });
  if (!createdPayload) {
    throw new Error("createComment was not called");
  }
  const body = String(createdPayload.body || "");
  if (!body.includes("CI Watch Failure Details")) {
    throw new Error("comment body missing ci watch section");
  }
  if (!body.includes("parse_error")) {
    throw new Error("comment body missing ci watch parse_error");
  }
  console.log("ok:create-comment-with-ci-watch-parse-error");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script, str(ci_watch_summary))
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create-comment-with-ci-watch-parse-error" in result.stdout


def test_comment_evaluation_report_pr_js_includes_ci_watch_validation_report(
    tmp_path: Path,
) -> None:
    ci_watch_validation = tmp_path / "ci_watch_validation.json"
    ci_watch_validation.write_text(
        json.dumps(
            {
                "verdict": "FAIL",
                "verdict_success": False,
                "summary": "verdict=FAIL, reason=workflow_failed, failed=1, missing_required=0, workflow_guardrail=ok, ci_workflow_overview=error",
                "sections": {
                    "readiness": {"present": True, "ok": True},
                    "soft_smoke": {"present": True, "overall_exit_code": 2, "attempts_total": 3},
                    "workflow_guardrail_summary": {
                        "present": True,
                        "overall_status": "ok",
                        "summary": "status=ok, workflow_health=ok, inventory=ok, publish_helper=ok",
                    },
                    "ci_workflow_guardrail_overview": {
                        "present": True,
                        "overall_status": "error",
                        "summary": "status=error, ci_watch=ok, workflow_guardrail=error",
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");
const reportPath = process.argv[1];

process.env.EVAL_COMBINED_SCORE = "0.830";
process.env.EVAL_VISION_SCORE = "0.840";
process.env.EVAL_OCR_SCORE = "0.930";
process.env.EVAL_MIN_COMBINED = "0.800";
process.env.EVAL_MIN_VISION = "0.650";
process.env.EVAL_MIN_OCR = "0.900";
process.env.CI_WATCH_VALIDATION_REPORT_JSON_FOR_COMMENT = reportPath;

let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 120 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called for empty comment list");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 997 },
  sha: "1122334455667788",
  runId: "929292929",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentEvaluationReportPR({ github, context, process: mockProcess });
  if (!createdPayload) {
    throw new Error("createComment was not called");
  }
  const body = String(createdPayload.body || "");
  if (!body.includes("CI Watch Validation Report")) {
    throw new Error("comment body missing ci watch validation report section");
  }
  if (!body.includes("verdict=FAIL, reason=workflow_failed, failed=1, missing_required=0, workflow_guardrail=ok, ci_workflow_overview=error")) {
    throw new Error("comment body missing ci watch validation report summary");
  }
  if (!body.includes("soft_smoke=exit=2, attempts=3")) {
    throw new Error("comment body missing ci watch validation soft-smoke detail");
  }
  if (!body.includes("ci_workflow_guardrail_overview=error:status=error, ci_watch=ok, workflow_guardrail=error")) {
    throw new Error("comment body missing ci watch validation overview detail");
  }
  if (!body.includes("**CI Watch Validation**")) {
    throw new Error("comment body missing ci watch validation signal");
  }
  console.log("ok:create-comment-with-ci-watch-validation-report");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script, str(ci_watch_validation))
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create-comment-with-ci-watch-validation-report" in result.stdout


def test_comment_evaluation_report_pr_js_marks_ci_watch_validation_report_parse_errors(
    tmp_path: Path,
) -> None:
    ci_watch_validation = tmp_path / "ci_watch_validation_invalid.json"
    ci_watch_validation.write_text("{invalid json", encoding="utf-8")

    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");
const reportPath = process.argv[1];

process.env.EVAL_COMBINED_SCORE = "0.830";
process.env.EVAL_VISION_SCORE = "0.840";
process.env.EVAL_OCR_SCORE = "0.930";
process.env.EVAL_MIN_COMBINED = "0.800";
process.env.EVAL_MIN_VISION = "0.650";
process.env.EVAL_MIN_OCR = "0.900";
process.env.CI_WATCH_VALIDATION_REPORT_JSON_FOR_COMMENT = reportPath;

let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 121 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called for empty comment list");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 998 },
  sha: "2233445566778899",
  runId: "939393939",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentEvaluationReportPR({ github, context, process: mockProcess });
  if (!createdPayload) {
    throw new Error("createComment was not called");
  }
  const body = String(createdPayload.body || "");
  if (!body.includes("CI Watch Validation Report")) {
    throw new Error("comment body missing ci watch validation report section");
  }
  if (!body.includes("parse_error")) {
    throw new Error("comment body missing ci watch validation report parse_error");
  }
  console.log("ok:create-comment-with-ci-watch-validation-parse-error");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script, str(ci_watch_validation))
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create-comment-with-ci-watch-validation-parse-error" in result.stdout


def test_comment_evaluation_report_pr_js_marks_workflow_health_parse_errors(
    tmp_path: Path,
) -> None:
    workflow_file_health = tmp_path / "workflow_file_health_invalid.json"
    workflow_file_health.write_text("{invalid json", encoding="utf-8")

    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");
const workflowFileHealthPath = process.argv[1];

process.env.EVAL_COMBINED_SCORE = "0.830";
process.env.EVAL_VISION_SCORE = "0.840";
process.env.EVAL_OCR_SCORE = "0.930";
process.env.EVAL_MIN_COMBINED = "0.800";
process.env.EVAL_MIN_VISION = "0.650";
process.env.EVAL_MIN_OCR = "0.900";
process.env.WORKFLOW_FILE_HEALTH_SUMMARY_JSON_FOR_COMMENT = workflowFileHealthPath;

let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 102 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called for empty comment list");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 989 },
  sha: "3344556677889900",
  runId: "303030303",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentEvaluationReportPR({ github, context, process: mockProcess });
  if (!createdPayload) {
    throw new Error("createComment was not called");
  }
  const body = String(createdPayload.body || "");
  if (!body.includes("Workflow File Health")) {
    throw new Error("comment body missing workflow file health section");
  }
  if (!body.includes("parse_error")) {
    throw new Error("comment body missing workflow file health parse_error");
  }
  console.log("ok:create-comment-with-workflow-health-parse-error");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script, str(workflow_file_health))
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create-comment-with-workflow-health-parse-error" in result.stdout


def test_comment_evaluation_report_pr_js_marks_multiple_parse_errors_together(
    tmp_path: Path,
) -> None:
    ci_watch_summary = tmp_path / "ci_watch_invalid.json"
    ci_watch_summary.write_text("{invalid json", encoding="utf-8")
    workflow_file_health = tmp_path / "workflow_file_health_invalid.json"
    workflow_file_health.write_text("{invalid json", encoding="utf-8")
    workflow_inventory = tmp_path / "workflow_inventory_invalid.json"
    workflow_inventory.write_text("{invalid json", encoding="utf-8")
    workflow_publish_helper = tmp_path / "workflow_publish_helper_invalid.json"
    workflow_publish_helper.write_text("{invalid json", encoding="utf-8")
    workflow_guardrail = tmp_path / "workflow_guardrail_invalid.json"
    workflow_guardrail.write_text("{invalid json", encoding="utf-8")
    ci_workflow_guardrail_overview = (
        tmp_path / "ci_workflow_guardrail_overview_invalid.json"
    )
    ci_workflow_guardrail_overview.write_text("{invalid json", encoding="utf-8")
    ci_watch_validation = tmp_path / "ci_watch_validation_invalid.json"
    ci_watch_validation.write_text("{invalid json", encoding="utf-8")

    node_script = r"""
const mod = require("./scripts/ci/comment_evaluation_report_pr.js");
const ciWatchPath = process.argv[1];
const workflowFileHealthPath = process.argv[2];
const workflowInventoryPath = process.argv[3];
const workflowPublishHelperPath = process.argv[4];
const workflowGuardrailPath = process.argv[5];
const ciWorkflowGuardrailOverviewPath = process.argv[6];
const ciWatchValidationPath = process.argv[7];

process.env.EVAL_COMBINED_SCORE = "0.830";
process.env.EVAL_VISION_SCORE = "0.840";
process.env.EVAL_OCR_SCORE = "0.930";
process.env.EVAL_MIN_COMBINED = "0.800";
process.env.EVAL_MIN_VISION = "0.650";
process.env.EVAL_MIN_OCR = "0.900";
process.env.CI_WATCH_SUMMARY_JSON_FOR_COMMENT = ciWatchPath;
process.env.WORKFLOW_FILE_HEALTH_SUMMARY_JSON_FOR_COMMENT = workflowFileHealthPath;
process.env.WORKFLOW_INVENTORY_REPORT_JSON_FOR_COMMENT = workflowInventoryPath;
process.env.WORKFLOW_PUBLISH_HELPER_SUMMARY_JSON_FOR_COMMENT = workflowPublishHelperPath;
process.env.WORKFLOW_GUARDRAIL_SUMMARY_JSON_FOR_COMMENT = workflowGuardrailPath;
process.env.CI_WORKFLOW_GUARDRAIL_OVERVIEW_JSON_FOR_COMMENT = ciWorkflowGuardrailOverviewPath;
process.env.CI_WATCH_VALIDATION_REPORT_JSON_FOR_COMMENT = ciWatchValidationPath;

let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({ data: [] }),
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 103 } };
      },
      updateComment: async () => {
        throw new Error("updateComment should not be called for empty comment list");
      },
    },
  },
};
const context = {
  repo: { owner: "zensgit", repo: "cad-ml-platform" },
  issue: { number: 990 },
  sha: "4455667788990011",
  runId: "404040404",
};
const mockProcess = { env: process.env };

(async () => {
  await mod.commentEvaluationReportPR({ github, context, process: mockProcess });
  if (!createdPayload) {
    throw new Error("createComment was not called");
  }
  const body = String(createdPayload.body || "");
  if (!body.includes("CI Watch Failure Details")) {
    throw new Error("comment body missing ci watch section");
  }
  if (!body.includes("Workflow File Health")) {
    throw new Error("comment body missing workflow file health section");
  }
  if (!body.includes("Workflow Inventory Audit")) {
    throw new Error("comment body missing workflow inventory section");
  }
  if (!body.includes("Workflow Publish Helper Adoption")) {
    throw new Error("comment body missing workflow publish helper section");
  }
  if (!body.includes("Workflow Guardrail Summary")) {
    throw new Error("comment body missing workflow guardrail section");
  }
  if (!body.includes("CI Watch Validation Report")) {
    throw new Error("comment body missing ci watch validation report section");
  }
  if (!body.includes("CI Workflow Guardrail Overview")) {
    throw new Error("comment body missing ci workflow guardrail overview section");
  }
  const parseErrorCount = (body.match(/parse_error/g) || []).length;
  if (parseErrorCount < 7) {
    throw new Error(`expected at least 7 parse_error markers, got ${parseErrorCount}`);
  }
  console.log("ok:create-comment-with-multiple-parse-errors");
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
        str(workflow_publish_helper),
        str(workflow_guardrail),
        str(ci_workflow_guardrail_overview),
        str(ci_watch_validation),
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create-comment-with-multiple-parse-errors" in result.stdout
