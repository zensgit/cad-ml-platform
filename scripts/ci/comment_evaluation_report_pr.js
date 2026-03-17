"use strict";

const fs = require("fs");
const {
  markdownFooter,
  markdownSection,
  markdownTable,
} = require("./comment_markdown_utils.js");

function envStr(name, fallback = "") {
  const value = process.env[name];
  if (value === undefined || value === null) {
    return String(fallback);
  }
  return String(value);
}

function envBool(name, fallback = false) {
  const text = envStr(name, fallback ? "true" : "false").trim().toLowerCase();
  return text === "1" || text === "true" || text === "yes" || text === "on";
}

function envFloat(name, fallback = Number.NaN) {
  const value = parseFloat(envStr(name, String(fallback)));
  return Number.isNaN(value) ? fallback : value;
}

function parseBoolText(raw, fallback = false) {
  const text = String(raw === undefined || raw === null ? "" : raw)
    .trim()
    .toLowerCase();
  if (!text) {
    return Boolean(fallback);
  }
  return text === "1" || text === "true" || text === "yes" || text === "on";
}

function strictPlaybookAnchor(channel, reason) {
  const key = String(channel || "").trim().toLowerCase();
  const token = String(reason || "").trim().toLowerCase();
  if (key === "graph2d_review") {
    if (token.includes("gate_failed_under_strict_mode")) {
      return "graph2d-review-gate-failed-under-strict-mode";
    }
    if (token.includes("strict_mode_disabled")) {
      return "graph2d-review-strict-mode-disabled";
    }
    return "graph2d-review-generic";
  }
  if (key === "hybrid_blind") {
    if (token.includes("strict_mode_requires_real_dataset")) {
      return "hybrid-blind-strict-mode-requires-real-dataset";
    }
    if (token.includes("gate_failed_under_strict_mode")) {
      return "hybrid-blind-gate-failed-under-strict-mode";
    }
    return "hybrid-blind-generic";
  }
  if (key === "hybrid_calibration") {
    if (token.includes("gate_failed_under_strict_mode")) {
      return "hybrid-calibration-gate-failed-under-strict-mode";
    }
    return "hybrid-calibration-generic";
  }
  if (key === "hybrid_superpass") {
    if (token.includes("superpass_failed_under_strict_mode")) {
      return "hybrid-superpass-superpass-failed-under-strict-mode";
    }
    return "hybrid-superpass-generic";
  }
  if (key === "hybrid_superpass_validation") {
    return "hybrid-superpass-validation-nonzero-exit";
  }
  return "generic-strict-gate";
}

function strictPlaybookLabel(channel, reason) {
  return `${String(channel || "unknown")}:${String(reason || "unknown")}`;
}

function buildEvaluationReportCommentBody({
  overallStatus,
  combined,
  minCombined,
  combinedStatus,
  vision,
  minVision,
  visionStatus,
  ocr,
  minOcr,
  ocrStatus,
  hasAnomalies,
  securityStatus,
  reviewPackStatus,
  reviewPackInsights,
  reviewGateStatus,
  reviewGateStrictStatus,
  trainSweepStatus,
  hybridBlindEvalStatus,
  hybridBlindGateStatus,
  hybridBlindStrictStatus,
  hybridBlindDriftStatus,
  blindGainSummary,
  hybridCalibrationStatus,
  hybridCalibrationGateStatus,
  hybridCalibrationStrictStatus,
  hybridSuperpassStrictStatus,
  hybridSuperpassValidationStrictStatus,
  hybridCalibrationBaselineStatus,
  evaluationStrictMode,
  evaluationStrictModeRawValue,
  strictDecisionResult,
  strictPlaybookSummary,
  ciWatchFailureSummary,
  workflowFileHealthSummary,
  workflowInventorySummary,
  reviewPackLight,
  reviewGateLight,
  trainSweepLight,
  hybridBlindLight,
  hybridCalibrationLight,
  strictDecisionLight,
  strictFailureRequestsCount,
  ciWatchFailureLight,
  workflowFileHealthLight,
  workflowInventoryLight,
  evaluationStrictModeResolvedRaw,
  strictFailureRequestSummary,
  strictActionItems,
  strictActionChecklist,
  runUrl,
  updatedAt,
  commitSha,
}) {
  return [
    "## 📊 CAD ML Platform - Evaluation Results",
    "",
    overallStatus,
    "",
    markdownSection(
      "Scores",
      markdownTable(
        ["Module", "Score", "Threshold", "Status"],
        [
          ["**Combined**", Number(combined).toFixed(3), minCombined, combinedStatus],
          ["**Vision**", Number(vision).toFixed(3), minVision, visionStatus],
          ["**OCR**", Number(ocr).toFixed(3), minOcr, ocrStatus],
        ],
      ),
    ),
    "",
    markdownSection("Formula", "`Combined Score = 0.5 × Vision + 0.5 × OCR_normalized`"),
    "",
    markdownSection(
      "Additional Analysis",
      markdownTable(
        ["Check", "Status"],
        [
          ["**Anomaly Detection**", hasAnomalies ? "⚠️ Anomalies detected" : "✅ No anomalies"],
          ["**Security Audit**", securityStatus === "pass" ? "✅ Passed" : "⚠️ Issues found"],
          ["**Graph2D Review Pack**", reviewPackStatus],
          ["**Graph2D Review Insights**", reviewPackInsights],
          ["**Graph2D Review Gate**", reviewGateStatus],
          ["**Graph2D Review Gate Strict**", reviewGateStrictStatus],
          ["**Graph2D Train Sweep**", trainSweepStatus],
          ["**Hybrid Blind Eval**", hybridBlindEvalStatus],
          ["**Hybrid Blind Gate**", hybridBlindGateStatus],
          ["**Hybrid Blind Strict**", hybridBlindStrictStatus],
          ["**Hybrid Blind Drift Alert**", hybridBlindDriftStatus],
          ["**Blind Gain (Hybrid-Graph2D)**", blindGainSummary],
          ["**Hybrid Calibration**", hybridCalibrationStatus],
          ["**Hybrid Calibration Gate**", hybridCalibrationGateStatus],
          ["**Hybrid Calibration Strict**", hybridCalibrationStrictStatus],
          ["**Hybrid Superpass Strict**", hybridSuperpassStrictStatus],
          ["**Hybrid Superpass Validation Strict**", hybridSuperpassValidationStrictStatus],
          ["**Hybrid Calibration Baseline**", hybridCalibrationBaselineStatus],
          [
            "**Strict Gate Policy**",
            `mode=${evaluationStrictMode}, raw=${evaluationStrictModeRawValue || "n/a"}, result=${strictDecisionResult}`,
          ],
          ["**Strict Gate Playbook**", strictPlaybookSummary],
          ["**CI Watch Failure Details**", ciWatchFailureSummary],
          ["**Workflow File Health**", workflowFileHealthSummary],
          ["**Workflow Inventory Audit**", workflowInventorySummary],
        ],
      ),
    ),
    "",
    markdownSection(
      "Signal Lights",
      markdownTable(
        ["Signal", "State", "Detail"],
        [
          ["**Review Pack**", reviewPackLight, reviewPackStatus],
          ["**Review Gate**", reviewGateLight, reviewGateStatus],
          ["**Train Sweep**", trainSweepLight, trainSweepStatus],
          ["**Hybrid Blind**", hybridBlindLight, hybridBlindEvalStatus],
          ["**Hybrid Calibration**", hybridCalibrationLight, hybridCalibrationStatus],
          [
            "**Strict Gate Policy**",
            strictDecisionLight,
            `mode=${evaluationStrictMode}, strict_requests=${strictFailureRequestsCount}, result=${strictDecisionResult}`,
          ],
          ["**CI Watcher**", ciWatchFailureLight, ciWatchFailureSummary],
          ["**Workflow Health**", workflowFileHealthLight, workflowFileHealthSummary],
          ["**Workflow Inventory**", workflowInventoryLight, workflowInventorySummary],
        ],
      ),
    ),
    "",
    markdownSection(
      "Strict Gate Decision Path",
      markdownTable(
        ["Item", "Value"],
        [
          [
            "**Mode**",
            `${evaluationStrictMode} (resolved=${evaluationStrictModeResolvedRaw || "n/a"}, raw=${evaluationStrictModeRawValue || "n/a"})`,
          ],
          ["**Requested Failures**", strictFailureRequestSummary],
          ["**Decision**", strictDecisionResult],
          ["**Playbook Links**", strictPlaybookSummary],
          ["**Recommended Action**", strictActionItems[0] || "n/a"],
        ],
      ),
    ),
    "",
    strictActionChecklist,
    "",
    markdownSection(
      "Quick Actions",
      [
        `- 📋 [View Full Report](${runUrl})`,
        `- 📈 [Download Artifacts](${runUrl}#artifacts)`,
        `- 🔍 [Check Logs](${runUrl}/jobs)`,
      ].join("\n"),
    ),
    "",
    markdownFooter({
      updatedAt,
      sha: String(commitSha || "").substring(0, 7),
    }),
  ].join("\n");
}

async function commentEvaluationReportPR({ github, context, process }) {
  try {
    const combined = envFloat("EVAL_COMBINED_SCORE", 0.0);
    const vision = envFloat("EVAL_VISION_SCORE", 0.0);
    const ocr = envFloat("EVAL_OCR_SCORE", 0.0);

    const minCombined = envFloat("EVAL_MIN_COMBINED", 0.8);
    const minVision = envFloat("EVAL_MIN_VISION", 0.65);
    const minOcr = envFloat("EVAL_MIN_OCR", 0.9);
    const evaluationStrictModeDefaultRaw = envStr(
      "EVALUATION_STRICT_FAIL_MODE",
      "hard",
    );
    const evaluationStrictModeResolvedRaw = envStr(
      "EVALUATION_STRICT_FAIL_MODE_RESOLVED",
      evaluationStrictModeDefaultRaw,
    );
    const evaluationStrictModeRaw = (
      evaluationStrictModeResolvedRaw ||
      evaluationStrictModeDefaultRaw ||
      "hard"
    )
      .trim()
      .toLowerCase();
    const evaluationStrictModeRawValue = envStr(
      "EVALUATION_STRICT_FAIL_MODE_RAW",
      evaluationStrictModeDefaultRaw,
    );
    const evaluationStrictMode =
      evaluationStrictModeRaw === "soft" ? "soft" : "hard";

    const combinedStatus = combined >= minCombined ? "✅ Pass" : "❌ Fail";
    const visionStatus = vision >= minVision ? "✅ Pass" : "❌ Fail";
    const ocrStatus = ocr >= minOcr ? "✅ Pass" : "❌ Fail";

    const overallStatus =
      combined >= minCombined && vision >= minVision && ocr >= minOcr
        ? "✅ **All checks passed!**"
        : "⚠️ **Some checks failed - review required**";

    const hasAnomalies = envBool("INSIGHTS_HAS_ANOMALIES", false);
    const securityStatus = envStr("SECURITY_STATUS", "unknown");
    const reviewPackEnabled = envBool("GRAPH2D_REVIEW_PACK_ENABLED", false);
    const reviewGateEnabled = envBool("GRAPH2D_REVIEW_GATE_ENABLED", false);
    const trainSweepEnabled = envBool("GRAPH2D_TRAIN_SWEEP_ENABLED", false);

    const reviewCandidates = envStr("GRAPH2D_REVIEW_CANDIDATES", "0");
    const reviewRejected = envStr("GRAPH2D_REVIEW_REJECTED", "0");
    const reviewConflicts = envStr("GRAPH2D_REVIEW_CONFLICTS", "0");
    const reviewTopReasons = envStr("GRAPH2D_REVIEW_TOP_REASONS", "");
    const reviewTopSources = envStr("GRAPH2D_REVIEW_TOP_SOURCES", "");
    const reviewExampleExplanations = envStr(
      "GRAPH2D_REVIEW_EXAMPLE_EXPLANATIONS",
      "",
    );

    const reviewGateStatusRaw = envStr("GRAPH2D_REVIEW_GATE_STATUS", "unknown");
    const reviewGateExitCode = envStr("GRAPH2D_REVIEW_GATE_EXIT_CODE", "n/a");
    const reviewGateHeadline = envStr("GRAPH2D_REVIEW_GATE_HEADLINE", "");
    const reviewGateStrictMode = envStr("GRAPH2D_REVIEW_GATE_STRICT_MODE", "false");
    const reviewGateStrictShouldFail = envStr(
      "GRAPH2D_REVIEW_GATE_STRICT_SHOULD_FAIL",
      "false",
    );
    const reviewGateStrictReason = envStr("GRAPH2D_REVIEW_GATE_STRICT_REASON", "");

    const sweepTotalRuns = envStr("GRAPH2D_SWEEP_TOTAL_RUNS", "0");
    const sweepFailedRuns = envStr("GRAPH2D_SWEEP_FAILED_RUNS", "0");
    const sweepBestRecipe = envStr("GRAPH2D_SWEEP_BEST_RECIPE", "n/a");
    const sweepBestSeed = envStr("GRAPH2D_SWEEP_BEST_SEED", "n/a");
    const sweepRecommendedEnv = envStr("GRAPH2D_SWEEP_RECOMMENDED_ENV", "n/a");
    const sweepBestRunScript = envStr("GRAPH2D_SWEEP_BEST_RUN_SCRIPT", "n/a");

    const graph2dBlindEnabled = envBool("GRAPH2D_BLIND_ENABLED", false);
    const graph2dBlindAccuracyRaw = envStr("GRAPH2D_BLIND_ACCURACY", "NaN");

    const hybridBlindEnabled = envBool("HYBRID_BLIND_ENABLED", false);
    const hybridBlindStatusRaw = envStr("HYBRID_BLIND_STATUS", "unknown");
    const hybridBlindExitCode = envStr("HYBRID_BLIND_EXIT_CODE", "n/a");
    const hybridBlindDatasetSource = envStr(
      "HYBRID_BLIND_DATASET_SOURCE",
      "unknown",
    );
    const hybridBlindAccuracyRaw = envStr("HYBRID_BLIND_ACCURACY", "NaN");
    const hybridBlindGraph2dAccuracyRaw = envStr(
      "HYBRID_BLIND_GRAPH2D_ACCURACY",
      "NaN",
    );
    const hybridBlindGainRaw = envStr("HYBRID_BLIND_GAIN", "NaN");
    const hybridBlindCoverageRaw = envStr("HYBRID_BLIND_COVERAGE", "n/a");

    const hybridBlindGateEnabled = envBool("HYBRID_BLIND_GATE_ENABLED", false);
    const hybridBlindGateStatusRaw = envStr("HYBRID_BLIND_GATE_STATUS", "unknown");
    const hybridBlindGateExitCode = envStr("HYBRID_BLIND_GATE_EXIT_CODE", "n/a");
    const hybridBlindGateHeadline = envStr("HYBRID_BLIND_GATE_HEADLINE", "");
    const hybridBlindStrictMode = envStr("HYBRID_BLIND_STRICT_MODE", "false");
    const hybridBlindStrictShouldFail = envStr(
      "HYBRID_BLIND_STRICT_SHOULD_FAIL",
      "false",
    );
    const hybridBlindStrictReason = envStr("HYBRID_BLIND_STRICT_REASON", "");
    const hybridBlindStrictRequireReal = envStr(
      "HYBRID_BLIND_STRICT_REQUIRE_REAL",
      "true",
    );

    const hybridBlindDriftEnabled = envBool("HYBRID_BLIND_DRIFT_ENABLED", false);
    const hybridBlindDriftStatusRaw = envStr("HYBRID_BLIND_DRIFT_STATUS", "unknown");
    const hybridBlindDriftExitCode = envStr("HYBRID_BLIND_DRIFT_EXIT_CODE", "n/a");
    const hybridBlindDriftHeadline = envStr("HYBRID_BLIND_DRIFT_HEADLINE", "");
    const hybridBlindDeltaAcc = envStr("HYBRID_BLIND_DELTA_ACC", "n/a");
    const hybridBlindDeltaGain = envStr("HYBRID_BLIND_DELTA_GAIN", "n/a");
    const hybridBlindDeltaCoverage = envStr("HYBRID_BLIND_DELTA_COVERAGE", "n/a");

    const hybridBlindLabelSliceEnabled = envStr(
      "HYBRID_BLIND_LABEL_SLICE_ENABLED",
      "false",
    );
    const hybridBlindLabelSliceAutoCap = envStr(
      "HYBRID_BLIND_LABEL_SLICE_AUTO_CAP",
      "true",
    );
    const hybridBlindLabelSliceEffectiveMinCommon = envStr(
      "HYBRID_BLIND_LABEL_SLICE_EFFECTIVE_MIN_COMMON",
      "n/a",
    );
    const hybridBlindLabelSliceCommonCount = envStr(
      "HYBRID_BLIND_LABEL_SLICE_COMMON_COUNT",
      "n/a",
    );
    const hybridBlindLabelSliceWorstAccDrop = envStr(
      "HYBRID_BLIND_LABEL_SLICE_WORST_ACC_DROP",
      "n/a",
    );
    const hybridBlindLabelSliceWorstAccDropLabel = envStr(
      "HYBRID_BLIND_LABEL_SLICE_WORST_ACC_DROP_LABEL",
      "n/a",
    );
    const hybridBlindLabelSliceWorstGainDrop = envStr(
      "HYBRID_BLIND_LABEL_SLICE_WORST_GAIN_DROP",
      "n/a",
    );
    const hybridBlindLabelSliceWorstGainDropLabel = envStr(
      "HYBRID_BLIND_LABEL_SLICE_WORST_GAIN_DROP_LABEL",
      "n/a",
    );

    const hybridBlindFamilySliceEnabled = envStr(
      "HYBRID_BLIND_FAMILY_SLICE_ENABLED",
      "false",
    );
    const hybridBlindFamilySliceAutoCap = envStr(
      "HYBRID_BLIND_FAMILY_SLICE_AUTO_CAP",
      "true",
    );
    const hybridBlindFamilySliceEffectiveMinCommon = envStr(
      "HYBRID_BLIND_FAMILY_SLICE_EFFECTIVE_MIN_COMMON",
      "n/a",
    );
    const hybridBlindFamilySliceCommonCount = envStr(
      "HYBRID_BLIND_FAMILY_SLICE_COMMON_COUNT",
      "n/a",
    );
    const hybridBlindFamilySliceWorstAccDrop = envStr(
      "HYBRID_BLIND_FAMILY_SLICE_WORST_ACC_DROP",
      "n/a",
    );
    const hybridBlindFamilySliceWorstAccDropFamily = envStr(
      "HYBRID_BLIND_FAMILY_SLICE_WORST_ACC_DROP_FAMILY",
      "n/a",
    );
    const hybridBlindFamilySliceWorstGainDrop = envStr(
      "HYBRID_BLIND_FAMILY_SLICE_WORST_GAIN_DROP",
      "n/a",
    );
    const hybridBlindFamilySliceWorstGainDropFamily = envStr(
      "HYBRID_BLIND_FAMILY_SLICE_WORST_GAIN_DROP_FAMILY",
      "n/a",
    );

    const hybridCalibrationEnabled = envBool("HYBRID_CALIBRATION_ENABLED", false);
    const hybridCalibrationStatusRaw = envStr("HYBRID_CALIBRATION_STATUS", "unknown");
    const hybridCalibrationExitCode = envStr("HYBRID_CALIBRATION_EXIT_CODE", "n/a");
    const hybridCalibrationSamples = envStr("HYBRID_CALIBRATION_SAMPLES", "0");
    const hybridCalibrationEce = envStr("HYBRID_CALIBRATION_ECE", "0");
    const hybridCalibrationBrier = envStr("HYBRID_CALIBRATION_BRIER", "0");
    const hybridCalibrationMce = envStr("HYBRID_CALIBRATION_MCE", "0");

    const hybridCalibrationGateEnabled = envBool(
      "HYBRID_CALIBRATION_GATE_ENABLED",
      false,
    );
    const hybridCalibrationGateStatusRaw = envStr(
      "HYBRID_CALIBRATION_GATE_STATUS",
      "unknown",
    );
    const hybridCalibrationGateExitCode = envStr(
      "HYBRID_CALIBRATION_GATE_EXIT_CODE",
      "n/a",
    );
    const hybridCalibrationGateHeadline = envStr(
      "HYBRID_CALIBRATION_GATE_HEADLINE",
      "",
    );
    const hybridCalibrationStrictMode = envStr(
      "HYBRID_CALIBRATION_STRICT_MODE",
      "false",
    );
    const hybridCalibrationStrictShouldFail = envStr(
      "HYBRID_CALIBRATION_STRICT_SHOULD_FAIL",
      "false",
    );
    const hybridCalibrationStrictReason = envStr(
      "HYBRID_CALIBRATION_STRICT_REASON",
      "",
    );
    const hybridSuperpassStrictMode = envStr(
      "HYBRID_SUPERPASS_STRICT_MODE",
      "false",
    );
    const hybridSuperpassStrictShouldFail = envStr(
      "HYBRID_SUPERPASS_STRICT_SHOULD_FAIL",
      "false",
    );
    const hybridSuperpassStrictReason = envStr(
      "HYBRID_SUPERPASS_STRICT_REASON",
      "",
    );
    const hybridSuperpassValidationStrictMode = envStr(
      "HYBRID_SUPERPASS_VALIDATION_STRICT_MODE",
      "false",
    );
    const hybridSuperpassValidationExitCode = envStr(
      "HYBRID_SUPERPASS_VALIDATION_EXIT_CODE",
      "0",
    );
    const hybridSuperpassValidationStatus = envStr(
      "HYBRID_SUPERPASS_VALIDATION_STATUS",
      "unknown",
    );

    const hybridCalibrationBaselineUpdateEnabled = envBool(
      "HYBRID_CALIBRATION_BASELINE_UPDATE_ENABLED",
      false,
    );
    const hybridCalibrationBaselineUpdateStatus = envStr(
      "HYBRID_CALIBRATION_BASELINE_UPDATE_STATUS",
      "unknown",
    );
    const hybridCalibrationBaselineUpdateExitCode = envStr(
      "HYBRID_CALIBRATION_BASELINE_UPDATE_EXIT_CODE",
      "n/a",
    );
    const hybridCalibrationBaselinePath = envStr(
      "HYBRID_CALIBRATION_BASELINE_PATH",
      "n/a",
    );

    const ciWatchSummaryPath = envStr("CI_WATCH_SUMMARY_JSON_FOR_COMMENT", "");
    let ciWatchFailureSummary = "⏭️ skipped (no summary path)";
    let ciWatchFailureLight = "⚪";
    if (ciWatchSummaryPath) {
      try {
        if (!fs.existsSync(ciWatchSummaryPath)) {
          ciWatchFailureSummary = `⚠️ summary missing at ${ciWatchSummaryPath}`;
          ciWatchFailureLight = "🟡";
        } else {
          const ciWatchSummaryPayload = JSON.parse(
            fs.readFileSync(ciWatchSummaryPath, "utf8"),
          );
          const ciWatchFailureDetails = Array.isArray(
            ciWatchSummaryPayload.failure_details,
          )
            ? ciWatchSummaryPayload.failure_details
            : [];
          const ciWatchCounts =
            ciWatchSummaryPayload.counts &&
            typeof ciWatchSummaryPayload.counts === "object"
              ? ciWatchSummaryPayload.counts
              : {};
          const ciWatchFailedCount =
            parseInt(
              String(ciWatchCounts.failed ?? ciWatchFailureDetails.length ?? 0),
              10,
            ) || 0;
          const ciWatchReason = String(ciWatchSummaryPayload.reason || "unknown");
          const ciWatchTop = ciWatchFailureDetails
            .slice(0, 3)
            .map((item) => {
              if (!item || typeof item !== "object") {
                return "unknown";
              }
              const wf = String(item.workflow_name || "unknown");
              const runId = String(item.run_id || "n/a");
              const step =
                Array.isArray(item.failed_steps) && item.failed_steps.length > 0
                  ? String(item.failed_steps[0])
                  : Array.isArray(item.failed_jobs) && item.failed_jobs.length > 0
                    ? String(item.failed_jobs[0])
                    : "n/a";
              return `${wf}#${runId}:${step}`;
            })
            .join("; ");
          if (ciWatchFailedCount > 0 || ciWatchFailureDetails.length > 0) {
            ciWatchFailureSummary = `failed=${ciWatchFailedCount}, reason=${ciWatchReason}, top=${ciWatchTop || "n/a"}`;
            ciWatchFailureLight = "🔴";
          } else {
            ciWatchFailureSummary = `failed=0, reason=${ciWatchReason}`;
            ciWatchFailureLight = "🟢";
          }
        }
      } catch (error) {
        const message = error && error.message ? error.message : String(error);
        ciWatchFailureSummary = `⚠️ parse_error: ${message}`;
        ciWatchFailureLight = "🟡";
      }
    }

    const workflowFileHealthSummaryPath = envStr(
      "WORKFLOW_FILE_HEALTH_SUMMARY_JSON_FOR_COMMENT",
      "",
    );
    let workflowFileHealthSummary = "⏭️ skipped (no summary path)";
    let workflowFileHealthLight = "⚪";
    if (workflowFileHealthSummaryPath) {
      try {
        if (!fs.existsSync(workflowFileHealthSummaryPath)) {
          workflowFileHealthSummary = `⚠️ summary missing at ${workflowFileHealthSummaryPath}`;
          workflowFileHealthLight = "🟡";
        } else {
          const payload = JSON.parse(
            fs.readFileSync(workflowFileHealthSummaryPath, "utf8"),
          );
          const failedCount =
            parseInt(String(payload.failed_count ?? payload.fail_count ?? 0), 10) ||
            0;
          const count = parseInt(String(payload.count ?? 0), 10) || 0;
          const modeUsed = String(payload.mode_used || "unknown");
          const fallbackReason = String(payload.fallback_reason || "none");
          if (failedCount > 0) {
            workflowFileHealthSummary = `failed=${failedCount}/${count}, mode=${modeUsed}, fallback=${fallbackReason}`;
            workflowFileHealthLight = "🔴";
          } else {
            workflowFileHealthSummary = `failed=0/${count}, mode=${modeUsed}, fallback=${fallbackReason}`;
            workflowFileHealthLight = "🟢";
          }
        }
      } catch (error) {
        const message = error && error.message ? error.message : String(error);
        workflowFileHealthSummary = `⚠️ parse_error: ${message}`;
        workflowFileHealthLight = "🟡";
      }
    }

    const workflowInventorySummaryPath = envStr(
      "WORKFLOW_INVENTORY_REPORT_JSON_FOR_COMMENT",
      "",
    );
    let workflowInventorySummary = "⏭️ skipped (no summary path)";
    let workflowInventoryLight = "⚪";
    if (workflowInventorySummaryPath) {
      try {
        if (!fs.existsSync(workflowInventorySummaryPath)) {
          workflowInventorySummary = `⚠️ summary missing at ${workflowInventorySummaryPath}`;
          workflowInventoryLight = "🟡";
        } else {
          const payload = JSON.parse(
            fs.readFileSync(workflowInventorySummaryPath, "utf8"),
          );
          const workflowCount =
            parseInt(String(payload.workflow_count ?? 0), 10) || 0;
          const duplicateCount =
            parseInt(String(payload.duplicate_name_count ?? 0), 10) || 0;
          const missingRequiredCount =
            parseInt(String(payload.missing_required_count ?? 0), 10) || 0;
          const nonUniqueRequiredCount =
            parseInt(String(payload.non_unique_required_count ?? 0), 10) || 0;
          const duplicates = Array.isArray(payload.duplicates)
            ? payload.duplicates
            : [];
          const requiredRows = Array.isArray(payload.required_workflow_mapping)
            ? payload.required_workflow_mapping
            : [];
          const duplicateNames = duplicates
            .map((row) => String((row && row.name) || "").trim())
            .filter(Boolean)
            .slice(0, 3);
          const missingRequiredNames = requiredRows
            .filter(
              (row) =>
                row &&
                typeof row === "object" &&
                String(row.status || "").trim() === "missing",
            )
            .map((row) => String(row.name || "").trim())
            .filter(Boolean)
            .slice(0, 3);
          const nonUniqueRequiredNames = requiredRows
            .filter(
              (row) =>
                row &&
                typeof row === "object" &&
                String(row.status || "").trim() === "non_unique",
            )
            .map((row) => String(row.name || "").trim())
            .filter(Boolean)
            .slice(0, 3);
          const workflowInventoryParts = [
            `workflows=${workflowCount}`,
            `duplicate=${duplicateCount}`,
            `missing_required=${missingRequiredCount}`,
            `non_unique_required=${nonUniqueRequiredCount}`,
          ];
          if (duplicateNames.length > 0) {
            workflowInventoryParts.push(`duplicate_names=${duplicateNames.join("/")}`);
          }
          if (missingRequiredNames.length > 0) {
            workflowInventoryParts.push(
              `missing_names=${missingRequiredNames.join("/")}`,
            );
          }
          if (nonUniqueRequiredNames.length > 0) {
            workflowInventoryParts.push(
              `non_unique_names=${nonUniqueRequiredNames.join("/")}`,
            );
          }
          workflowInventorySummary = workflowInventoryParts.join(", ");
          workflowInventoryLight =
            duplicateCount > 0 ||
            missingRequiredCount > 0 ||
            nonUniqueRequiredCount > 0
              ? "🔴"
              : "🟢";
        }
      } catch (error) {
        const message = error && error.message ? error.message : String(error);
        workflowInventorySummary = `⚠️ parse_error: ${message}`;
        workflowInventoryLight = "🟡";
      }
    }

    const reviewCandidateCount = parseInt(reviewCandidates || "0", 10);
    const sweepTotalRunsInt = parseInt(sweepTotalRuns || "0", 10);
    const sweepFailedRunsInt = parseInt(sweepFailedRuns || "0", 10);
    const graph2dBlindAccuracy = parseFloat(graph2dBlindAccuracyRaw || "NaN");
    const hybridBlindAccuracy = parseFloat(hybridBlindAccuracyRaw || "NaN");
    const hybridBlindGain = parseFloat(hybridBlindGainRaw || "NaN");

    const reviewPackStatus = reviewPackEnabled
      ? `🧪 candidates=${reviewCandidates}, rejected=${reviewRejected}, conflict=${reviewConflicts}`
      : "⏭️ skipped";
    const reviewPackInsights = reviewPackEnabled
      ? `reasons=${reviewTopReasons || "n/a"}, sources=${reviewTopSources || "n/a"}, examples=${reviewExampleExplanations || "n/a"}`
      : "⏭️ skipped";
    const reviewGateStatus = reviewGateEnabled
      ? `${reviewGateStatusRaw} (exit=${reviewGateExitCode}, headline=${reviewGateHeadline || "n/a"})`
      : "⏭️ skipped";
    const reviewGateStrictStatus = reviewGateEnabled
      ? `strict=${reviewGateStrictMode || "false"}, should_fail=${reviewGateStrictShouldFail || "false"}, reason=${reviewGateStrictReason || "n/a"}`
      : "⏭️ skipped";
    const trainSweepStatus = trainSweepEnabled
      ? `runs=${sweepTotalRuns}, failed=${sweepFailedRuns}, best=${sweepBestRecipe}@${sweepBestSeed}, env=${sweepRecommendedEnv}, script=${sweepBestRunScript}`
      : "⏭️ skipped";
    const hybridBlindEvalStatus = hybridBlindEnabled
      ? `${hybridBlindStatusRaw || "unknown"} (source=${hybridBlindDatasetSource || "unknown"}, exit=${hybridBlindExitCode || "n/a"}, coverage=${hybridBlindCoverageRaw || "n/a"}, hybrid_acc=${hybridBlindAccuracyRaw || "n/a"}, graph2d_acc=${hybridBlindGraph2dAccuracyRaw || "n/a"}, gain=${hybridBlindGainRaw || "n/a"})`
      : "⏭️ skipped";
    const hybridBlindGateStatus = hybridBlindGateEnabled
      ? `${hybridBlindGateStatusRaw || "unknown"} (exit=${hybridBlindGateExitCode || "n/a"}, headline=${hybridBlindGateHeadline || "n/a"})`
      : "⏭️ skipped";
    const hybridBlindStrictStatus = hybridBlindGateEnabled
      ? `strict=${hybridBlindStrictMode || "false"}, require_real=${hybridBlindStrictRequireReal || "true"}, should_fail=${hybridBlindStrictShouldFail || "false"}, reason=${hybridBlindStrictReason || "n/a"}`
      : "⏭️ skipped";
    const hybridBlindDriftStatus = hybridBlindDriftEnabled
      ? `${hybridBlindDriftStatusRaw || "unknown"} (exit=${hybridBlindDriftExitCode || "n/a"}, headline=${hybridBlindDriftHeadline || "n/a"}, delta_acc=${hybridBlindDeltaAcc || "n/a"}, delta_gain=${hybridBlindDeltaGain || "n/a"}, delta_cov=${hybridBlindDeltaCoverage || "n/a"}, label_slice_enabled=${hybridBlindLabelSliceEnabled || "false"}, label_auto_cap=${hybridBlindLabelSliceAutoCap || "true"}, label_effective_min_common=${hybridBlindLabelSliceEffectiveMinCommon || "n/a"}, label_common=${hybridBlindLabelSliceCommonCount || "n/a"}, worst_label_acc_drop=${hybridBlindLabelSliceWorstAccDrop || "n/a"}@${hybridBlindLabelSliceWorstAccDropLabel || "n/a"}, worst_label_gain_drop=${hybridBlindLabelSliceWorstGainDrop || "n/a"}@${hybridBlindLabelSliceWorstGainDropLabel || "n/a"}, family_slice_enabled=${hybridBlindFamilySliceEnabled || "false"}, family_auto_cap=${hybridBlindFamilySliceAutoCap || "true"}, family_effective_min_common=${hybridBlindFamilySliceEffectiveMinCommon || "n/a"}, family_common=${hybridBlindFamilySliceCommonCount || "n/a"}, worst_family_acc_drop=${hybridBlindFamilySliceWorstAccDrop || "n/a"}@${hybridBlindFamilySliceWorstAccDropFamily || "n/a"}, worst_family_gain_drop=${hybridBlindFamilySliceWorstGainDrop || "n/a"}@${hybridBlindFamilySliceWorstGainDropFamily || "n/a"})`
      : "⏭️ skipped";

    let blindGainSummary = "n/a";
    if (!Number.isNaN(hybridBlindGain)) {
      blindGainSummary = hybridBlindGain.toFixed(4);
    } else if (
      !Number.isNaN(hybridBlindAccuracy) &&
      !Number.isNaN(graph2dBlindAccuracy)
    ) {
      blindGainSummary = (hybridBlindAccuracy - graph2dBlindAccuracy).toFixed(4);
    }

    const hybridCalibrationStatus = hybridCalibrationEnabled
      ? `${hybridCalibrationStatusRaw || "unknown"} (exit=${hybridCalibrationExitCode || "n/a"}, n_samples=${hybridCalibrationSamples || "0"}, ece=${hybridCalibrationEce || "0"}, brier=${hybridCalibrationBrier || "0"}, mce=${hybridCalibrationMce || "0"})`
      : "⏭️ skipped";
    const hybridCalibrationGateStatus = hybridCalibrationGateEnabled
      ? `${hybridCalibrationGateStatusRaw || "unknown"} (exit=${hybridCalibrationGateExitCode || "n/a"}, headline=${hybridCalibrationGateHeadline || "n/a"})`
      : "⏭️ skipped";
    const hybridCalibrationStrictStatus = hybridCalibrationGateEnabled
      ? `strict=${hybridCalibrationStrictMode || "false"}, should_fail=${hybridCalibrationStrictShouldFail || "false"}, reason=${hybridCalibrationStrictReason || "n/a"}`
      : "⏭️ skipped";
    const hybridSuperpassStrictStatus = `strict=${hybridSuperpassStrictMode || "false"}, should_fail=${hybridSuperpassStrictShouldFail || "false"}, reason=${hybridSuperpassStrictReason || "n/a"}`;
    const hybridSuperpassValidationStrictStatus = `strict=${hybridSuperpassValidationStrictMode || "false"}, exit=${hybridSuperpassValidationExitCode || "0"}, status=${hybridSuperpassValidationStatus || "unknown"}`;
    const hybridCalibrationBaselineStatus = hybridCalibrationBaselineUpdateEnabled
      ? `${hybridCalibrationBaselineUpdateStatus || "unknown"} (exit=${hybridCalibrationBaselineUpdateExitCode || "n/a"}, path=${hybridCalibrationBaselinePath || "n/a"})`
      : "⏭️ skipped";

    const strictFailureItems = [];
    if (reviewGateEnabled && parseBoolText(reviewGateStrictShouldFail, false)) {
      strictFailureItems.push({
        channel: "graph2d_review",
        reason: reviewGateStrictReason || "gate_failed_under_strict_mode",
      });
    }
    if (
      hybridBlindGateEnabled &&
      parseBoolText(hybridBlindStrictShouldFail, false)
    ) {
      strictFailureItems.push({
        channel: "hybrid_blind",
        reason: hybridBlindStrictReason || "gate_failed_under_strict_mode",
      });
    }
    if (
      hybridCalibrationGateEnabled &&
      parseBoolText(hybridCalibrationStrictShouldFail, false)
    ) {
      strictFailureItems.push({
        channel: "hybrid_calibration",
        reason: hybridCalibrationStrictReason || "gate_failed_under_strict_mode",
      });
    }
    if (parseBoolText(hybridSuperpassStrictShouldFail, false)) {
      strictFailureItems.push({
        channel: "hybrid_superpass",
        reason: hybridSuperpassStrictReason || "superpass_failed_under_strict_mode",
      });
    }
    if (
      parseBoolText(hybridSuperpassValidationStrictMode, false) &&
      String(hybridSuperpassValidationExitCode || "0") !== "0"
    ) {
      strictFailureItems.push({
        channel: "hybrid_superpass_validation",
        reason: `validation_exit_${hybridSuperpassValidationExitCode || "nonzero"}`,
      });
    }

    const strictFailureRequests = strictFailureItems.map((item) =>
      strictPlaybookLabel(item.channel, item.reason),
    );
    const strictFailureRequestSummary =
      strictFailureRequests.length > 0
        ? strictFailureRequests.join("; ")
        : "none";
    const strictPlaybookBaseUrl = `https://github.com/${context.repo.owner}/${context.repo.repo}/blob/${context.sha}/docs/STRICT_GATE_PLAYBOOK.md`;
    const strictPlaybookLinks = strictFailureItems
      .slice(0, 5)
      .map((item) => {
        const anchor = strictPlaybookAnchor(item.channel, item.reason);
        const label = strictPlaybookLabel(item.channel, item.reason);
        return `[${label}](${strictPlaybookBaseUrl}#${anchor})`;
      });
    const strictPlaybookSummary =
      strictPlaybookLinks.length > 0
        ? strictPlaybookLinks.join(" ; ")
        : `[strict-gate-playbook](${strictPlaybookBaseUrl})`;
    const strictDecisionResult =
      strictFailureRequests.length === 0
        ? "no_strict_fail_requests"
        : evaluationStrictMode === "soft"
          ? "downgraded_to_warning"
          : "blocking_failure_expected";
    const strictDecisionLight =
      strictFailureRequests.length === 0
        ? "🟢"
        : evaluationStrictMode === "soft"
          ? "🟡"
          : "🔴";
    const strictActionItems = [];
    if (strictFailureRequests.length === 0) {
      strictActionItems.push("- ✅ No strict gate failure request detected.");
    } else if (evaluationStrictMode === "soft") {
      strictActionItems.push(
        "- ⚠️ Soft mode downgraded strict gate failures to warnings for this run.",
      );
      strictActionItems.push(
        "- 🔁 Before merge, switch `EVALUATION_STRICT_FAIL_MODE=hard` and rerun.",
      );
      strictActionItems.push(`- 🧭 Fix targets: ${strictFailureRequestSummary}`);
      strictActionItems.push(`- 📚 Playbook: ${strictPlaybookSummary}`);
    } else {
      strictActionItems.push(
        "- ❌ Hard mode blocking is active and strict gate failure was requested.",
      );
      strictActionItems.push(`- 🧭 Fix targets: ${strictFailureRequestSummary}`);
      strictActionItems.push(`- 📚 Playbook: ${strictPlaybookSummary}`);
    }
    const strictActionChecklist = strictActionItems.join("\n");

    const reviewPackLight = !reviewPackEnabled
      ? "⚪"
      : reviewCandidateCount > 0
        ? "🟡"
        : "🟢";
    const reviewGateLight = !reviewGateEnabled
      ? "⚪"
      : reviewGateStatusRaw === "passed"
        ? "🟢"
        : "🔴";
    const trainSweepLight = !trainSweepEnabled
      ? "⚪"
      : sweepTotalRunsInt <= 0
        ? "🟡"
        : sweepFailedRunsInt > 0
          ? "🔴"
          : "🟢";
    const hybridBlindLight = !hybridBlindEnabled
      ? "⚪"
      : hybridBlindGateEnabled && hybridBlindGateStatusRaw === "passed"
        ? "🟢"
        : hybridBlindGateEnabled && hybridBlindGateStatusRaw !== "passed"
          ? "🔴"
          : "🟡";
    const hybridCalibrationLight = !hybridCalibrationEnabled
      ? "⚪"
      : hybridCalibrationStatusRaw === "ok" &&
          (!hybridCalibrationGateEnabled ||
            hybridCalibrationGateStatusRaw === "passed")
        ? "🟢"
        : hybridCalibrationGateEnabled && hybridCalibrationGateStatusRaw !== "passed"
          ? "🔴"
          : "🟡";

    const runUrl = `https://github.com/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`;

    const body = buildEvaluationReportCommentBody({
      overallStatus,
      combined,
      minCombined,
      combinedStatus,
      vision,
      minVision,
      visionStatus,
      ocr,
      minOcr,
      ocrStatus,
      hasAnomalies,
      securityStatus,
      reviewPackStatus,
      reviewPackInsights,
      reviewGateStatus,
      reviewGateStrictStatus,
      trainSweepStatus,
      hybridBlindEvalStatus,
      hybridBlindGateStatus,
      hybridBlindStrictStatus,
      hybridBlindDriftStatus,
      blindGainSummary,
      hybridCalibrationStatus,
      hybridCalibrationGateStatus,
      hybridCalibrationStrictStatus,
      hybridSuperpassStrictStatus,
      hybridSuperpassValidationStrictStatus,
      hybridCalibrationBaselineStatus,
      evaluationStrictMode,
      evaluationStrictModeRawValue,
      strictDecisionResult,
      strictPlaybookSummary,
      ciWatchFailureSummary,
      workflowFileHealthSummary,
      workflowInventorySummary,
      reviewPackLight,
      reviewGateLight,
      trainSweepLight,
      hybridBlindLight,
      hybridCalibrationLight,
      strictDecisionLight,
      strictFailureRequestsCount: strictFailureRequests.length,
      ciWatchFailureLight,
      workflowFileHealthLight,
      workflowInventoryLight,
      evaluationStrictModeResolvedRaw,
      strictFailureRequestSummary,
      strictActionItems,
      strictActionChecklist,
      runUrl,
      updatedAt: new Date().toISOString().replace("T", " ").substring(0, 19),
      commitSha: context.sha,
    });

    const { data: comments } = await github.rest.issues.listComments({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: context.issue.number,
    });

    const botComment = comments.find(
      (comment) =>
        comment.user.type === "Bot" &&
        comment.body.includes("CAD ML Platform - Evaluation Results"),
    );

    if (botComment) {
      await github.rest.issues.updateComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        comment_id: botComment.id,
        body,
      });
    } else {
      await github.rest.issues.createComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: context.issue.number,
        body,
      });
    }
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    console.warn(`PR comment skipped: ${message}`);
  }
}

module.exports = {
  buildEvaluationReportCommentBody,
  commentEvaluationReportPR,
};
