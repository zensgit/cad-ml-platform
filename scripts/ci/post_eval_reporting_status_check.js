"use strict";

const fs = require("fs");

/**
 * Map release_readiness to GitHub commit status state.
 *
 * GitHub commit status API accepts: error, failure, pending, success.
 * We map: ready→success, degraded→success (with description), unavailable→failure.
 */
function mapReadinessToState(readiness) {
  switch (String(readiness || "").trim()) {
    case "ready":
      return "success";
    case "degraded":
      return "success";
    case "unavailable":
      return "failure";
    default:
      return "failure";
  }
}

function mapReadinessToDescription(readiness, summary) {
  const ss = summary || {};
  const missing = Number(ss.missing_count || 0);
  const stale = Number(ss.stale_count || 0);
  const mismatch = Number(ss.mismatch_count || 0);

  switch (String(readiness || "").trim()) {
    case "ready":
      return "Eval reporting stack is healthy";
    case "degraded":
      return `Degraded: missing=${missing}, stale=${stale}, mismatch=${mismatch}`;
    case "unavailable":
      return "Eval reporting stack summary unavailable";
    default:
      return `Unknown readiness: ${readiness}`;
  }
}

function loadReleaseSummary(path) {
  try {
    if (!path || !fs.existsSync(path)) {
      return null;
    }
    return JSON.parse(fs.readFileSync(path, "utf-8"));
  } catch (_) {
    return null;
  }
}

async function postEvalReportingStatusCheck({ github, context, releaseSummaryPath }) {
  const summary = loadReleaseSummary(releaseSummaryPath);
  const readiness = summary ? String(summary.release_readiness || "unavailable") : "unavailable";
  const state = mapReadinessToState(readiness);
  const description = mapReadinessToDescription(readiness, summary);

  try {
    await github.rest.repos.createCommitStatus({
      owner: context.repo.owner,
      repo: context.repo.repo,
      sha: context.sha,
      state,
      description,
      context: "Eval Reporting",
    });
    console.log(`Posted status check: state=${state}, description=${description}`);
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    console.warn(`Status check skipped (fail-soft): ${message}`);
  }
}

module.exports = {
  mapReadinessToState,
  mapReadinessToDescription,
  loadReleaseSummary,
  postEvalReportingStatusCheck,
};
