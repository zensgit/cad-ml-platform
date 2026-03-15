"use strict";

const fs = require("fs");

function envStr(name, fallback = "") {
  const value = process.env[name];
  if (value === undefined || value === null) {
    return String(fallback);
  }
  return String(value);
}

function parsePositiveInt(raw) {
  const parsed = parseInt(String(raw === undefined || raw === null ? "" : raw), 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null;
  }
  return parsed;
}

function normalizeAttemptValue(value, fallback) {
  if (value === undefined || value === null || value === "") {
    return fallback;
  }
  return String(value);
}

function buildAttemptLines(attempts) {
  if (!Array.isArray(attempts) || attempts.length === 0) {
    return ["- attempts: none"];
  }
  return attempts.map((attempt, index) => {
    const item = attempt && typeof attempt === "object" ? attempt : {};
    const attemptNo = normalizeAttemptValue(item.attempt, index + 1);
    const dispatchExit = normalizeAttemptValue(item.dispatch_exit_code, "n/a");
    const markerOk = normalizeAttemptValue(item.soft_marker_ok, "n/a");
    return `- attempt ${attemptNo}: dispatch_exit_code=${dispatchExit}, soft_marker_ok=${markerOk}`;
  });
}

async function commentSoftModeSmokePR({ github, context, process }) {
  try {
    const summaryPath = envStr("SOFT_SMOKE_SUMMARY_JSON", "").trim();
    if (!summaryPath) {
      console.warn("PR comment skipped: SOFT_SMOKE_SUMMARY_JSON is empty");
      return;
    }
    if (!fs.existsSync(summaryPath)) {
      console.warn(`PR comment skipped: summary file not found at ${summaryPath}`);
      return;
    }

    let summary;
    try {
      summary = JSON.parse(fs.readFileSync(summaryPath, "utf8"));
    } catch (error) {
      const message = error && error.message ? error.message : String(error);
      console.warn(`PR comment skipped: failed to parse summary JSON: ${message}`);
      return;
    }

    const triggerPr = parsePositiveInt(envStr("SOFT_SMOKE_TRIGGER_PR", ""));
    const contextPr = parsePositiveInt(context && context.issue ? context.issue.number : "");
    const issueNumber = triggerPr || contextPr;
    if (!issueNumber) {
      console.warn("PR comment skipped: no valid PR number");
      return;
    }

    const dispatch = summary && typeof summary.dispatch === "object" ? summary.dispatch : {};
    const attempts = Array.isArray(summary.attempts) ? summary.attempts : [];
    const attemptLines = buildAttemptLines(attempts).join("\n");
    const runId = normalizeAttemptValue(dispatch.run_id, "");
    const runUrl = normalizeAttemptValue(dispatch.run_url, "");

    const body = `## CAD ML Platform - Soft Mode Smoke

| Field | Value |
|---|---|
| overall_exit_code | ${normalizeAttemptValue(summary.overall_exit_code, "n/a")} |
| soft_marker_ok | ${normalizeAttemptValue(summary.soft_marker_ok, "n/a")} |
| restore_ok | ${normalizeAttemptValue(summary.restore_ok, "n/a")} |
| run_id | ${runId || "n/a"} |
| run_url | ${runUrl || "n/a"} |
| attempts_total | ${attempts.length} |

### Attempts
${attemptLines}

*Updated: ${new Date().toISOString().replace("T", " ").substring(0, 19)} UTC*
*Commit: ${context.sha.substring(0, 7)}*`;

    const { data: comments } = await github.rest.issues.listComments({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: issueNumber,
    });

    const botComment = comments.find(
      (comment) =>
        comment.user.type === "Bot" &&
        String(comment.body || "").includes("CAD ML Platform - Soft Mode Smoke"),
    );

    if (botComment) {
      await github.rest.issues.updateComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        comment_id: botComment.id,
        body,
      });
      return;
    }

    await github.rest.issues.createComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: issueNumber,
      body,
    });
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    console.warn(`PR comment skipped: ${message}`);
  }
}

module.exports = {
  commentSoftModeSmokePR,
};
