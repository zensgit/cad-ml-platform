"use strict";

const fs = require("fs");
const {
  markdownFooter,
  markdownSection,
  markdownTable,
} = require("./comment_markdown_utils.js");
const { upsertBotIssueComment } = require("./comment_pr_utils.js");

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
    const message = normalizeAttemptValue(
      item.soft_marker_message !== undefined ? item.soft_marker_message : item.message,
      "n/a",
    );
    return `- attempt ${attemptNo}: dispatch_exit_code=${dispatchExit}, soft_marker_ok=${markerOk}, message=${message}`;
  });
}

function buildSoftModeSmokeCommentBody({ summary, title, commitSha, updatedAt }) {
  const summaryObj = summary && typeof summary === "object" ? summary : {};
  const dispatch =
    summaryObj.dispatch && typeof summaryObj.dispatch === "object" ? summaryObj.dispatch : {};
  const attempts = Array.isArray(summaryObj.attempts) ? summaryObj.attempts : [];
  const attemptLines = buildAttemptLines(attempts).join("\n");
  const runId = normalizeAttemptValue(dispatch.run_id, "n/a");
  const runUrl = normalizeAttemptValue(dispatch.run_url, "n/a");
  const shortSha = normalizeAttemptValue(commitSha, "n/a").substring(0, 7) || "n/a";

  return [
    `## ${normalizeAttemptValue(title, "CAD ML Platform - Soft Mode Smoke")}`,
    "",
    markdownTable(
      ["Field", "Value"],
      [
        ["overall_exit_code", normalizeAttemptValue(summaryObj.overall_exit_code, "n/a")],
        ["dispatch_exit_code", normalizeAttemptValue(summaryObj.dispatch_exit_code, "n/a")],
        ["soft_marker_ok", normalizeAttemptValue(summaryObj.soft_marker_ok, "n/a")],
        ["restore_ok", normalizeAttemptValue(summaryObj.restore_ok, "n/a")],
        ["run_id", runId],
        ["run_url", runUrl],
        ["attempts_total", attempts.length],
      ],
    ),
    "",
    markdownSection("Attempts", attemptLines),
    "",
    markdownFooter({
      updatedAt: normalizeAttemptValue(updatedAt, ""),
      sha: shortSha,
    }),
  ].join("\n");
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

    const body = buildSoftModeSmokeCommentBody({
      summary,
      title: "CAD ML Platform - Soft Mode Smoke",
      commitSha: context.sha,
      updatedAt: new Date().toISOString().replace("T", " ").substring(0, 19),
    });

    await upsertBotIssueComment({
      github,
      owner: context.repo.owner,
      repo: context.repo.repo,
      issueNumber,
      body,
      marker: "CAD ML Platform - Soft Mode Smoke",
    });
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    console.warn(`PR comment skipped: ${message}`);
  }
}

module.exports = {
  buildSoftModeSmokeCommentBody,
  commentSoftModeSmokePR,
};
