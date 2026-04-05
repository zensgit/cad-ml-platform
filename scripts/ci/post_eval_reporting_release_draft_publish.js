"use strict";

const fs = require("fs");
const path = require("path");

function loadDashboardPayload(filePath) {
  try {
    if (!filePath || !fs.existsSync(filePath)) return null;
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch (_) {
    return null;
  }
}

function buildPublishResult({
  payload,
  publishEnabled = false,
}) {
  const p = payload && typeof payload === "object" ? payload : {};

  const readiness = String(p.release_readiness || "unavailable").trim();
  const headline = String(p.dashboard_headline || "").trim() || `readiness=${readiness}`;
  const landingUrl = String(p.public_landing_page_url || "").trim();
  const staticUrl = String(p.public_static_report_url || "").trim();
  const interactiveUrl = String(p.public_interactive_report_url || "").trim();

  // Derive draft title and body internally (absorbs former draft_payload logic)
  const draftTitle = `Eval Reporting — ${readiness}`;
  const bodyLines = [
    "## Eval Reporting",
    "",
    `- Release readiness: **${readiness}**`,
    `- Headline: ${headline}`,
    "",
    "### Reports",
    "",
  ];
  bodyLines.push(landingUrl ? `- Landing Page: [${landingUrl}](${landingUrl})` : "- Landing Page: n/a");
  bodyLines.push(staticUrl ? `- Static Report: [${staticUrl}](${staticUrl})` : "- Static Report: n/a");
  bodyLines.push(interactiveUrl ? `- Interactive Report: [${interactiveUrl}](${interactiveUrl})` : "- Interactive Report: n/a");
  bodyLines.push("");
  const draftBody = bodyLines.join("\n") + "\n";

  const tag = `eval-report-v${new Date().toISOString().substring(0, 10)}`;
  const publishAllowed = readiness === "ready" && publishEnabled;
  const publishMode = publishEnabled
    ? (publishAllowed ? "publish" : "blocked")
    : "disabled";

  return {
    status: readiness,
    surface_kind: "eval_reporting_release_draft_publish_result",
    generated_at: new Date().toISOString().replace("T", " ").substring(0, 19) + "Z",
    release_readiness: readiness,
    publish_enabled: publishEnabled,
    publish_allowed: publishAllowed,
    publish_attempted: false,
    publish_succeeded: false,
    publish_mode: publishMode,
    github_release_tag: tag,
    github_release_id: null,
    _draft_title: draftTitle,
    _draft_body: draftBody,
    _result_markdown: [
      "# Eval Reporting Release Draft Publish Result",
      "",
      `- Publish Attempted: false`,
      `- Publish Succeeded: false`,
      `- Publish Mode: ${publishMode}`,
      `- Publish Enabled: ${publishEnabled}`,
      `- Release readiness: **${readiness}**`,
      `- GitHub Release Tag: \`${tag}\``,
      "",
    ].join("\n") + "\n",
  };
}

async function postReleaseDraftPublish({
  github,
  context,
  dashboardPayloadPath,
  publishEnabled = false,
  outputJsonPath = "reports/ci/eval_reporting_release_draft_publish_result.json",
  outputMdPath = "reports/ci/eval_reporting_release_draft_publish_result.md",
}) {
  const payload = loadDashboardPayload(dashboardPayloadPath);
  const result = buildPublishResult({ payload, publishEnabled });

  const outDir = path.dirname(outputJsonPath);
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  if (result.publish_allowed && result.publish_enabled) {
    result.publish_attempted = true;
    try {
      const release = await github.rest.repos.createRelease({
        owner: context.repo.owner,
        repo: context.repo.repo,
        tag_name: result.github_release_tag,
        name: result._draft_title,
        body: result._draft_body,
        draft: true,
      });
      result.publish_succeeded = true;
      result.github_release_id = release.data.id;
      console.log(`Draft release created: tag=${result.github_release_tag}, id=${release.data.id}`);
    } catch (error) {
      const message = error && error.message ? error.message : String(error);
      console.warn(`Draft release publish failed (fail-soft): ${message}`);
      result.publish_succeeded = false;
    }
  } else {
    console.log(`Publish skipped: mode=${result.publish_mode}, enabled=${result.publish_enabled}`);
  }

  result._result_markdown = [
    "# Eval Reporting Release Draft Publish Result",
    "",
    `- Publish Attempted: ${result.publish_attempted}`,
    `- Publish Succeeded: ${result.publish_succeeded}`,
    `- Publish Mode: ${result.publish_mode}`,
    `- Publish Enabled: ${result.publish_enabled}`,
    `- Release readiness: **${result.release_readiness}**`,
    `- GitHub Release Tag: \`${result.github_release_tag}\``,
    `- GitHub Release ID: ${result.github_release_id || "n/a"}`,
    "",
  ].join("\n") + "\n";

  const output = { ...result };
  delete output._draft_title;
  delete output._draft_body;
  delete output._result_markdown;

  fs.writeFileSync(outputJsonPath, JSON.stringify(output, null, 2), "utf-8");
  fs.writeFileSync(outputMdPath, result._result_markdown, "utf-8");

  return output;
}

module.exports = {
  loadDashboardPayload,
  buildPublishResult,
  postReleaseDraftPublish,
};
