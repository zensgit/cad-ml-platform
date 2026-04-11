"use strict";

const fs = require("fs");
const path = require("path");

function loadDeliveryRequest(filePath) {
  try {
    if (!filePath || !fs.existsSync(filePath)) return null;
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch (_) {
    return null;
  }
}

function buildDeliveryResult({
  request,
  deliveryEnabled = false,
  webhookUrl = "",
}) {
  const r = request && typeof request === "object" ? request : {};

  const readiness = String(r.release_readiness || "unavailable").trim();
  const deliveryAllowed = Boolean(r.delivery_allowed) && deliveryEnabled;
  const deliveryMode = deliveryEnabled
    ? (deliveryAllowed ? "deliver" : "blocked")
    : "disabled";
  const eventType = String(r.webhook_event_type || "eval_reporting.updated").trim();
  const targetKind = String(r.delivery_target_kind || "external_webhook").trim();
  const timeoutSeconds = Number(r.request_timeout_seconds) || 30;

  return {
    status: readiness,
    surface_kind: "eval_reporting_webhook_delivery_result",
    generated_at: new Date().toISOString().replace("T", " ").substring(0, 19) + "Z",
    release_readiness: readiness,
    delivery_enabled: deliveryEnabled,
    delivery_allowed: deliveryAllowed,
    delivery_attempted: false,
    delivery_succeeded: false,
    delivery_mode: deliveryMode,
    delivery_target_kind: targetKind,
    webhook_event_type: eventType,
    http_status: null,
    delivery_error: null,
    retry_recommended: false,
    retry_hint: "n/a",
    request_timeout_seconds: timeoutSeconds,
    _request_body_json: String(r.request_body_json || "{}"),
    _webhook_url: webhookUrl,
  };
}

async function postWebhookDelivery({
  deliveryRequestPath,
  deliveryEnabled = false,
  webhookUrl = "",
  outputJsonPath = "reports/ci/eval_reporting_webhook_delivery_result.json",
  outputMdPath = "reports/ci/eval_reporting_webhook_delivery_result.md",
}) {
  const request = loadDeliveryRequest(deliveryRequestPath);
  const result = buildDeliveryResult({ request, deliveryEnabled, webhookUrl });

  const outDir = path.dirname(outputJsonPath);
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  if (result.delivery_allowed && result.delivery_enabled && result._webhook_url) {
    result.delivery_attempted = true;
    try {
      const https = require("https");
      const http = require("http");
      const url = new URL(result._webhook_url);
      const mod = url.protocol === "https:" ? https : http;
      const body = result._request_body_json;

      const statusCode = await new Promise((resolve, reject) => {
        const req = mod.request(
          url,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "Content-Length": Buffer.byteLength(body),
            },
            timeout: result.request_timeout_seconds * 1000,
          },
          (res) => resolve(res.statusCode),
        );
        req.on("error", reject);
        req.on("timeout", () => { req.destroy(); reject(new Error("timeout")); });
        req.write(body);
        req.end();
      });

      result.http_status = statusCode;
      result.delivery_succeeded = statusCode >= 200 && statusCode < 300;
      if (!result.delivery_succeeded) {
        result.delivery_error = `HTTP ${statusCode}`;
        result.retry_recommended = statusCode >= 500;
        result.retry_hint = statusCode >= 500 ? "server_error_retry" : "client_error_no_retry";
      }
      console.log(`Webhook delivery: status=${statusCode}, succeeded=${result.delivery_succeeded}`);
    } catch (error) {
      const message = error && error.message ? error.message : String(error);
      result.delivery_error = message;
      result.retry_recommended = true;
      result.retry_hint = "network_or_timeout_retry";
      console.warn(`Webhook delivery failed (fail-soft): ${message}`);
    }
  } else {
    console.log(`Delivery skipped: mode=${result.delivery_mode}, enabled=${result.delivery_enabled}, url=${result._webhook_url ? "set" : "empty"}`);
  }

  // Build markdown
  const md = [
    "# Eval Reporting Webhook Delivery Result",
    "",
    `- Delivery Attempted: ${result.delivery_attempted}`,
    `- Delivery Succeeded: ${result.delivery_succeeded}`,
    `- Delivery Mode: ${result.delivery_mode}`,
    `- HTTP Status: ${result.http_status || "n/a"}`,
    `- Retry Recommended: ${result.retry_recommended}`,
    `- Retry Hint: ${result.retry_hint}`,
    `- Release readiness: **${result.release_readiness}**`,
    `- Webhook Event: \`${result.webhook_event_type}\``,
    "",
  ].join("\n") + "\n";

  // Write artifacts (strip internal fields)
  const output = { ...result };
  delete output._request_body_json;
  delete output._webhook_url;

  fs.writeFileSync(outputJsonPath, JSON.stringify(output, null, 2), "utf-8");
  fs.writeFileSync(outputMdPath, md, "utf-8");

  return output;
}

module.exports = {
  loadDeliveryRequest,
  buildDeliveryResult,
  postWebhookDelivery,
};
