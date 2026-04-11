from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_BIN = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE_BIN is None, reason="node is not available")


def _run_node_inline(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [NODE_BIN, "-e", script],  # type: ignore[list-item]
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_build_delivery_result_default_disabled() -> None:
    script = r"""
const mod = require("./scripts/ci/post_eval_reporting_webhook_delivery.js");
const r = mod.buildDeliveryResult({
  request: { release_readiness: "ready", delivery_allowed: true, webhook_event_type: "eval_reporting.updated" },
});
if (r.delivery_mode !== "disabled") throw new Error("expected disabled, got " + r.delivery_mode);
if (r.delivery_enabled !== false) throw new Error("expected delivery_enabled=false");
if (r.delivery_attempted !== false) throw new Error("expected delivery_attempted=false");
if (r.delivery_succeeded !== false) throw new Error("expected delivery_succeeded=false");
if (r.http_status !== null) throw new Error("expected http_status=null");
if (r.retry_recommended !== false) throw new Error("expected retry_recommended=false");
console.log("ok:default-disabled");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:default-disabled" in result.stdout


def test_build_delivery_result_deliver_mode_when_enabled_and_allowed() -> None:
    script = r"""
const mod = require("./scripts/ci/post_eval_reporting_webhook_delivery.js");
const r = mod.buildDeliveryResult({
  request: { release_readiness: "ready", delivery_allowed: true },
  deliveryEnabled: true,
  webhookUrl: "https://example.com/webhook",
});
if (r.delivery_mode !== "deliver") throw new Error("expected deliver, got " + r.delivery_mode);
if (r.delivery_allowed !== true) throw new Error("expected delivery_allowed=true");
console.log("ok:deliver-mode");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:deliver-mode" in result.stdout


def test_build_delivery_result_blocked_when_not_allowed() -> None:
    script = r"""
const mod = require("./scripts/ci/post_eval_reporting_webhook_delivery.js");
const r = mod.buildDeliveryResult({
  request: { release_readiness: "degraded", delivery_allowed: false },
  deliveryEnabled: true,
});
if (r.delivery_mode !== "blocked") throw new Error("expected blocked, got " + r.delivery_mode);
console.log("ok:blocked");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:blocked" in result.stdout


def test_delivery_result_has_required_fields() -> None:
    script = r"""
const mod = require("./scripts/ci/post_eval_reporting_webhook_delivery.js");
const r = mod.buildDeliveryResult({
  request: {
    release_readiness: "ready",
    delivery_allowed: true,
    delivery_target_kind: "external_webhook",
    webhook_event_type: "eval_reporting.updated",
    request_timeout_seconds: 30,
  },
});
const required = [
  "status", "surface_kind", "generated_at", "release_readiness",
  "delivery_enabled", "delivery_allowed", "delivery_attempted",
  "delivery_succeeded", "delivery_mode", "delivery_target_kind",
  "webhook_event_type", "http_status", "delivery_error",
  "retry_recommended", "retry_hint", "request_timeout_seconds",
];
for (const key of required) {
  if (!(key in r)) throw new Error("missing field: " + key);
}
if (r.surface_kind !== "eval_reporting_webhook_delivery_result") {
  throw new Error("wrong surface_kind: " + r.surface_kind);
}
console.log("ok:required-fields");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:required-fields" in result.stdout


def test_load_delivery_request_null_for_missing() -> None:
    script = r"""
const mod = require("./scripts/ci/post_eval_reporting_webhook_delivery.js");
const r = mod.loadDeliveryRequest("/tmp/nonexistent_delivery_99999.json");
if (r !== null) throw new Error("expected null");
console.log("ok:null");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:null" in result.stdout


def test_js_module_does_not_own_content_logic() -> None:
    source = (REPO_ROOT / "scripts" / "ci" / "post_eval_reporting_webhook_delivery.js").read_text(encoding="utf-8")
    forbidden = [
        "buildEvaluationReportCommentBody",
        "generateHtml",
        "buildWeekly",
        "plotTrend",
        "materializeBundle",
        "generate_eval_reporting_webhook_export",
        "generate_eval_reporting_dashboard_payload",
    ]
    for name in forbidden:
        assert name not in source, f"JS module must not reference {name}"
