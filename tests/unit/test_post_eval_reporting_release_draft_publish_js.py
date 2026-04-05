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


def test_build_publish_result_default_disabled() -> None:
    script = r"""
const mod = require("./scripts/ci/post_eval_reporting_release_draft_publish.js");
const r = mod.buildPublishResult({
  payload: { release_readiness: "ready", publish_allowed: true, github_release_tag: "v1" },
});
if (r.publish_mode !== "disabled") throw new Error("expected disabled, got " + r.publish_mode);
if (r.publish_enabled !== false) throw new Error("expected publish_enabled=false");
if (r.publish_attempted !== false) throw new Error("expected publish_attempted=false");
if (r.publish_succeeded !== false) throw new Error("expected publish_succeeded=false");
if (r.github_release_id !== null) throw new Error("expected github_release_id=null");
console.log("ok:default-disabled");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:default-disabled" in result.stdout


def test_build_publish_result_publish_mode_when_enabled_and_allowed() -> None:
    script = r"""
const mod = require("./scripts/ci/post_eval_reporting_release_draft_publish.js");
const r = mod.buildPublishResult({
  payload: { release_readiness: "ready", publish_allowed: true, github_release_tag: "v1" },
  publishEnabled: true,
});
if (r.publish_mode !== "publish") throw new Error("expected publish, got " + r.publish_mode);
if (r.publish_allowed !== true) throw new Error("expected publish_allowed=true");
console.log("ok:publish-mode");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:publish-mode" in result.stdout


def test_build_publish_result_blocked_when_not_allowed() -> None:
    script = r"""
const mod = require("./scripts/ci/post_eval_reporting_release_draft_publish.js");
const r = mod.buildPublishResult({
  payload: { release_readiness: "degraded", publish_allowed: false, github_release_tag: "v1" },
  publishEnabled: true,
});
if (r.publish_mode !== "blocked") throw new Error("expected blocked, got " + r.publish_mode);
console.log("ok:blocked");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:blocked" in result.stdout


def test_publish_result_has_required_fields() -> None:
    script = r"""
const mod = require("./scripts/ci/post_eval_reporting_release_draft_publish.js");
const r = mod.buildPublishResult({
  payload: { release_readiness: "ready", github_release_tag: "v1", draft_title: "T", draft_body_markdown: "B" },
});
const required = [
  "status", "surface_kind", "generated_at", "release_readiness",
  "publish_enabled", "publish_allowed", "publish_attempted",
  "publish_succeeded", "publish_mode", "github_release_tag", "github_release_id",
];
for (const key of required) {
  if (!(key in r)) throw new Error("missing field: " + key);
}
if (r.surface_kind !== "eval_reporting_release_draft_publish_result") {
  throw new Error("wrong surface_kind: " + r.surface_kind);
}
const md = r._result_markdown;
if (!md.includes("Publish Attempted")) throw new Error("md missing Publish Attempted");
if (!md.includes("Publish Succeeded")) throw new Error("md missing Publish Succeeded");
if (!md.includes("Publish Mode")) throw new Error("md missing Publish Mode");
if (!md.includes("Release readiness")) throw new Error("md missing Release readiness");
if (!md.includes("GitHub Release Tag")) throw new Error("md missing GitHub Release Tag");
console.log("ok:required-fields");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:required-fields" in result.stdout


def test_load_publish_payload_null_for_missing() -> None:
    script = r"""
const mod = require("./scripts/ci/post_eval_reporting_release_draft_publish.js");
const p = mod.loadDashboardPayload("/tmp/nonexistent_publish_99999.json");
if (p !== null) throw new Error("expected null");
console.log("ok:null");
"""
    result = _run_node_inline(script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:null" in result.stdout


def test_js_module_does_not_own_content_logic() -> None:
    source = (REPO_ROOT / "scripts" / "ci" / "post_eval_reporting_release_draft_publish.js").read_text(encoding="utf-8")
    forbidden = [
        "buildEvaluationReportCommentBody",
        "generateHtml",
        "buildWeekly",
        "plotTrend",
        "materializeBundle",
        "generate_eval_reporting_release_draft_prefill",
        "generate_eval_reporting_release_note_snippet",
        "generate_eval_reporting_dashboard_payload",
        "generate_eval_reporting_release_draft_publish_payload",
    ]
    for name in forbidden:
        assert name not in source, f"JS module must not reference {name}"
