from __future__ import annotations

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


def test_issue_upsert_utils_js_updates_matching_open_issue() -> None:
    node_script = r"""
const mod = require("./scripts/ci/issue_upsert_utils.js");

let listPayload = null;
let updatePayload = null;
let createPayload = null;
const github = {
  rest: {
    issues: {
      listForRepo: async (payload) => {
        listPayload = payload;
        return {
          data: [
            { number: 4, state: "closed", body: "<!-- ci:test --> closed" },
            { number: 7, state: "open", body: "<!-- ci:test --> old-body" },
          ],
        };
      },
      update: async (payload) => {
        updatePayload = payload;
        return { data: { number: payload.issue_number } };
      },
      create: async (payload) => {
        createPayload = payload;
        return { data: { number: 99 } };
      },
    },
  },
};

(async () => {
  const result = await mod.upsertOpenIssue({
    github,
    owner: "zensgit",
    repo: "cad-ml-platform",
    title: "Issue title",
    body: "<!-- ci:test --> new-body",
    labels: ["alert", "performance"],
    marker: "<!-- ci:test -->",
    listLabels: "alert,performance",
  });
  if (!listPayload || listPayload.labels !== "alert,performance" || listPayload.per_page !== 100) {
    throw new Error("expected listForRepo payload with labels and default per_page");
  }
  if (!updatePayload || updatePayload.issue_number !== 7 || updatePayload.title !== "Issue title") {
    throw new Error("expected update on matching open issue");
  }
  if (!Array.isArray(updatePayload.labels) || updatePayload.labels.length !== 2) {
    throw new Error("expected labels on update payload");
  }
  if (createPayload) {
    throw new Error("create should not be called when open matching issue exists");
  }
  if (!result || result.action !== "updated" || result.issueNumber !== 7) {
    throw new Error("expected updated result metadata");
  }
  console.log("ok:update-issue");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:update-issue" in result.stdout


def test_issue_upsert_utils_js_creates_when_no_matching_open_issue() -> None:
    node_script = r"""
const mod = require("./scripts/ci/issue_upsert_utils.js");

let updatePayload = null;
let createPayload = null;
const github = {
  rest: {
    issues: {
      listForRepo: async () => ({
        data: [
          { number: 3, state: "open", body: "different marker" },
        ],
      }),
      update: async (payload) => {
        updatePayload = payload;
        return { data: { number: payload.issue_number } };
      },
      create: async (payload) => {
        createPayload = payload;
        return { data: { number: 123 } };
      },
    },
  },
};

(async () => {
  const result = await mod.upsertOpenIssue({
    github,
    owner: "zensgit",
    repo: "cad-ml-platform",
    title: "Fresh issue",
    body: "<!-- ci:fresh --> body",
    labels: ["badge-review"],
    marker: "<!-- ci:fresh -->",
    listLabels: "badge-review",
    perPage: 50,
  });
  if (updatePayload) {
    throw new Error("update should not be called when no matching issue exists");
  }
  if (!createPayload || createPayload.title !== "Fresh issue") {
    throw new Error("expected create payload");
  }
  if (!Array.isArray(createPayload.labels) || createPayload.labels[0] !== "badge-review") {
    throw new Error("expected labels on create payload");
  }
  if (!result || result.action !== "created" || result.issueNumber !== 123) {
    throw new Error("expected created result metadata");
  }
  console.log("ok:create-issue");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create-issue" in result.stdout
