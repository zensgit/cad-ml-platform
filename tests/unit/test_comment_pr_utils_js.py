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


def test_comment_pr_utils_js_updates_matching_bot_comment() -> None:
    node_script = r"""
const mod = require("./scripts/ci/comment_pr_utils.js");

let updatedPayload = null;
let createdPayload = null;
let listPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async (payload) => {
        listPayload = payload;
        return {
          data: [
            { id: 7, user: { type: "User" }, body: "CAD ML Platform - Evaluation Results" },
            { id: 42, user: { type: "Bot" }, body: "## CAD ML Platform - Evaluation Results\n\nold" },
          ],
        };
      },
      updateComment: async (payload) => {
        updatedPayload = payload;
        return { data: { id: payload.comment_id } };
      },
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 99 } };
      },
    },
  },
};

(async () => {
  const result = await mod.upsertBotIssueComment({
    github,
    owner: "zensgit",
    repo: "cad-ml-platform",
    issueNumber: 123,
    body: "new-body",
    marker: "CAD ML Platform - Evaluation Results",
  });
  if (!listPayload || listPayload.per_page !== 100) {
    throw new Error("expected default per_page=100");
  }
  if (!updatedPayload || updatedPayload.comment_id !== 42 || updatedPayload.body !== "new-body") {
    throw new Error("expected updateComment to target matching bot comment");
  }
  if (createdPayload) {
    throw new Error("createComment should not be called when update path matches");
  }
  if (!result || result.action !== "updated" || result.commentId !== 42) {
    throw new Error("result metadata should report updated comment");
  }
  console.log("ok:update");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:update" in result.stdout


def test_comment_pr_utils_js_creates_when_no_matching_bot_comment() -> None:
    node_script = r"""
const mod = require("./scripts/ci/comment_pr_utils.js");

let updatedPayload = null;
let createdPayload = null;
const github = {
  rest: {
    issues: {
      listComments: async () => ({
        data: [
          { id: 8, user: { type: "User" }, body: "CAD ML Platform - Soft Mode Smoke" },
          { id: 9, user: { type: "Bot" }, body: "unrelated marker" },
        ],
      }),
      updateComment: async (payload) => {
        updatedPayload = payload;
        return { data: { id: payload.comment_id } };
      },
      createComment: async (payload) => {
        createdPayload = payload;
        return { data: { id: 1234 } };
      },
    },
  },
};

(async () => {
  const result = await mod.upsertBotIssueComment({
    github,
    owner: "zensgit",
    repo: "cad-ml-platform",
    issueNumber: 456,
    body: "fresh-body",
    marker: "CAD ML Platform - Soft Mode Smoke",
    perPage: 50,
  });
  if (updatedPayload) {
    throw new Error("updateComment should not be called when marker does not match");
  }
  if (!createdPayload || createdPayload.issue_number !== 456 || createdPayload.body !== "fresh-body") {
    throw new Error("expected createComment payload");
  }
  if (!result || result.action !== "created" || result.commentId !== 1234) {
    throw new Error("result metadata should report created comment");
  }
  console.log("ok:create");
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
"""

    result = _run_node_inline(node_script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:create" in result.stdout
