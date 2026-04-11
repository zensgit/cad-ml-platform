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


def test_comment_markdown_utils_js_renders_tables_sections_and_footer() -> None:
    node_script = r"""
const mod = require("./scripts/ci/comment_markdown_utils.js");

const table = mod.markdownTable(
  ["Field", "Value"],
  [
    ["status", "ok"],
    ["count", 2],
  ],
);
const section = mod.markdownSection("Quick Actions", "- one\n- two");
const footer = mod.markdownFooter({ updatedAt: "2026-03-17 10:00:00", sha: "abcdef1" });

if (!table.includes("| Field | Value |")) {
  throw new Error("table header missing");
}
if (!table.includes("| status | ok |")) {
  throw new Error("table row missing");
}
if (!section.startsWith("### Quick Actions\n- one")) {
  throw new Error("section heading missing");
}
if (!footer.includes("*Updated: 2026-03-17 10:00:00 UTC*")) {
  throw new Error("footer updated line missing");
}
if (!footer.includes("*Commit: abcdef1*")) {
  throw new Error("footer commit line missing");
}
console.log("ok:comment-markdown-utils");
"""

    result = _run_node_inline(node_script)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok:comment-markdown-utils" in result.stdout
