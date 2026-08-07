"""Durable regression guards for public-repo self-hosted runner security.

Parses **every** workflow under ``.github/workflows/`` dynamically (no hand-written
workflow allowlist). Policy is **fail-closed**:

1. ``runs-on`` may be only:
   - a literal label from the explicit approved GitHub-hosted set
     (currently ``ubuntu-latest`` only — not ``ubuntu-*`` prefix matching), or
   - the exact normalized canonical trusted CADML expression.
   Custom labels (e.g. ``ubuntu-private``) can be self-hosted runner tags and
   are rejected. Every other dynamic expression is rejected until policy is
   deliberately extended.
2. No ``pull_request*`` path may select self-hosted (canonical CADML evaluates
   to hosted under PR contexts; non-canonical is rejected outright).
3. ``cancel-in-progress: true`` concurrency must use the exact normalized
   canonical event-scoped group (not mere token presence).
4. ``uvnet-inspector-gate`` is skip-proof via **exact canonical critical steps**
   (dedicated torch install, torch import assertion, inspector pytest): no job
   ``if`` / ``needs`` / ``continue-on-error`` key, no workflow/job
   ``defaults.run.shell``, no step ``if`` / ``shell`` / ``continue-on-error``
   key (key must be absent — not merely a known-false spelling), and ``run``
   must equal the standalone command exactly. Workflow must have a positive
   ``pull_request`` trigger contract: mapping or null/empty (not activity
   sequence/scalar shorthand); no ``types`` / ``branches-ignore`` / ``paths`` /
   ``paths-ignore``; if ``branches`` is set it must list ``main``. Ordinary
   dependency install is separate.

These invariants exist because this is a **public** repository with persistent
repo-level self-hosted runners (``CADML_LINUX_RUNNER=cad-ml``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest
import yaml

# Repo standard is Python 3.10+ (Agents.md). System python3 on some dev hosts is
# still 3.9; collecting the suite there fails deep in FastAPI route annotations
# (str | None) with hundreds of cryptic ERRORs. Fail fast with a clear signal.
if sys.version_info < (3, 10):  # pragma: no cover - environment guard
    raise RuntimeError(
        "test_workflow_runner_security.py requires Python >= 3.10 "
        f"(got {sys.version_info.major}.{sys.version_info.minor}). "
        "Use the repo CI interpreter (3.11) or: python3.11 -m pytest ..."
    )

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"

# ---------------------------------------------------------------------------
# Canonical policy strings (exact normalized forms used in this repo)
# ---------------------------------------------------------------------------

# Trusted fail-closed self-hosted selection: main push / schedule / dispatch@main.
# Whitespace is normalized before comparison — do not "approximately" match.
CANONICAL_TRUSTED_CADML_RUNS_ON = (
    "${{ ((github.event_name == 'push' && github.ref == 'refs/heads/main') || "
    "(github.event_name == 'schedule') || "
    "(github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main')) && "
    "(vars.CADML_LINUX_RUNNER || 'ubuntu-latest') || 'ubuntu-latest' }}"
)

# Event-scoped cancel group: PR number stable identity; run_id unique non-PR fallback.
CANONICAL_CANCEL_CONCURRENCY_GROUP = (
    "${{ github.workflow }}-${{ github.event_name }}-"
    "${{ github.event.pull_request.number || github.run_id }}"
)

# Explicit allowlist of GitHub-hosted runner labels used by this repo today.
# Self-hosted runners may register arbitrary custom labels (including names that
# look like GH-hosted prefixes such as "ubuntu-private"); prefix/regex matching
# is therefore not proof of GitHub-hosted. New labels require deliberate policy
# extension here — do not broaden to ubuntu-*/windows-*/macos-* patterns.
APPROVED_GITHUB_HOSTED_LABELS = frozenset({"ubuntu-latest"})

# Canonical uvnet-inspector-gate security-proof steps (exact run equality).
# Ordinary dependency install is intentionally NOT among these three.
CANONICAL_UVNET_TORCH_INSTALL_CMD = 'python -m pip install "torch==2.1.0"'
CANONICAL_UVNET_TORCH_IMPORT_CMD = (
    "python -c \"import torch; print('torch', torch.__version__)\""
)
CANONICAL_UVNET_INSPECTOR_PYTEST_CMD = (
    "python -m pytest tests/unit/test_uvnet_checkpoint_inspect.py -v -rs"
)

# (step name, exact run command) — order is not required; names are fixed.
CANONICAL_UVNET_CRITICAL_STEPS = (
    ("Install pinned torch (skip-proof)", CANONICAL_UVNET_TORCH_INSTALL_CMD),
    (
        "Assert torch importable (inspector test must execute, not skip)",
        CANONICAL_UVNET_TORCH_IMPORT_CMD,
    ),
    (
        "Run uvnet checkpoint inspector tests (skip-proof)",
        CANONICAL_UVNET_INSPECTOR_PYTEST_CMD,
    ),
)


# ---------------------------------------------------------------------------
# Workflow discovery / loading (dynamic — no allowlist)
# ---------------------------------------------------------------------------


def _iter_workflow_paths(workflows_dir: Optional[Path] = None) -> List[Path]:
    """Discover every workflow file. GitHub accepts both ``.yml`` and ``.yaml``.

    Both extensions are required — dropping either is a fail-open for files the
    other glob would miss. No hand-written name allowlist.
    """
    root = workflows_dir if workflows_dir is not None else WORKFLOWS_DIR
    paths = sorted(root.glob("*.yml")) + sorted(root.glob("*.yaml"))
    assert paths, f"expected workflows under {root}"
    return paths


def _load_workflow(path: Path) -> Dict[str, Any]:
    # BaseLoader keeps ${{ ... }} expressions as plain strings (no YAML tags).
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(data, dict), f"{path.name}: workflow root must be a mapping"
    return data


def _all_workflows() -> List[Tuple[Path, Dict[str, Any]]]:
    return [(p, _load_workflow(p)) for p in _iter_workflow_paths()]


def _workflow_ids() -> List[str]:
    return [p.name for p in _iter_workflow_paths()]


def _jobs(workflow: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    jobs = workflow.get("jobs") or {}
    assert isinstance(jobs, dict)
    for name, job in jobs.items():
        if isinstance(job, dict):
            yield name, job


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_as_str(v) for v in value)
    return str(value)


def _normalize_policy_text(text: str) -> str:
    """Collapse whitespace for exact policy comparison (expressions may wrap)."""
    return re.sub(r"\s+", " ", text.strip())


def _unwrap_expression(raw: str) -> str:
    """Strip a single outer ${{ ... }} if present; leave bare strings alone."""
    s = raw.strip()
    if s.startswith("${{") and s.endswith("}}"):
        return s[3:-2].strip()
    return s


# ---------------------------------------------------------------------------
# Minimal GitHub Actions expression evaluator (canonical CADML only)
# ---------------------------------------------------------------------------


class _ExprError(ValueError):
    pass


def _eval_github_expr(expr: str, ctx: Dict[str, Any]) -> Any:
    """Evaluate a restricted subset of GitHub Actions expression language."""
    s = expr.strip()
    pos = 0

    def peek() -> str:
        nonlocal pos
        while pos < len(s) and s[pos].isspace():
            pos += 1
        return s[pos] if pos < len(s) else ""

    def parse_primary() -> Any:
        nonlocal pos
        ch = peek()
        if ch == "(":
            pos += 1
            val = parse_or()
            if peek() != ")":
                raise _ExprError(f"expected ')' in {expr!r}")
            pos += 1
            return val
        if ch in ("'", '"'):
            quote = ch
            pos += 1
            start = pos
            while pos < len(s) and s[pos] != quote:
                pos += 1
            if pos >= len(s):
                raise _ExprError(f"unterminated string in {expr!r}")
            val = s[start:pos]
            pos += 1
            return val
        start = pos
        while pos < len(s) and (s[pos].isalnum() or s[pos] in "._"):
            pos += 1
        token = s[start:pos]
        if not token:
            raise _ExprError(f"unexpected char {ch!r} in {expr!r}")
        if token == "true":
            return True
        if token == "false":
            return False
        if token == "null":
            return None
        parts = token.split(".")
        cur: Any = ctx
        for part in parts:
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    def parse_comparison() -> Any:
        nonlocal pos
        left = parse_primary()
        peek()  # skip whitespace before operator
        if s[pos : pos + 2] == "==":
            pos += 2
            right = parse_primary()
            return left == right
        if s[pos : pos + 2] == "!=":
            pos += 2
            right = parse_primary()
            return left != right
        return left

    def _truthy(v: Any) -> bool:
        return bool(v)

    def parse_and() -> Any:
        nonlocal pos
        left = parse_comparison()
        while True:
            save = pos
            if peek() == "&" and s[pos : pos + 2] == "&&":
                pos += 2
                right = parse_comparison()
                left = right if _truthy(left) else left
            else:
                pos = save
                return left

    def parse_or() -> Any:
        nonlocal pos
        left = parse_and()
        while True:
            save = pos
            if peek() == "|" and s[pos : pos + 2] == "||":
                pos += 2
                right = parse_and()
                left = left if _truthy(left) else right
            else:
                pos = save
                return left

    result = parse_or()
    if peek():
        raise _ExprError(f"trailing input in {expr!r} at {pos}")
    return result


def _is_hosted_label(label: str) -> bool:
    """True only for labels in APPROVED_GITHUB_HOSTED_LABELS (exact match)."""
    label = label.strip()
    if not label or "${{" in label:
        return False
    return label in APPROVED_GITHUB_HOSTED_LABELS


def _is_self_hosted_resolved(resolved: str) -> bool:
    """True when a *successfully resolved* label is non-hosted.

    Empty / unknown is treated as unsafe by callers via policy classification —
    this helper never certifies empty as safe.
    """
    resolved = resolved.strip()
    if not resolved:
        return True  # fail-closed: unresolved is not "hosted"
    if _is_hosted_label(resolved):
        return False
    tokens = resolved.split()
    if len(tokens) > 1:
        return any(not _is_hosted_label(t) for t in tokens)
    return True


# ---------------------------------------------------------------------------
# Fail-closed runs-on policy
# ---------------------------------------------------------------------------


def _is_canonical_trusted_cadml(runs_on_raw: str) -> bool:
    return _normalize_policy_text(runs_on_raw) == _normalize_policy_text(
        CANONICAL_TRUSTED_CADML_RUNS_ON
    )


def _is_literal_hosted_runs_on(raw: Any) -> bool:
    if isinstance(raw, list):
        # Multi-label form is never a pure GitHub-hosted single label in this repo.
        # Require every token to be a hosted label (no self-hosted multi-label).
        if not raw:
            return False
        return all(_is_literal_hosted_runs_on(item) for item in raw)
    text = _as_str(raw).strip()
    if not text or "${{" in text:
        return False
    return _is_hosted_label(text)


def _assert_runs_on_policy_allowed(path: Path, job_name: str, raw: Any) -> str:
    """Fail-closed policy gate for a job's runs-on.

    Returns classification: ``literal-hosted`` or ``canonical-trusted-cadml``.
    Raises AssertionError for every other form (including unevaluable dynamics).
    """
    raw_s = _as_str(raw).strip()
    assert raw_s, f"{path.name} job {job_name!r}: runs-on must be set"

    if _is_literal_hosted_runs_on(raw):
        return "literal-hosted"

    if isinstance(raw, list):
        raise AssertionError(
            f"{path.name} job {job_name!r}: multi-label / list runs-on is not in the "
            f"accepted policy set (literal hosted label or canonical CADML); got {raw!r}"
        )

    if _is_canonical_trusted_cadml(raw_s):
        return "canonical-trusted-cadml"

    # Everything else is fail-closed reject — including matrix.*, vars.OTHER_*,
    # inputs.*, partial CADML, unconditional CADML, and unevaluable expressions.
    raise AssertionError(
        f"{path.name} job {job_name!r}: runs-on is not an accepted policy form. "
        f"Allowed: (1) literal label in {sorted(APPROVED_GITHUB_HOSTED_LABELS)}, "
        f"or (2) the exact canonical trusted CADML expression. Custom labels "
        f"(e.g. ubuntu-private), dynamics, and unevaluable forms are rejected "
        f"until the policy is deliberately extended. got: {raw_s!r}"
    )


def _resolve_canonical_cadml(ctx: Dict[str, Any]) -> str:
    """Evaluate only the canonical trusted CADML expression under ctx."""
    inner = _unwrap_expression(CANONICAL_TRUSTED_CADML_RUNS_ON)
    value = _eval_github_expr(inner, ctx)
    if value is None or value is False:
        return ""
    return str(value)


def _base_ctx(
    *,
    event_name: str,
    ref: str,
    pr_number: Optional[int] = None,
    run_id: str = "run-1",
    cadml_runner: Optional[str] = "cad-ml",
    extra_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    event: Dict[str, Any] = {}
    if pr_number is not None:
        event["pull_request"] = {"number": pr_number}
    vars_map: Dict[str, Any] = {}
    if cadml_runner is not None:
        vars_map["CADML_LINUX_RUNNER"] = cadml_runner
    else:
        vars_map["CADML_LINUX_RUNNER"] = ""
    if extra_vars:
        vars_map.update(extra_vars)
    return {
        "github": {
            "event_name": event_name,
            "ref": ref,
            "run_id": run_id,
            "event": event,
            "workflow": "test-workflow",
        },
        "vars": vars_map,
        "matrix": {},
        "inputs": {},
    }


def _pr_contexts() -> List[Dict[str, Any]]:
    return [
        _base_ctx(event_name="pull_request", ref="refs/pull/1/merge", pr_number=1),
        _base_ctx(event_name="pull_request", ref="refs/heads/main", pr_number=99),
        _base_ctx(
            event_name="pull_request_target", ref="refs/pull/2/merge", pr_number=2
        ),
        _base_ctx(event_name="pull_request", ref="refs/heads/feature/x", pr_number=3),
        _base_ctx(
            event_name="pull_request",
            ref="refs/pull/4/merge",
            pr_number=4,
            cadml_runner=None,
        ),
    ]


def _untrusted_non_pr_contexts() -> List[Dict[str, Any]]:
    return [
        _base_ctx(event_name="push", ref="refs/heads/feature/x"),
        _base_ctx(event_name="push", ref="refs/heads/master"),
        _base_ctx(event_name="workflow_dispatch", ref="refs/heads/feature/x"),
        _base_ctx(event_name="workflow_dispatch", ref="refs/tags/v1.0.0"),
    ]


def _trusted_contexts() -> List[Dict[str, Any]]:
    return [
        _base_ctx(event_name="push", ref="refs/heads/main"),
        _base_ctx(event_name="schedule", ref="refs/heads/main"),
        _base_ctx(event_name="workflow_dispatch", ref="refs/heads/main"),
    ]


# ---------------------------------------------------------------------------
# Concurrency / uvnet helpers
# ---------------------------------------------------------------------------


def _concurrency_block(workflow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    conc = workflow.get("concurrency")
    return conc if isinstance(conc, dict) else None


def _parse_literal_cancel_in_progress(conc: Dict[str, Any]) -> Optional[bool]:
    """Parse concurrency.cancel-in-progress under a fail-closed exact contract.

    When the key is present, only BaseLoader literal strings ``\"true\"`` or
    ``\"false\"`` are accepted. Expression/dynamic forms (``${{ true }}``),
    ``yes``/``1``/unknown spellings are rejected — they would either skip
    validation while GitHub still cancels, or mis-classify cancel policy.

    Returns:
        True / False when the key is present and valid.
        None when the key is absent.
    """
    if "cancel-in-progress" not in conc:
        return None
    raw = conc["cancel-in-progress"]
    # BaseLoader yields plain strings for YAML bools; reject anything else.
    if not isinstance(raw, str):
        raise AssertionError(
            "concurrency.cancel-in-progress must be a BaseLoader literal string "
            f"'true' or 'false' when set; got {raw!r} ({type(raw).__name__})"
        )
    if raw == "true":
        return True
    if raw == "false":
        return False
    raise AssertionError(
        "concurrency.cancel-in-progress must be the exact literal string "
        f"'true' or 'false' (reject expressions, yes/1/unknown spellings that "
        f"fail-open validation while GitHub may still cancel); got {raw!r}"
    )


def _assert_canonical_cancel_group(path: Path, conc: Dict[str, Any]) -> None:
    group = _as_str(conc.get("group", ""))
    assert (
        group
    ), f"{path.name}: concurrency.group must be set when cancel-in-progress is true"
    norm = _normalize_policy_text(group)
    expected = _normalize_policy_text(CANONICAL_CANCEL_CONCURRENCY_GROUP)
    assert norm == expected, (
        f"{path.name}: cancel-in-progress: true concurrency.group must equal the "
        f"exact canonical event-scoped form {CANONICAL_CANCEL_CONCURRENCY_GROUP!r}; "
        f"got {group!r}"
    )


def _assert_concurrency_policy(path: Path, workflow: Dict[str, Any]) -> None:
    """Full concurrency contract for a workflow (fail-closed on dynamic booleans)."""
    conc = _concurrency_block(workflow)
    if conc is None:
        return
    cancel = _parse_literal_cancel_in_progress(conc)
    if cancel is True:
        _assert_canonical_cancel_group(path, conc)
    # cancel is False: no cancellation — cannot cross-cancel; group form free.
    # cancel is None: key absent — no cancel-in-progress policy to validate.


def _assert_canonical_cancel_concurrency(path: Path, conc: Dict[str, Any]) -> None:
    """Assert cancel-in-progress is literal true with the exact canonical group."""
    cancel = _parse_literal_cancel_in_progress(conc)
    assert cancel is True, (
        f"{path.name}: expected concurrency.cancel-in-progress literal 'true' "
        f"with canonical group; got cancel-in-progress={conc.get('cancel-in-progress')!r}"
    )
    _assert_canonical_cancel_group(path, conc)


def _truthy_gha_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _defaults_run_shell(obj: Dict[str, Any]) -> Optional[str]:
    """Return defaults.run.shell if set on a workflow or job mapping."""
    defaults = obj.get("defaults")
    if not isinstance(defaults, dict):
        return None
    run = defaults.get("run")
    if not isinstance(run, dict):
        return None
    shell = run.get("shell")
    if shell is None:
        return None
    return _as_str(shell).strip()


def _find_step_by_name(job: Dict[str, Any], name: str) -> Dict[str, Any]:
    for step in job.get("steps") or []:
        if isinstance(step, dict) and step.get("name") == name:
            return step
    raise AssertionError(f"missing required step {name!r}")


def _assert_key_absent(obj: Dict[str, Any], key: str, where: str) -> None:
    """Require a key to be completely absent (not merely a known-false spelling).

    Expression forms like ``continue-on-error: ${{ true }}`` are still a soft-fail
    surface; parsing truthy spellings is insufficient.
    """
    assert key not in obj, (
        f"{where} must not set {key!r} (key must be absent; expression forms "
        f"like ${{{{ true }}}} bypass spelling-based truthy checks); "
        f"got {obj.get(key)!r}"
    )


def _assert_pull_request_trigger_unfiltered(workflow: Dict[str, Any]) -> None:
    """Positive pull_request trigger contract for the security gate workflow.

    Skip-proof requires the gate to **run on normal PRs targeting main**, not
    merely to avoid ``paths`` filters. Reject activity-only / non-main branch
    filters that never schedule the job for ordinary open PRs to main.
    """
    on = workflow.get("on")
    assert on is not None, "workflow must define on:"
    # Top-level unfiltered: `on: pull_request` (all PR activity, all branches).
    if on == "pull_request":
        return
    assert isinstance(
        on, dict
    ), f"workflow on: must be a mapping including pull_request; got {type(on).__name__}"
    assert "pull_request" in on, (
        "workflow must include an on.pull_request trigger so uvnet-inspector-gate "
        "actually runs on public PRs (skip-proof requires execution, not absence)"
    )
    pr = on["pull_request"]
    # Unfiltered null / empty string / empty mapping.
    if pr is None or pr == "" or pr == {}:
        return
    # Reject sequence/scalar activity shorthand (e.g. [closed], "closed") —
    # those do not schedule normal opened/synchronize PRs to main.
    assert isinstance(pr, dict), (
        "on.pull_request must be a mapping or null/empty unfiltered form; "
        "sequence/scalar activity shorthand is rejected (e.g. [closed] only "
        "runs on those activities, not normal open PRs to main); "
        f"got {pr!r} ({type(pr).__name__})"
    )

    # Positive denylist of filters that prevent normal main-targeting PR runs.
    for banned, why in (
        (
            "types",
            "activity types filters (e.g. [closed]) skip open/synchronize PRs",
        ),
        (
            "branches-ignore",
            "branches-ignore can exclude main and skip the security gate",
        ),
        (
            "paths",
            "path filters skip the gate on docs-only/CODEOWNERS-only PRs",
        ),
        (
            "paths-ignore",
            "paths-ignore can drop the gate for relevant PR file sets",
        ),
    ):
        assert banned not in pr, (
            f"on.pull_request must not set {banned}: ({why}); "
            f"got {banned}={pr.get(banned)!r}"
        )

    # If branches is present, it must be a list that includes main
    # (main + master is the current valid policy).
    if "branches" in pr:
        branches = pr["branches"]
        assert isinstance(branches, list), (
            "on.pull_request.branches must be a list when set; "
            f"got {branches!r} ({type(branches).__name__})"
        )
        assert "main" in branches, (
            "on.pull_request.branches must include 'main' so normal PRs "
            f"targeting main run the gate (main/master lists are valid); "
            f"got {branches!r}"
        )


def _assert_critical_step_exact(
    *,
    step: Dict[str, Any],
    expected_name: str,
    expected_run: str,
) -> None:
    """Critical gate step: exact run equality; no if/shell/continue-on-error key."""
    assert step.get("name") == expected_name
    # Any conditional skip surface is forbidden (not only if: false).
    assert "if" not in step, (
        f"critical gate step {expected_name!r} must not set if: "
        f"(skips on some events are still a skip-proof bypass); got if={step.get('if')!r}"
    )
    assert "shell" not in step, (
        f"critical gate step {expected_name!r} must not override shell: "
        f"(custom shells can ignore exit codes); got shell={step.get('shell')!r}"
    )
    # Key must be absent — ${{ true }} is not caught by spelling-based truthy checks.
    _assert_key_absent(
        step, "continue-on-error", f"critical gate step {expected_name!r}"
    )
    assert "run" in step, f"critical gate step {expected_name!r} must have run:"
    actual = _normalize_policy_text(_as_str(step.get("run")))
    expected = _normalize_policy_text(expected_run)
    assert actual == expected, (
        f"critical gate step {expected_name!r} run must exactly equal the canonical "
        f"standalone command (no trailing maskers, no extra statements). "
        f"expected={expected_run!r} got={step.get('run')!r}"
    )


def _assert_uvnet_gate_skip_proof(
    workflow: Dict[str, Any], job: Dict[str, Any]
) -> None:
    """Structural skip-proof checks for uvnet-inspector-gate.

    Policy is exact-equality of three dedicated critical steps — not a growing
    denylist of shell mask tokens. Soft-fail / skip surfaces outside ``run``
    (``if``, ``needs``, ``continue-on-error`` keys, custom shells, filtered PR
    triggers) are rejected by key absence or exact trigger shape.
    """
    _assert_pull_request_trigger_unfiltered(workflow)

    raw = job.get("runs-on")
    raw_s = _as_str(raw).strip()
    assert _is_literal_hosted_runs_on(raw), (
        "uvnet-inspector-gate must use a literal approved GitHub-hosted runs-on "
        f"{sorted(APPROVED_GITHUB_HOSTED_LABELS)} (skip-proof torch setup); "
        f"got {raw_s!r}"
    )
    assert "CADML_LINUX_RUNNER" not in raw_s
    assert "self-hosted" not in raw_s

    assert "if" not in job, (
        "uvnet-inspector-gate must not set job-level if: "
        f"(job skip is a gate bypass); got if={job.get('if')!r}"
    )
    # continue-on-error key must be absent (not merely parse false / ${{ true }}).
    _assert_key_absent(job, "continue-on-error", "uvnet-inspector-gate job")

    # Independent security proof: skipped/failed upstream must never skip this job.
    needs = job.get("needs", None)
    if "needs" in job:
        # Allow only an explicitly empty dependency list if the key is present.
        empty = needs is None or needs == [] or needs == ""
        assert empty, (
            "uvnet-inspector-gate must not declare needs: dependencies "
            "(a skipped upstream yields skipped/success for this job and can "
            f"satisfy a required check without running the gate); got needs={needs!r}"
        )

    wf_shell = _defaults_run_shell(workflow)
    assert wf_shell is None, (
        "uvnet-inspector-gate workflow must not set defaults.run.shell "
        f"(custom shells can ignore exit codes); got {wf_shell!r}"
    )
    job_shell = _defaults_run_shell(job)
    assert job_shell is None, (
        "uvnet-inspector-gate job must not set defaults.run.shell "
        f"(custom shells can ignore exit codes); got {job_shell!r}"
    )

    for step_name, expected_run in CANONICAL_UVNET_CRITICAL_STEPS:
        step = _find_step_by_name(job, step_name)
        _assert_critical_step_exact(
            step=step, expected_name=step_name, expected_run=expected_run
        )

    # Ordinary deps install must remain separate (not fold torch into it).
    dep_steps = [
        s
        for s in (job.get("steps") or [])
        if isinstance(s, dict) and "requirements.txt" in _as_str(s.get("run", ""))
    ]
    assert dep_steps, (
        "uvnet-inspector-gate must keep an ordinary dependency install step "
        "(requirements.txt) separate from the pinned-torch critical step"
    )
    for s in dep_steps:
        run = _as_str(s.get("run", ""))
        assert "torch==" not in run, (
            "ordinary dependency install must not also install torch; "
            "pinned torch belongs only in the dedicated critical step. "
            f"step={s.get('name')!r}"
        )


# ---------------------------------------------------------------------------
# Tests — discovery
# ---------------------------------------------------------------------------


def test_workflows_dir_is_dynamically_discovered() -> None:
    paths = _iter_workflow_paths()
    assert len(paths) >= 10
    names = {p.name for p in paths}
    assert "ci.yml" in names
    assert "ci-tiered-tests.yml" in names
    source = Path(__file__).read_text(encoding="utf-8")
    # Both GitHub-accepted extensions must be scanned (structural lock in source).
    assert 'glob("*.yml")' in source or "glob('*.yml')" in source
    assert 'glob("*.yaml")' in source or "glob('*.yaml')" in source
    assert not re.search(
        r"^(?!\s*#).*WORKFLOW_ALLOWLIST\s*=", source, flags=re.MULTILINE
    )
    assert not re.search(
        r"^(?!\s*#).*ALLOWED_WORKFLOWS\s*=", source, flags=re.MULTILINE
    )


def test_iter_workflow_paths_discovers_both_yml_and_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observed discovery: both .yml and .yaml are returned (no allowlist).

    A regression that drops ``*.yaml`` would leave evil.yaml invisible to every
    dynamic security assertion while GitHub still executes it.
    """
    (tmp_path / "alpha.yml").write_text(
        "name: alpha\non: push\njobs: {}\n", encoding="utf-8"
    )
    (tmp_path / "beta.yaml").write_text(
        "name: beta\non: push\njobs: {}\n", encoding="utf-8"
    )
    (tmp_path / "not-a-workflow.txt").write_text("ignore\n", encoding="utf-8")

    # Pin the module global used when no dir arg is passed (same-module lookup).
    monkeypatch.setattr(sys.modules[__name__], "WORKFLOWS_DIR", tmp_path)
    via_arg = {p.name for p in _iter_workflow_paths(tmp_path)}
    via_module = {p.name for p in _iter_workflow_paths()}
    assert via_arg == {"alpha.yml", "beta.yaml"}
    assert via_module == {"alpha.yml", "beta.yaml"}
    assert "not-a-workflow.txt" not in via_arg


# ---------------------------------------------------------------------------
# Tests — runs-on policy (dynamic over every workflow)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path,workflow", _all_workflows(), ids=_workflow_ids())
def test_all_jobs_runs_on_match_fail_closed_policy(
    path: Path, workflow: Dict[str, Any]
) -> None:
    """Every job runs-on is either literal hosted or exact canonical CADML."""
    for job_name, job in _jobs(workflow):
        raw = job.get("runs-on")
        if raw is None:
            continue
        _assert_runs_on_policy_allowed(path, job_name, raw)


@pytest.mark.parametrize("path,workflow", _all_workflows(), ids=_workflow_ids())
def test_no_pr_path_selects_self_hosted(path: Path, workflow: Dict[str, Any]) -> None:
    """PR contexts never land on self-hosted under accepted policy forms."""
    for job_name, job in _jobs(workflow):
        raw = job.get("runs-on")
        if raw is None:
            continue
        kind = _assert_runs_on_policy_allowed(path, job_name, raw)
        if kind == "literal-hosted":
            continue
        # canonical-trusted-cadml: evaluate under every PR context.
        for ctx in _pr_contexts():
            resolved = _resolve_canonical_cadml(ctx)
            assert not _is_self_hosted_resolved(resolved), (
                f"{path.name} job {job_name!r}: PR context "
                f"{ctx['github']['event_name']}/{ctx['github']['ref']} resolved "
                f"runs-on={resolved!r} — self-hosted forbidden on pull_request"
            )
            assert _is_hosted_label(resolved), (
                f"{path.name} job {job_name!r}: PR must resolve to a hosted label; "
                f"got {resolved!r}"
            )


@pytest.mark.parametrize("path,workflow", _all_workflows(), ids=_workflow_ids())
def test_untrusted_non_pr_events_fail_closed_to_hosted(
    path: Path, workflow: Dict[str, Any]
) -> None:
    for job_name, job in _jobs(workflow):
        raw = job.get("runs-on")
        if raw is None:
            continue
        kind = _assert_runs_on_policy_allowed(path, job_name, raw)
        if kind == "literal-hosted":
            continue
        for ctx in _untrusted_non_pr_contexts():
            resolved = _resolve_canonical_cadml(ctx)
            assert _is_hosted_label(resolved), (
                f"{path.name} job {job_name!r}: untrusted event "
                f"{ctx['github']['event_name']}@{ctx['github']['ref']} resolved "
                f"{resolved!r}; expected hosted"
            )


@pytest.mark.parametrize("path,workflow", _all_workflows(), ids=_workflow_ids())
def test_trusted_contexts_select_cadml_when_configured(
    path: Path, workflow: Dict[str, Any]
) -> None:
    for job_name, job in _jobs(workflow):
        raw = job.get("runs-on")
        if raw is None:
            continue
        kind = _assert_runs_on_policy_allowed(path, job_name, raw)
        if kind != "canonical-trusted-cadml":
            continue
        for ctx in _trusted_contexts():
            resolved = _resolve_canonical_cadml(ctx)
            assert resolved == "cad-ml", (
                f"{path.name} job {job_name!r}: trusted "
                f"{ctx['github']['event_name']}@{ctx['github']['ref']} should select "
                f"cad-ml; got {resolved!r}"
            )
        # Unset var → hosted even on trusted events.
        for ctx in _trusted_contexts():
            ctx_unset = _base_ctx(
                event_name=ctx["github"]["event_name"],
                ref=ctx["github"]["ref"],
                cadml_runner=None,
            )
            resolved = _resolve_canonical_cadml(ctx_unset)
            assert _is_hosted_label(resolved), (
                f"{path.name} job {job_name!r}: unset CADML_LINUX_RUNNER must fail "
                f"closed to hosted; got {resolved!r}"
            )


# ---------------------------------------------------------------------------
# Tests — concurrency (exact canonical group)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path,workflow", _all_workflows(), ids=_workflow_ids())
def test_concurrency_policy_fail_closed_on_every_workflow(
    path: Path, workflow: Dict[str, Any]
) -> None:
    """Every workflow concurrency block is validated (no skip on dynamic booleans)."""
    _assert_concurrency_policy(path, workflow)


# ---------------------------------------------------------------------------
# Tests — uvnet-inspector-gate
# ---------------------------------------------------------------------------


def test_uvnet_inspector_gate_is_skip_proof_with_explicit_torch() -> None:
    path = WORKFLOWS_DIR / "ci-tiered-tests.yml"
    workflow = _load_workflow(path)
    assert (
        "uvnet-inspector-gate" in workflow["jobs"]
    ), "uvnet-inspector-gate must remain present (do not remove or weaken the gate)"
    job = workflow["jobs"]["uvnet-inspector-gate"]
    _assert_uvnet_gate_skip_proof(workflow, job)


# ---------------------------------------------------------------------------
# Mutation tests — fail-open class closures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,reason",
    [
        ("${{ matrix.runner }}", "matrix.runner"),
        ("${{ vars.OTHER_RUNNER || 'ubuntu-latest' }}", "arbitrary vars runner"),
        ("${{ inputs.runner }}", "inputs.runner"),
        ("${{ some.unresolved.expression }}", "unresolved expression"),
        (
            "${{ vars.CADML_LINUX_RUNNER || 'ubuntu-latest' }}",
            "unconditional CADML (pre-fix)",
        ),
        (
            (
                "${{ github.event_name == 'pull_request' && "
                "vars.CADML_LINUX_RUNNER || 'ubuntu-latest' }}"
            ),
            "PR-positive CADML arm",
        ),
        ("${{ matrix.os }}", "matrix.os"),
        ("self-hosted", "literal self-hosted"),
        ("[self-hosted, linux]", "multi-label self-hosted"),
        ("ubuntu-private", "custom label masquerading as ubuntu-"),
        ("ubuntu-evil", "custom label masquerading as ubuntu-"),
        ("ubuntu-22.04", "GH-hosted version not in approved set"),
        ("windows-latest", "windows not in approved set for this repo"),
        ("macos-latest", "macos not in approved set for this repo"),
    ],
)
def test_mutation_rejects_non_canonical_runs_on(raw: str, reason: str) -> None:
    """Non-policy runs-on forms must be rejected (not silently certified safe)."""
    with pytest.raises(AssertionError, match="not an accepted policy form|multi-label"):
        _assert_runs_on_policy_allowed(Path("synthetic.yml"), f"mut-{reason}", raw)


def test_mutation_ubuntu_private_not_literal_hosted() -> None:
    """P1: custom ubuntu-* labels are not GitHub-hosted proof."""
    assert _is_hosted_label("ubuntu-private") is False
    assert _is_hosted_label("ubuntu-evil") is False
    with pytest.raises(AssertionError, match="not an accepted policy form"):
        _assert_runs_on_policy_allowed(Path("synthetic.yml"), "j", "ubuntu-private")
    with pytest.raises(AssertionError, match="not an accepted policy form"):
        _assert_runs_on_policy_allowed(Path("synthetic.yml"), "j", "ubuntu-evil")


def test_mutation_list_with_custom_label_rejected() -> None:
    """List runs-on containing a custom label must not pass as hosted."""
    with pytest.raises(AssertionError, match="multi-label|not an accepted policy form"):
        _assert_runs_on_policy_allowed(
            Path("synthetic.yml"),
            "j",
            ["ubuntu-latest", "ubuntu-private"],
        )
    with pytest.raises(AssertionError, match="multi-label|not an accepted policy form"):
        _assert_runs_on_policy_allowed(
            Path("synthetic.yml"),
            "j",
            ["ubuntu-private"],
        )


def test_mutation_matrix_runner_not_certified_safe_via_empty_resolve() -> None:
    """Regression for P1: unevaluable matrix.runner must not count as hosted."""
    raw = "${{ matrix.runner }}"
    with pytest.raises(AssertionError, match="not an accepted policy form"):
        _assert_runs_on_policy_allowed(Path("synthetic.yml"), "j", raw)
    assert _is_self_hosted_resolved("") is True
    assert _is_hosted_label("") is False


def test_mutation_arbitrary_vars_runner_not_passable_when_unset() -> None:
    """vars.OTHER_RUNNER must not pass merely because it is absent from the test ctx."""
    raw = "${{ vars.OTHER_RUNNER || 'ubuntu-latest' }}"
    with pytest.raises(AssertionError, match="not an accepted policy form"):
        _assert_runs_on_policy_allowed(Path("synthetic.yml"), "j", raw)
    ctx = _base_ctx(
        event_name="pull_request",
        ref="refs/pull/1/merge",
        pr_number=1,
        extra_vars={"OTHER_RUNNER": "cad-ml"},
    )
    try:
        resolved = str(_eval_github_expr(_unwrap_expression(raw), ctx) or "")
    except _ExprError:
        resolved = ""
    assert resolved == "cad-ml"
    assert _is_self_hosted_resolved(resolved)


def test_mutation_inputs_runner_rejected() -> None:
    with pytest.raises(AssertionError, match="not an accepted policy form"):
        _assert_runs_on_policy_allowed(
            Path("synthetic.yml"), "j", "${{ inputs.runner }}"
        )


def test_mutation_unresolved_expression_rejected() -> None:
    with pytest.raises(AssertionError, match="not an accepted policy form"):
        _assert_runs_on_policy_allowed(
            Path("synthetic.yml"), "j", "${{ env.TOTALLY_UNKNOWN }}"
        )


def test_mutation_concurrency_token_presence_insufficient() -> None:
    """Token-presence checks are not enough — must match exact canonical group."""
    almost = {
        "group": (
            "${{ github.workflow }}-${{ github.event_name }}-${{ github.ref }}-"
            "${{ github.event.pull_request.number || github.run_id }}"
        ),
        "cancel-in-progress": "true",
    }
    with pytest.raises(AssertionError, match="exact canonical"):
        _assert_canonical_cancel_concurrency(Path("synthetic.yml"), almost)

    soup = {
        "group": "github.event_name github.event.pull_request.number github.run_id ||",
        "cancel-in-progress": "true",
    }
    with pytest.raises(AssertionError, match="exact canonical"):
        _assert_canonical_cancel_concurrency(Path("synthetic.yml"), soup)

    _assert_canonical_cancel_concurrency(
        Path("synthetic.yml"),
        {
            "group": CANONICAL_CANCEL_CONCURRENCY_GROUP,
            "cancel-in-progress": "true",
        },
    )


def test_mutation_concurrency_dynamic_true_expression_rejected() -> None:
    """P1: ${{ true }} must not skip group validation while GitHub still cancels."""
    conc = {
        "group": "${{ github.workflow }}-${{ github.ref }}",
        "cancel-in-progress": "${{ true }}",
    }
    with pytest.raises(
        AssertionError, match="literal string|'true' or 'false'|expression"
    ):
        _assert_concurrency_policy(Path("synthetic.yml"), {"concurrency": conc})
    with pytest.raises(
        AssertionError, match="literal string|'true' or 'false'|expression"
    ):
        _parse_literal_cancel_in_progress(conc)


def test_mutation_concurrency_yes_and_one_spellings_rejected() -> None:
    for bad in ("yes", "1", "True", "TRUE", "on"):
        with pytest.raises(AssertionError, match="literal string|'true' or 'false'"):
            _parse_literal_cancel_in_progress({"cancel-in-progress": bad})


def test_concurrency_literal_true_canonical_and_literal_false_controls() -> None:
    """Positive controls: exact true+canonical group; exact false (no cancel)."""
    _assert_concurrency_policy(
        Path("synthetic.yml"),
        {
            "concurrency": {
                "group": CANONICAL_CANCEL_CONCURRENCY_GROUP,
                "cancel-in-progress": "true",
            }
        },
    )
    # false: no cancellation — group may be non-canonical; policy still accepts.
    _assert_concurrency_policy(
        Path("synthetic.yml"),
        {
            "concurrency": {
                "group": "evaluation-soft-mode-smoke-${{ github.ref }}",
                "cancel-in-progress": "false",
            }
        },
    )
    assert _parse_literal_cancel_in_progress({"cancel-in-progress": "false"}) is False
    assert _parse_literal_cancel_in_progress({"cancel-in-progress": "true"}) is True
    assert _parse_literal_cancel_in_progress({}) is None


def _uvnet_good_workflow() -> Dict[str, Any]:
    """Minimal workflow with unfiltered pull_request (branches allowed)."""
    return {
        "name": "synthetic",
        "on": {"pull_request": {"branches": ["main"]}},
        "jobs": {},
    }


def _uvnet_good_job() -> Dict[str, Any]:
    """Minimal job matching the three exact critical steps (+ separate deps)."""
    steps: List[Dict[str, Any]] = [
        {
            "name": "Install dependencies",
            "run": "pip install -r requirements.txt",
        },
    ]
    for name, cmd in CANONICAL_UVNET_CRITICAL_STEPS:
        steps.append({"name": name, "run": cmd})
    # No needs, no continue-on-error, no if — keys absent by construction.
    return {"runs-on": "ubuntu-latest", "steps": steps}


def test_mutation_uvnet_positive_control_synthetic_and_real() -> None:
    """Exact canonical steps pass; real workflow job also passes."""
    _assert_uvnet_gate_skip_proof(_uvnet_good_workflow(), _uvnet_good_job())
    path = WORKFLOWS_DIR / "ci-tiered-tests.yml"
    workflow = _load_workflow(path)
    _assert_uvnet_gate_skip_proof(workflow, workflow["jobs"]["uvnet-inspector-gate"])


def test_mutation_uvnet_job_if_false_rejected() -> None:
    job = _uvnet_good_job()
    job["if"] = "false"
    with pytest.raises(AssertionError, match="job-level if"):
        _assert_uvnet_gate_skip_proof(_uvnet_good_workflow(), job)


def test_mutation_uvnet_pytest_if_push_only_rejected() -> None:
    """PR-skipping if on pytest step is a gate bypass even when not literally false."""
    job = _uvnet_good_job()
    for step in job["steps"]:
        if step["name"] == CANONICAL_UVNET_CRITICAL_STEPS[2][0]:
            step["if"] = "${{ github.event_name == 'push' }}"
    with pytest.raises(AssertionError, match="must not set if"):
        _assert_uvnet_gate_skip_proof(_uvnet_good_workflow(), job)


def test_mutation_uvnet_pytest_shell_bash0_rejected() -> None:
    job = _uvnet_good_job()
    for step in job["steps"]:
        if step["name"] == CANONICAL_UVNET_CRITICAL_STEPS[2][0]:
            step["shell"] = "bash {0}"
    with pytest.raises(AssertionError, match="must not override shell"):
        _assert_uvnet_gate_skip_proof(_uvnet_good_workflow(), job)


def test_mutation_uvnet_job_defaults_run_shell_rejected() -> None:
    job = _uvnet_good_job()
    job["defaults"] = {"run": {"shell": "bash {0}"}}
    with pytest.raises(AssertionError, match="defaults.run.shell"):
        _assert_uvnet_gate_skip_proof(_uvnet_good_workflow(), job)


def test_mutation_uvnet_workflow_defaults_run_shell_rejected() -> None:
    workflow = _uvnet_good_workflow()
    workflow["defaults"] = {"run": {"shell": "bash {0}"}}
    with pytest.raises(AssertionError, match="defaults.run.shell"):
        _assert_uvnet_gate_skip_proof(workflow, _uvnet_good_job())


def test_mutation_uvnet_pytest_or_echo_swallowed_rejected() -> None:
    job = _uvnet_good_job()
    for step in job["steps"]:
        if step["name"] == CANONICAL_UVNET_CRITICAL_STEPS[2][0]:
            step["run"] = CANONICAL_UVNET_INSPECTOR_PYTEST_CMD + " || echo swallowed"
    with pytest.raises(AssertionError, match="exactly equal the canonical"):
        _assert_uvnet_gate_skip_proof(_uvnet_good_workflow(), job)


def test_mutation_uvnet_pytest_semicolon_exit_0_rejected() -> None:
    job = _uvnet_good_job()
    for step in job["steps"]:
        if step["name"] == CANONICAL_UVNET_CRITICAL_STEPS[2][0]:
            step["run"] = CANONICAL_UVNET_INSPECTOR_PYTEST_CMD + "; exit 0"
    with pytest.raises(AssertionError, match="exactly equal the canonical"):
        _assert_uvnet_gate_skip_proof(_uvnet_good_workflow(), job)


def test_mutation_uvnet_missing_critical_step_rejected() -> None:
    job = _uvnet_good_job()
    job["steps"] = [
        s
        for s in job["steps"]
        if "torch" not in s.get("name", "").lower()
        or "Install pinned" not in s.get("name", "")
    ]
    # remove torch install step only
    job["steps"] = [
        s
        for s in _uvnet_good_job()["steps"]
        if s["name"] != CANONICAL_UVNET_CRITICAL_STEPS[0][0]
    ]
    with pytest.raises(AssertionError, match="missing required step"):
        _assert_uvnet_gate_skip_proof(_uvnet_good_workflow(), job)


def test_mutation_uvnet_torch_folded_into_deps_rejected() -> None:
    job = _uvnet_good_job()
    for step in job["steps"]:
        if step["name"] == "Install dependencies":
            step["run"] = (
                "pip install -r requirements.txt\n"
                'python -m pip install "torch==2.1.0"\n'
            )
    with pytest.raises(AssertionError, match="must not also install torch"):
        _assert_uvnet_gate_skip_proof(_uvnet_good_workflow(), job)


def test_mutation_uvnet_job_continue_on_error_rejected() -> None:
    """Key presence is enough to fail — including expression form ${{ true }}."""
    job = _uvnet_good_job()
    job["continue-on-error"] = "${{ true }}"
    with pytest.raises(AssertionError, match="continue-on-error"):
        _assert_uvnet_gate_skip_proof(_uvnet_good_workflow(), job)


def test_mutation_uvnet_step_continue_on_error_rejected() -> None:
    """Critical-step continue-on-error: ${{ true }} must not parse as safe."""
    job = _uvnet_good_job()
    for step in job["steps"]:
        if step["name"] == CANONICAL_UVNET_CRITICAL_STEPS[2][0]:
            step["continue-on-error"] = "${{ true }}"
    with pytest.raises(AssertionError, match="continue-on-error"):
        _assert_uvnet_gate_skip_proof(_uvnet_good_workflow(), job)


def test_mutation_uvnet_job_needs_upstream_rejected() -> None:
    """Skipped upstream must not be able to skip the independent security gate."""
    job = _uvnet_good_job()
    job["needs"] = ["synthetic-skipped-upstream"]
    with pytest.raises(AssertionError, match="needs"):
        _assert_uvnet_gate_skip_proof(_uvnet_good_workflow(), job)


def test_mutation_uvnet_missing_pull_request_trigger_rejected() -> None:
    workflow = _uvnet_good_workflow()
    workflow["on"] = {"push": {"branches": ["main"]}}
    with pytest.raises(AssertionError, match="pull_request"):
        _assert_uvnet_gate_skip_proof(workflow, _uvnet_good_job())


def test_mutation_uvnet_pull_request_paths_filter_rejected() -> None:
    workflow = _uvnet_good_workflow()
    workflow["on"] = {
        "pull_request": {
            "branches": ["main"],
            "paths": [".github/workflows/**"],
        }
    }
    with pytest.raises(AssertionError, match="paths"):
        _assert_uvnet_gate_skip_proof(workflow, _uvnet_good_job())


def test_mutation_uvnet_pull_request_paths_ignore_rejected() -> None:
    workflow = _uvnet_good_workflow()
    workflow["on"] = {
        "pull_request": {
            "branches": ["main"],
            "paths-ignore": ["docs/**"],
        }
    }
    with pytest.raises(AssertionError, match="paths-ignore"):
        _assert_uvnet_gate_skip_proof(workflow, _uvnet_good_job())


def test_mutation_uvnet_pull_request_types_closed_only_rejected() -> None:
    """types: [closed] never runs the gate on normal open/sync PRs to main."""
    workflow = _uvnet_good_workflow()
    workflow["on"] = {"pull_request": {"types": ["closed"]}}
    with pytest.raises(AssertionError, match="types"):
        _assert_uvnet_gate_skip_proof(workflow, _uvnet_good_job())


def test_mutation_uvnet_pull_request_branches_feature_only_rejected() -> None:
    """branches: [feature/**] without main skips PRs targeting main."""
    workflow = _uvnet_good_workflow()
    workflow["on"] = {"pull_request": {"branches": ["feature/**"]}}
    with pytest.raises(AssertionError, match="main"):
        _assert_uvnet_gate_skip_proof(workflow, _uvnet_good_job())


def test_mutation_uvnet_pull_request_branches_ignore_main_rejected() -> None:
    """branches-ignore: [main] explicitly excludes the protected default branch."""
    workflow = _uvnet_good_workflow()
    workflow["on"] = {"pull_request": {"branches-ignore": ["main"]}}
    with pytest.raises(AssertionError, match="branches-ignore"):
        _assert_uvnet_gate_skip_proof(workflow, _uvnet_good_job())


def test_mutation_uvnet_pull_request_sequence_activity_shorthand_rejected() -> None:
    """Scalar/sequence activity form [closed] is not a mapping trigger contract."""
    workflow = _uvnet_good_workflow()
    workflow["on"] = {"pull_request": ["closed"]}
    with pytest.raises(AssertionError, match="mapping|sequence|shorthand"):
        _assert_uvnet_gate_skip_proof(workflow, _uvnet_good_job())


def test_mutation_uvnet_pull_request_main_master_branches_accepted() -> None:
    """Positive control: branches [main, master] matches the real workflow policy."""
    workflow = _uvnet_good_workflow()
    workflow["on"] = {"pull_request": {"branches": ["main", "master"]}}
    _assert_uvnet_gate_skip_proof(workflow, _uvnet_good_job())


def test_mutation_uvnet_self_hosted_runs_on_rejected() -> None:
    job = _uvnet_good_job()
    job["runs-on"] = "${{ vars.CADML_LINUX_RUNNER || 'ubuntu-latest' }}"
    with pytest.raises(AssertionError, match="literal approved GitHub-hosted|literal"):
        _assert_uvnet_gate_skip_proof(_uvnet_good_workflow(), job)


def test_mutation_uvnet_unpinned_or_wrong_cmd_rejected() -> None:
    job = _uvnet_good_job()
    for step in job["steps"]:
        if step["name"] == CANONICAL_UVNET_CRITICAL_STEPS[0][0]:
            step["run"] = "pip install torch"
    with pytest.raises(AssertionError, match="exactly equal the canonical"):
        _assert_uvnet_gate_skip_proof(_uvnet_good_workflow(), job)


def test_canonical_forms_accepted_and_evaluate() -> None:
    """Green controls for the two accepted runs-on forms + concurrency."""
    assert (
        _assert_runs_on_policy_allowed(Path("synthetic.yml"), "h", "ubuntu-latest")
        == "literal-hosted"
    )
    assert (
        _assert_runs_on_policy_allowed(
            Path("synthetic.yml"), "c", CANONICAL_TRUSTED_CADML_RUNS_ON
        )
        == "canonical-trusted-cadml"
    )
    pr = _base_ctx(event_name="pull_request", ref="refs/pull/1/merge", pr_number=1)
    assert _is_hosted_label(_resolve_canonical_cadml(pr))
    trusted = _base_ctx(event_name="push", ref="refs/heads/main")
    assert _resolve_canonical_cadml(trusted) == "cad-ml"
    _assert_canonical_cancel_concurrency(
        Path("synthetic.yml"),
        {
            "group": CANONICAL_CANCEL_CONCURRENCY_GROUP,
            "cancel-in-progress": "true",
        },
    )


def test_empty_resolved_label_is_not_hosted() -> None:
    """P1 control: empty resolution must never be treated as safe/hosted."""
    assert _is_self_hosted_resolved("") is True
    assert _is_hosted_label("") is False
