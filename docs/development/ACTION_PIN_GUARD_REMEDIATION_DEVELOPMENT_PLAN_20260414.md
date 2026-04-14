# Action Pin Guard Remediation Development Plan

Date: 2026-04-14
Branch: `submit/local-main-20260414`
PR: `#398`

## Background

After the isolated workflow regressions were fixed, the only persistent repo-level gate still failing on PR `#398` was `Action Pin Guard`.

Local and remote evidence matched:

- remote workflow: `Action Pin Guard`
- failing step: `Validate workflow action pins`
- local checker: `scripts/ci/check_workflow_action_pins.py`

The failure was not a checker bug. It was actual workflow debt caused by tag-based action references that violated the repository pin policy.

## Violation Inventory

The violation distribution was concentrated in four workflow files:

1. `.github/workflows/evaluation-report.yml` — 50 violations
2. `.github/workflows/code-quality.yml` — 14 violations
3. `.github/workflows/hybrid-superpass-e2e.yml` — 3 violations
4. `.github/workflows/hybrid-superpass-nightly.yml` — 3 violations

All violations were `tag_ref_not_allowed` and already had expected SHAs emitted by the guard.

## Planned Changes

Replace all tag-based refs in the four workflows with the exact SHAs required by the current action pin policy:

- `actions/checkout` → `de0fac2e4500dabe0009e67214ff5f5447ce83dd`
- `actions/setup-python` → `a309ff8b426b58ec0e2a45f0f869d46889d02405`
- `actions/upload-artifact` → `bbbca2ddaa5d8feaa63e36b76fdaad77386f024f`
- `actions/download-artifact` → `3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c`
- `actions/github-script` → `d7906e4ad0b1822421a7e6a35d5ca353c962f410`
- `actions/configure-pages` → `1f0c5cde4bc74cd7e1254d0cb4de8d49e9068c7d`
- `actions/upload-pages-artifact` → `56afc609e74202658d3ffba0e8f6dda462b719fa`

Update the affected workflow regression tests where they still asserted tag refs instead of policy-compliant SHAs.

## Acceptance Criteria

1. Local `check_workflow_action_pins.py` returns `violations_count: 0`.
2. Affected workflow unit tests pass.
3. Remote `Action Pin Guard` turns green on the pushed head.

