# DEV_KNOWLEDGE_CI_SMOKE_20260203

## Summary
- Attempted to trigger CI workflows after adding knowledge test artifacts.
- `ci-enhanced.yml` dispatch succeeded.
- `ci.yml` dispatch initially failed due to GitHub API workflow parsing error referencing `secrets` in an `if` expression.
- Updated `ci.yml` to use `github.token` for GHCR login, then re-dispatched successfully.
- The dispatched `ci.yml` run failed in the lint step due to pre-existing flake8 violations (line length and shadowed imports), so knowledge tests/artifacts were not produced in that run.

## Commands & Results
1) Trigger CI (ci.yml) - initial attempt
- Command:
  - `gh workflow run ci.yml`
- Result: Failed to dispatch.
  - Error: `HTTP 422: Invalid Argument - failed to parse workflow (Line: 151, Col: 13): Unrecognized named-value: 'secrets'.`
  - Context: `startsWith(env.DEDUPCAD_VISION_IMAGE, 'ghcr.io/') && secrets.GHCR_TOKEN != ''` in `ci.yml`.

2) Fix workflow dispatch parsing and retry
- Change:
  - Replaced `secrets.GHCR_TOKEN` with `github.token` in GHCR login step.
- Command:
  - `gh workflow run ci.yml --ref main`
- Result: Dispatch succeeded.

3) Trigger CI Enhanced (ci-enhanced.yml)
- Command:
  - `gh workflow run ci-enhanced.yml`
- Result: Dispatch succeeded.

4) CI dispatch outcome (ci.yml)
- Result: Failed at `lint-type` job.
  - Example lint failures (flake8):
    - `src/core/assistant/api_docs.py:157:101 E501 line too long`
    - `src/core/assistant/knowledge_retriever.py:356:101 E501 line too long`
    - `src/core/audit_enhanced/query.py:383:21 F402 import 'field' shadowed by loop variable`
  - Impact: unit tests and knowledge-test artifacts did not run for this dispatch.

## Notes
- The CI YAML parse error appears specific to workflow dispatch via the GitHub API. The same workflow should still run on push; follow up if CI remains blocked.
