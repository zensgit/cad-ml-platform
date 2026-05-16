# CAD ML Forward Scorecard Knowledge Grounding Development

Date: 2026-05-12

## Goal

Continue Phase 6 by making the forward scorecard enforce knowledge grounding
coverage. Knowledge readiness alone is no longer enough for `release_ready`; the
scorecard also requires rule-source and rule-version evidence.

## Changes

- Updated `src/core/benchmark/forward_scorecard.py`.
  - `knowledge` component now reads `knowledge_grounding`.
  - It reports:
    - `grounding_sample_size`
    - `knowledge_evidence_coverage_rate`
    - `rule_source_coverage_rate`
    - `rule_version_coverage_rate`
    - `rule_sources`
    - `rule_versions`
    - `grounding_gaps`
  - It downgrades a ready knowledge foundation to `benchmark_ready_with_gap` when
    grounding coverage is missing or below release thresholds.
  - Markdown component evidence now includes rule-source and rule-version coverage.
  - Recommendations now call out missing knowledge grounding coverage.
- Updated `tests/unit/test_forward_scorecard.py`.
  - Ready fixture now includes full `knowledge_grounding` coverage.
  - Added a regression test proving missing grounding downgrades the scorecard from
    `release_ready` to `benchmark_ready_with_gap`.
- Updated `tests/unit/test_forward_scorecard_release_gate.py`.
  - CI wrapper ready fixture now provides knowledge grounding evidence so true-ready
    wrapper output remains `release_ready`.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Coverage Contract

The scorecard expects this optional shape under the knowledge summary:

```json
{
  "knowledge_grounding": {
    "sample_size": 80,
    "knowledge_evidence_coverage_rate": 1.0,
    "rule_source_coverage_rate": 1.0,
    "rule_version_coverage_rate": 1.0,
    "rule_sources": ["materials_catalog"],
    "rule_versions": ["knowledge_grounding.v1"]
  }
}
```

For release readiness, the component requires:

- evidence coverage at or above `0.8`
- rule-source coverage at or above `0.95`
- rule-version coverage at or above `0.95`
- non-empty `rule_sources`
- non-empty `rule_versions`

## Remaining Phase 6 Work

- Connect process, cost, and DFM checks into the same analyze evidence path.
- Add explicit fixtures for material substitution, H7/g6 fit validation, surface
  finish recommendation, machining process route, and manufacturability risk.
