# CAD ML Forward Scorecard Manufacturing Evidence Development

Date: 2026-05-12

## Goal

Continue the Phase 6 follow-up by making manufacturing evidence measurable in the
forward scorecard. The scorecard now tracks whether analyze results include DFM,
process, cost, and manufacturing-decision evidence at release-quality coverage.

## Changes

- Updated `src/core/benchmark/forward_scorecard.py`.
  - Added `manufacturing_evidence` as a scorecard component.
  - Added release thresholds for:
    - sample size
    - manufacturing evidence coverage
    - required evidence source coverage
    - reviewed source correctness
    - reviewed top-level payload quality
    - reviewed nested detail quality
  - Added recommendations when manufacturing evidence is missing or incomplete.
  - Added Markdown component rendering for manufacturing evidence.
- Updated `scripts/export_forward_scorecard.py`.
  - Added `--manufacturing-evidence-summary`.
  - Records the provided manufacturing evidence summary path in scorecard artifacts.
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Added workflow/env input support for manufacturing evidence summaries.
  - Emits `manufacturing_evidence_status` to GitHub outputs.
- Updated `.github/workflows/evaluation-report.yml`.
  - Added repository-variable pass-through for manufacturing evidence summary paths.
- Updated forward scorecard tests and CI-wrapper tests.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md` to include
  manufacturing evidence coverage as a scorecard aggregate.

## Input Contract

The scorecard accepts an object through `--manufacturing-evidence-summary`:

```json
{
  "sample_size": 80,
  "records_with_manufacturing_evidence": 80,
  "manufacturing_evidence_coverage_rate": 1.0,
  "source_counts": {
    "dfm": 80,
    "manufacturing_process": 80,
    "manufacturing_cost": 80,
    "manufacturing_decision": 80
  },
  "sources": [
    "dfm",
    "manufacturing_process",
    "manufacturing_cost",
    "manufacturing_decision"
  ]
}
```

Required sources:

- `dfm`
- `manufacturing_process`
- `manufacturing_cost`
- `manufacturing_decision`

## Status Rules

- `release_ready`: at least 30 samples, evidence coverage at or above 90%, and every
  required source at or above 80% coverage. Reviewed source correctness must include
  at least 30 samples, source precision at or above 90%, and source recall at or
  above 90%. Reviewed payload quality must include at least 30 samples and payload
  quality accuracy at or above 90%. Reviewed nested detail quality must include at
  least 30 samples and detail quality accuracy at or above 90%.
- `benchmark_ready_with_gap`: at least 10 samples, evidence coverage at or above 50%,
  and at least two manufacturing evidence sources present.
- `shadow_only`: some evidence exists but not enough for benchmark-ready status.
- `blocked`: no manufacturing evidence sample is provided.

## Compatibility

The new scorecard input is optional. Existing exports still run without the new
argument, but a missing manufacturing evidence component now keeps a full
release-ready scorecard from being declared release-ready. That matches the product
direction: manufacturing intelligence claims should require manufacturing evidence.

## Remaining Work

- DXF benchmark exporters now produce forward-scorecard-compatible manufacturing
  evidence summaries from real analyze outputs.
- CI/release workflow now uploads the consumed manufacturing evidence summary as a
  separate artifact when the configured path exists.
- Labeled DXF benchmark exporter now emits reviewed source correctness metrics, and
  the forward scorecard gates release readiness on precision/recall.
- Labeled DXF benchmark exporter now emits reviewed payload quality metrics, and the
  forward scorecard gates release readiness on payload accuracy.
- Labeled DXF benchmark exporter now emits nested detail payload quality metrics, and
  the forward scorecard gates release readiness on detail accuracy.
- Populate real reviewed source, payload, and detail labels for the release benchmark
  set.
- Tune source, payload, and detail thresholds after the release review set is stable.
- Extend the same summary contract to OCR-only benchmark runs if they use a separate
  exporter from the DXF analyze path.
