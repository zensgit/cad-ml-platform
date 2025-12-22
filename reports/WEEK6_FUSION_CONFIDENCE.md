#!/usr/bin/env markdown
# Week 6 Report: Fusion Confidence

## Summary
- Added confidence breakdown/source in classification output and strengthened
  3D override when high-confidence geometry signals exist.

## Changes
- `src/api/v1/analyze.py`: surface `confidence_breakdown` and `confidence_source`
  in classification; active learning payload enriched.
- `src/core/knowledge/fusion.py`: strong 3D override when top 3D score >= 0.6.

## Tests
- `python3 -m pytest tests -k fusion -q`

## Verification
- Run an analysis with strong 3D signals and confirm:
  - `classification.confidence_source == "fusion"`
  - `classification.confidence_breakdown` present
