# CAD ML B-Rep Golden Manifest Development

Date: 2026-05-12

## Goal

Make the Phase 4 STEP/IGES golden set auditable before real files are collected.
The repository does not currently contain 50-100 real STEP/IGES benchmark files, so
this slice adds the manifest contract and release-readiness gate without fabricating
sample coverage.

## Changes

- Added `scripts/validate_brep_golden_manifest.py`.
  - Validates `schema_version=brep_golden_manifest.v1`.
  - Checks required case fields: `id`, `path`, `source_type`, `part_family`,
    `license`, and `expected_behavior`.
  - Rejects duplicate case IDs.
  - Checks file existence by default.
  - Requires topology expectations for `parse_success` cases.
  - Requires stable failure reasons for expected parse or graph failures.
  - Excludes `fixture`, `synthetic_demo`, and `generated_mock` cases from release
    sample counts.
  - Reports `release_eligible_count` and fails release readiness below the default
    minimum of 50 samples.
- Added `config/brep_golden_manifest.example.json`.
  - Uses the existing `tests/fixtures/mock_cube.step` as a contract example.
  - Explicitly marks the case as `release_eligible=false`.
  - Produces `insufficient_release_samples`, not release-ready.
- Updated `scripts/eval_brep_step_dir.py`.
  - Added `--manifest`.
  - Manifest paths resolve from the manifest `root`.
  - Manifest-driven runs produce the same strict `results.csv`, `summary.json`, and
    `graph_qa.json` outputs.
- Updated `tests/unit/test_eval_brep_step_dir.py`.
  - Covers manifest path resolution.
- Added `tests/unit/test_validate_brep_golden_manifest.py`.
  - Covers release-ready 50-file manifest behavior with temp files.
  - Covers fixture exclusion from release counts.
  - Covers duplicate IDs, missing files, missing failure reasons, CLI output, and
    release gate failure.
- Updated the Phase 4 TODO.

## Manifest Contract

Minimal release-eligible case:

```json
{
  "id": "vendor_block_001",
  "path": "vendor/block_001.step",
  "format": "step",
  "source_type": "vendor",
  "release_eligible": true,
  "part_family": "block",
  "license": "internal-eval",
  "expected_behavior": "parse_success",
  "expected_topology": {
    "faces_min": 1,
    "edges_min": 0,
    "solids_min": 0,
    "graph_nodes_min": 1,
    "surface_types": ["plane"]
  }
}
```

Release-excluded fixture/demo cases must use:

```json
{
  "source_type": "fixture",
  "release_eligible": false
}
```

## Commands

Validate the example manifest without enforcing release readiness:

```bash
python scripts/validate_brep_golden_manifest.py \
  --manifest config/brep_golden_manifest.example.json \
  --output-json reports/benchmark/brep_golden_manifest/example_validation.json
```

Gate a real manifest for release readiness:

```bash
python scripts/validate_brep_golden_manifest.py \
  --manifest config/brep_golden_manifest.json \
  --output-json reports/benchmark/brep_golden_manifest/latest_validation.json \
  --fail-on-not-release-ready
```

Run strict evaluation from a manifest:

```bash
python scripts/eval_brep_step_dir.py \
  --manifest config/brep_golden_manifest.json \
  --strict \
  --output-dir reports/benchmark/brep_step_iges_golden/latest
```

## Remaining Work

- Collect 50-100 real STEP/IGES files with source/license metadata.
- Fill `config/brep_golden_manifest.json` or a private equivalent.
- Enable `BREP_GOLDEN_EVAL_ENABLE=true` on an OCC-enabled runner and feed the emitted
  `summary.json` into the forward scorecard through the CI wrapper.
