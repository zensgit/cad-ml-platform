# CAD ML Real Evidence Sourcing Worklist

Date: 2026-06-22
Scope: B-Rep golden evidence and manufacturing reviewed labels

## 0. Purpose

The code gates are now in place. This worklist defines the human evidence needed
to move from "tooling is ready" to "release evidence is credible" without
recreating fake-green data.

Two streams can run in parallel:

- B-Rep golden set: real STEP/IGES files with auditable license provenance and
  enough human-verified topology.
- Manufacturing reviewed labels: real reviewer-approved source and payload
  labels with reviewer metadata.

This file is a worklist, not a data artifact. Do not use it to claim release
readiness until the commands in section 4 pass on real artifacts.

## 1. Hard Rules

- Do not fabricate license provenance. A release-eligible B-Rep case needs a
  real `license_source` pointer.
- Do not bulk-fill `license_status`. Classify each file from its actual source.
- `public_nc` and `non_commercial` are the same exclusion fact and must agree.
- Parser-derived topology can be useful coverage, but it does not satisfy the
  verified-topology floor.
- `topology_source=verified` requires a real evidence pointer and a meaningful
  topology floor, not just copied OCC output.
- Manufacturing suggestions are not labels. Only approved rows with reviewer
  metadata count toward reviewed release evidence.

## 2. B-Rep Sourcing Sheet

Create one row per candidate STEP/IGES file before editing the manifest.

| Column | Required | Notes |
| --- | --- | --- |
| `candidate_id` | yes | Stable short ID; later maps to manifest `id`. |
| `local_path` | yes | Target path under `data/brep_golden/<bucket>/...`. |
| `source_bucket` | yes | First path segment: `internal`, `public_cad`, `public_nc`, `vendor`, etc. |
| `source_url_or_record` | yes | URL, PLM record, vendor contract, or dataset release note. |
| `license_status` | yes | `internal`, `public_domain`, `permissive`, `proprietary_authorized`, `non_commercial`, or `unverified`. |
| `license_source` | release usable | Required for release-usable statuses, including `internal`. |
| `release_floor_candidate` | yes | `yes` only when license and source bucket are release-eligible. |
| `part_family` | yes | Human-readable family such as housing, bracket, shaft, flange. |
| `topology_source` | yes | `verified` or `derived`. |
| `topology_evidence` | verified only | Reviewer note, CAD viewer screenshot ID, second-kernel report, or audit record. |
| `faces_min` | release case | Conservative lower bound from independent review for verified rows. |
| `edges_min` | release case | Conservative lower bound. |
| `solids_min` | release case | Can be 0 for valid surface-like parts. |
| `graph_nodes_min` | release case | Conservative lower bound. |
| `surface_types` | preferred | Expected surface categories when known. |
| `reviewer` | verified only | Human who checked license/topology. |
| `reviewed_at` | verified only | ISO date. |
| `notes` | optional | Any ambiguity or exclusion reason. |

Minimum target:

- 50-100 release-eligible STEP/IGES cases.
- Verified topology floor is enforced by the validator: at production floor this
  means at least `max(10, ceil(0.2 * release_eligible_count))`, capped at N.
- Non-commercial and unverified files may remain as coverage-only cases, but
  they must not enter the release floor.

## 3. Manufacturing Review Sheet

Use the generated reviewer template instead of editing benchmark manifests by
hand.

Reviewer-editable fields:

- `review_status`
- `reviewer`
- `reviewed_at`
- `reviewed_manufacturing_evidence_sources`
- `reviewed_manufacturing_evidence_payload_json`
- `review_notes`

Minimum target:

- At least 30 approved reviewed rows.
- Reviewer metadata is required.
- Source labels, payload labels, and detail payload coverage must be filled from
  human review, not copied from suggestions without review.

## 4. Command Flow

### B-Rep

Place real files under the bucketed tree:

```bash
mkdir -p data/brep_golden/internal data/brep_golden/public_cad data/brep_golden/public_nc
```

Generate the skeleton:

```bash
python scripts/build_brep_golden_manifest_skeleton.py \
  --root data/brep_golden \
  --manifest-root ../data/brep_golden \
  --output-json config/brep_golden_manifest.real.json
```

`manifest.root` is resolved relative to the manifest file. Because this manifest
is written under `config/`, use `../data/brep_golden` so validation resolves to
the repo-level `data/brep_golden` tree instead of `config/data/brep_golden`.

Fill every release candidate with:

- `part_family`
- `license`
- `license_status`
- `license_source`
- `expected_topology`
- `topology_source`
- `topology_evidence` when `topology_source=verified`
- remove all `TODO-*` tags after signoff

Validate the manifest:

```bash
python scripts/validate_brep_golden_manifest.py \
  --manifest config/brep_golden_manifest.real.json \
  --output-json reports/benchmark/brep_golden_manifest_validation.json \
  --fail-on-not-release-ready
```

Run strict B-Rep eval in the OCC-capable environment:

```bash
python scripts/eval_brep_step_dir.py \
  --manifest config/brep_golden_manifest.real.json \
  --strict \
  --output-dir reports/benchmark/brep_golden_eval
```

Feed both artifacts to the forward scorecard:

```bash
python scripts/export_forward_scorecard.py \
  --brep-summary reports/benchmark/brep_golden_eval/summary.json \
  --brep-manifest-validation-summary reports/benchmark/brep_golden_manifest_validation.json \
  --output-json reports/benchmark/forward_scorecard/latest.json \
  --output-md reports/benchmark/forward_scorecard/latest.md
```

### Manufacturing Review

Build or validate the review manifest from a benchmark results CSV:

```bash
python scripts/build_manufacturing_review_manifest.py \
  --from-results-csv reports/experiments/<run>/benchmark_results.csv \
  --output-csv reports/benchmark/forward_scorecard/manufacturing_review_manifest.csv \
  --summary-json reports/benchmark/forward_scorecard/manufacturing_review_manifest_validation.json \
  --progress-md reports/benchmark/forward_scorecard/manufacturing_review_manifest_progress.md \
  --gap-csv reports/benchmark/forward_scorecard/manufacturing_review_manifest_gaps.csv \
  --assignment-md reports/benchmark/forward_scorecard/manufacturing_review_assignment.md \
  --reviewer-template-csv reports/benchmark/forward_scorecard/manufacturing_reviewer_template.csv \
  --handoff-md reports/benchmark/forward_scorecard/manufacturing_review_handoff.md \
  --min-reviewed-samples 30 \
  --require-reviewer-metadata \
  --fail-under-minimum
```

Preflight a filled reviewer template:

```bash
python scripts/build_manufacturing_review_manifest.py \
  --validate-reviewer-template reports/benchmark/forward_scorecard/manufacturing_reviewer_template.filled.csv \
  --base-manifest reports/benchmark/forward_scorecard/manufacturing_review_manifest.csv \
  --summary-json reports/benchmark/forward_scorecard/manufacturing_reviewer_template_preflight.json \
  --reviewer-template-preflight-md reports/benchmark/forward_scorecard/manufacturing_reviewer_template_preflight.md \
  --reviewer-template-preflight-gap-csv reports/benchmark/forward_scorecard/manufacturing_reviewer_template_preflight_gaps.csv \
  --min-reviewed-samples 30 \
  --require-reviewer-metadata \
  --fail-under-minimum
```

Apply approved rows:

```bash
python scripts/build_manufacturing_review_manifest.py \
  --apply-reviewer-template reports/benchmark/forward_scorecard/manufacturing_reviewer_template.filled.csv \
  --base-manifest reports/benchmark/forward_scorecard/manufacturing_review_manifest.csv \
  --output-csv reports/benchmark/forward_scorecard/manufacturing_review_manifest.reviewed.csv \
  --summary-json reports/benchmark/forward_scorecard/manufacturing_review_manifest_apply.json \
  --reviewer-template-apply-audit-csv reports/benchmark/forward_scorecard/manufacturing_reviewer_template_apply_audit.csv \
  --min-reviewed-samples 30 \
  --require-reviewer-metadata \
  --fail-under-minimum
```

## 5. Stop Conditions

Stop and review manually when any of these occur:

- A public file lacks a license source.
- A file's license is ambiguous between commercial and non-commercial use.
- `public_nc` and `license_status=non_commercial` disagree.
- A release candidate still has `TODO-*` tags.
- A verified topology row lacks independent evidence.
- B-Rep validator reports `insufficient_verified_topology`.
- Manufacturing preflight reports missing reviewer metadata.
- Manufacturing reviewed fields match suggestions exactly with no reviewer notes
  or review trail.

## 6. Done Definition

B-Rep is evidence-ready when:

- Validator status is `release_ready`.
- Strict eval passes on the manifest.
- Forward scorecard shows no B-Rep manifest provenance gap.
- The verified-vs-derived counts are visible in the scorecard artifact.

Manufacturing labels are evidence-ready when:

- Review manifest validation status is `release_label_ready`.
- Gap CSV contains no blocking rows for the release floor.
- Forward scorecard no longer reports manufacturing review manifest gaps.

Only after both streams are stable should a separate policy PR consider making
B-Rep a hard release-decision blocker instead of a scorecard-visible weak lane.
