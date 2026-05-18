# CAD ML Reviewer Template Preflight Gap CSV Development

Date: 2026-05-14

## Goal

Make filled reviewer-template preflight failures machine-readable. The Markdown
preflight report is useful for human triage, but reviewer closeout also needs a
CSV that can be filtered, assigned, and diffed without parsing Markdown.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Adds `REVIEWER_TEMPLATE_PREFLIGHT_GAP_COLUMNS`.
  - Adds `build_reviewer_template_preflight_gap_rows`.
  - Adds `--reviewer-template-preflight-gap-csv`.
  - Writes one CSV row for each blocking filled-template row.
  - Includes row identity, file metadata, status, reviewer metadata, blocking
    reasons, duplicate-row flag, source/payload/detail readiness, reviewed fields,
    review notes, and suggested fields.
- Updated handoff Markdown generation.
  - Adds the preflight gap CSV path to the artifact map.
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds
    `FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_GAP_CSV`.
  - Passes the gap CSV path into reviewer-template preflight.
  - Passes the same path into review handoff generation.
  - Emits `manufacturing_reviewer_template_preflight_gap_csv` as a GitHub output.
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for the preflight gap CSV.
  - Uploads the CSV with reviewer-template preflight artifacts.
- Updated targeted tests for:
  - preflight gap row construction
  - CLI gap CSV output
  - CI output wiring in ready and blocked preflight modes
  - workflow env and upload artifact wiring
- Updated Phase 6 TODO.

## CSV Contract

The CSV is diagnostic only. It reports blockers such as:

- duplicate `row_id`
- missing source labels
- missing payload labels
- missing `details.*` payload labels
- unapproved review status
- missing reviewer metadata when required

It does not apply labels, merge manifests, or approve suggestions.

## Default CI Path

```text
reports/benchmark/forward_scorecard/manufacturing_reviewer_template_preflight_gaps.csv
```

Override with:

```text
FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_GAP_CSV=<path>
```

## Release Impact

Blocked reviewer-template preflight now produces both human-readable and
machine-readable evidence. Reviewers can use the CSV for closeout tracking, and
CI can upload it even when fail-on-blocked mode stops the job.

## Remaining Work

- Run the gap CSV against the first real filled reviewer template.
- Use the CSV to assign and close blocking template rows.
- Apply only a preflight-ready template.
- Tune thresholds after the reviewed release set is stable.
