# CAD ML Manufacturing Reviewer Template Preflight Report Development

Date: 2026-05-14

## Goal

Make reviewer-template preflight actionable for human reviewers and CI triage.
The existing JSON summary is useful for automation, but reviewer closeout needs a
small Markdown artifact that lists blocking rows, reasons, and next actions before
the filled template is applied to the full review manifest.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Adds `--reviewer-template-preflight-md`.
  - Adds `--max-preflight-rows` for bounding the blocking-row section.
  - Emits a Markdown report when `--validate-reviewer-template` runs.
  - Reuses the same validation rules as the JSON summary, including:
    - duplicate `row_id`
    - missing or unsupported `review_status`
    - missing source labels
    - missing payload labels
    - missing `details.*` payload labels
    - optional reviewer metadata checks
  - Includes reviewer-oriented next actions without treating machine suggestions as
    approved labels.
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds
    `FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MD`.
  - Writes the Markdown artifact beside the preflight JSON summary.
  - Emits `manufacturing_reviewer_template_preflight_md` as a CI output.
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for the Markdown output path.
  - Uploads the Markdown report with the reviewer-template preflight artifact.
- Updated targeted tests for:
  - Markdown content generation from ready and blocked rows.
  - CLI preflight writing both JSON and Markdown.
  - wrapper output wiring for ready and blocked preflight.
  - workflow env and upload path wiring.
- Updated the Phase 6 TODO item for the human-readable preflight artifact.

## CI Usage

The default Markdown output path is:

```text
reports/benchmark/forward_scorecard/manufacturing_reviewer_template_preflight.md
```

Release CI can override it with:

```text
FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MD=<path>
```

When `FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV` points to a
filled template, the optional scorecard wrapper now writes both:

```text
manufacturing_reviewer_template_preflight.json
manufacturing_reviewer_template_preflight.md
```

If preflight is blocked, the Markdown report is still produced before the wrapper
decides whether to fail through the fail-on-blocked flag.

## Report Contract

The Markdown report contains:

- Overall status.
- Ready and blocking row counts.
- Issue counts grouped by validation reason.
- Duplicate `row_id` values.
- A bounded list of blocking rows with file path, row id, status, reasons, and
  reviewer next action.

The report is intentionally diagnostic. It does not merge labels, approve rows,
or change the review manifest.

## Release Impact

The reviewer-template path now has a human-readable stop point before mutation:

1. Reviewer fills the template.
2. CI runs preflight.
3. CI uploads JSON and Markdown preflight artifacts.
4. Reviewers fix any blocking rows from the Markdown report.
5. Only a ready template proceeds to apply, validation, merge, and scorecard use.

This keeps the release gate auditable for non-automation reviewers while
preserving the existing machine-readable summary.

## Remaining Work

- Run the preflight report against the first real filled reviewer template.
- Use the Markdown artifact to close source, payload, and detail label gaps.
- Populate real domain-approved release labels.
- Tune release thresholds after the reviewed benchmark set is stable.
