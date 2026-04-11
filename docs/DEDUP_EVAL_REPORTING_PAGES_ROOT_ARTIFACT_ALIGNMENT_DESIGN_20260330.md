# Eval Reporting Pages Root Artifact Alignment — Design

日期：2026-03-30

## Scope

Batch 8A of the Claude Collaboration Eval Reporting plan:

- Create `scripts/ci/assemble_eval_reporting_pages_root.py` thin assembler
- Update workflow to produce `eval-reporting-pages-<run>` artifact
- Update `deploy-pages` job to download the Pages-ready artifact instead of static report

## Design Decisions

### 1. Thin Assembler — `scripts/ci/assemble_eval_reporting_pages_root.py`

Pure file copy — reads existing generated files, copies to a flat Pages root:

| Source | Destination in Pages Root |
|---|---|
| `<eval-history-dir>/index.html` | `index.html` (landing page = public root) |
| `<eval-history-dir>/report_static/` | `report_static/` |
| `<eval-history-dir>/report_interactive/` | `report_interactive/` |
| `<eval-history-dir>/eval_reporting_bundle.json` | `eval_reporting_bundle.json` |
| `<eval-history-dir>/eval_reporting_bundle_health_report.json` | `eval_reporting_bundle_health_report.json` |
| `<eval-history-dir>/eval_reporting_index.json` | `eval_reporting_index.json` |
| `<eval-history-dir>/*.md` (canonical) | additive |

The assembler does NOT: regenerate landing page, re-render reports, recompute health/index/bundle.

### 2. Pages-Ready Artifact

New artifact: `eval-reporting-pages-${{ github.run_number }}`

Content: `reports/eval_pages/` (assembled by the assembler step).

### 3. deploy-pages Job

Changed from:
- Download `evaluation-report-${{ github.run_number }}` (static report only)

To:
- Download `eval-reporting-pages-${{ github.run_number }}` (landing + static + interactive + JSON)

The `./public` directory now contains the full Pages-ready root with landing page as `index.html`.

### 4. Owner Boundaries

The assembler is a thin wrapper. All content generation remains owned by their canonical scripts (landing page renderer, static report, interactive report, etc.).
