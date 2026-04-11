# Eval Reporting Landing Page Discovery Surface Alignment — Design

日期：2026-03-30

## Scope

Batch 6A of the Claude Collaboration Eval Reporting plan:

- Create `scripts/generate_eval_reporting_landing_page.py` as a lightweight HTML landing/discovery page
- The page only reads existing canonical artifacts; it does NOT materialize, summarize, or render charts

## Design Decisions

### 1. Landing Page Renderer — `scripts/generate_eval_reporting_landing_page.py`

Single-file static HTML output. Consumes three canonical JSON artifacts:

| Artifact | Default Path | Purpose |
|---|---|---|
| `eval_reporting_index.json` | `<eval-history-dir>/eval_reporting_index.json` | Discovery pointers |
| `eval_reporting_stack_summary.json` | `reports/ci/eval_reporting_stack_summary.json` | Status / health counts |
| `eval_reporting_bundle_health_report.json` | `<eval-history-dir>/eval_reporting_bundle_health_report.json` | Per-check health |

### 2. Page Content

The landing page renders:

- Overall status badge (`ok` / `degraded` / `missing`)
- Missing / Stale / Mismatch summary counts
- Artifact link table (Static Report, Interactive Report, Eval Signal Bundle, History Sequence Bundle, Top-Level Bundle, Health Report)
- Health checks detail table (name, status badge, detail)
- Missing artifact warnings when any input JSON is absent

### 3. What the Page Does NOT Do

- Does not embed or duplicate static/interactive report content
- Does not render charts or trend images
- Does not compute or re-aggregate any metrics
- Does not introduce new JS dependencies
- Does not own a new summary schema

### 4. Missing Artifact Handling

When any of the three input JSONs is missing:
- Page still renders (graceful degradation)
- A visible warning box lists which artifacts are missing
- Link table shows "missing" badges for empty paths

### 5. CLI Interface

```
--eval-history-dir   (default: reports/eval_history)
--index-json         (override index path)
--stack-summary-json (override summary path)
--health-json        (override health path)
--out                (default: <eval-history-dir>/index.html)
```

### 6. Owner Boundaries

The renderer does NOT own:
- Bundle materialization
- Summary/metrics computation
- Health check execution
- Weekly/trend generation
- Pages/workflow orchestration (deferred to Batch 6B)
