# Unattended autonomy runbook — CAD Reuse Workbench

**Date**: 2026-08-08  
**Purpose**: Continue 90-day plan engineering without interactive confirmation.

## Default autonomy policy

When the owner is away:

1. **Fix CI** on open workbench PRs (especially #547) without asking.
2. **Implement residual_eng** inside the single L3 runtime slot (#547) — no second L3 PR.
3. **Push** commits; refresh OpenAPI when routes change.
4. **Do not** invent Track C customers or enable decisions in production.
5. **Merge** only when required CI is green and branch protection allows (use `gh pr merge` when clean).

## Hard human blockers (cannot automate)

| Item | Why |
|---|---|
| Design-lock owner ratification | Legal/product authority |
| `REVIEW_REUSE_DECISIONS_ENABLED=true` in pilot | Explicit owner enable |
| Track C contacts / paid pilot / commercial | Real-world humans |
| Day-90 strategy clock change | Strategy amendment |

## Automation installed

| Mechanism | Role |
|---|---|
| Grok scheduler `019fdd5c18f2` | Every 15m babysit #547 / residual |
| Task `cad-reuse-workbench-gap-check` | Weekday gap inventory |
| Workflows `cad-reuse-workbench-{dev,system,90d}` | On-demand / verify |

## Operator offline checklist

```bash
# Local truth
pytest tests/unit/test_review_reuse_*.py -q
gh pr checks 547
gh pr view 547 --json state,mergeable,statusCheckRollup
# When green:
gh pr merge 547 --merge  # or --squash per repo default
```
