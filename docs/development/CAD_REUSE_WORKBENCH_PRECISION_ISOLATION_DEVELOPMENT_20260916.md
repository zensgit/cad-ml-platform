# ReviewReuse precision + tenant isolation — Development Plan

**Date**: 2026-09-16  
**Branch**: `eng/workbench-precision-isolation-20260915`  
**PR**: https://github.com/zensgit/cad-ml-platform/pull/586  
**Base**: `origin/main@22e3c77c` (post-#565)  
**Authority**: `CAD_REUSE_WORKBENCH_90_DAY_PLAN_20260807.md` §8, `PRODUCT_STRATEGY.md` §3.3  
**Design-lock**: `L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md` (PROPOSED)  
**Verification**: `CAD_REUSE_WORKBENCH_PRECISION_ISOLATION_VERIFICATION_20260916.md`  
**Board**: `CAD_REUSE_WORKBENCH_TASK_BOARD_20260808.md` O10 / O11 / SYS25

---

## 1. Goal

Close the honesty and isolation gaps that remained after the ReviewReuse MVP
(#547–#565) without opening residual_human work:

- local L4 precision only when both sides have geom-json;
- no visual→geometric copy;
- filesystem tenant dirs that cannot collide (`a/b` vs `a_b`);
- live recall asks for geometric / L4 and keeps ML off;
- JWT middleware e2e (no patched `_reviewer_id`);
- isolated-archive `--file` fail-closed;
- Day 61–90 *review-workflow* labels only.

## 2. Holds (do not weaken)

| Hold | Rule |
|---|---|
| Decision default-off | `REVIEW_REUSE_DECISIONS_ENABLED` unset → POST decision 403 |
| R2 HOLD | no `feedback.py` JSONL; audit export stays `audit_quarantine` |
| Track C / R11 / R12 | residual_human — do not claim complete |
| No Track E / cost_cap / PLM write-back | out of this PR |
| DWG | not auto-converted |
| L3 runtime WIP=1 | this PR is the single open runtime slot |

## 3. Model routing (12-hour unattended)

| Difficulty | Model | Work |
|---|---|---|
| D1 mechanical (docs, board, list JSON, tests) | grok-4.5 | markdown, board flips, operator list fields |
| D2 contract (precision labels, cleanup matching, JWT) | grok-4.6 | store/precision/evidence honesty |
| D3 “closed/safe” claims | grok-4.6 + Sol 5.6 | re-review vs `origin/main`; do not self-approve |
| Isolation / tenancy | grok-4.6 | hashed dirs, mixed-legacy refuse |

## 4. In scope (this PR)

| Area | Deliverable |
|---|---|
| Precision | `src/core/review_reuse/precision.py` — JSON/DXF query geom; candidate `geom_json` or geom-store load; lazy DXF extract; L4 `verification.level >= 4`; low score → `different` |
| File gate | `files.py` — `.dxf/.dwg/.pdf` + rasters |
| Adapter | preserve `geom_json`; live `enable_geometric=True`, `enable_ml=False` |
| Evidence | `_top_confidence` trusts geometric only with verified `precision-l4`; vision-only / missing geom / low precision stay low |
| Store | hashed `sha256(tenant_id)[:24]` + `tenant_meta.json`; unique tmp; skip rewrite if valid |
| Store ops | list merge; cleanup `--tenant` by id/meta/hash; refuse mixed legacy dirs |
| JWT | IntegrationAuthMiddleware e2e |
| Isolated archive | `--file` without seed, decisions off |
| Metrics | `false_duplicate` / `missed_reuse` / `usefulness:1-5` |

## 5. Out of scope

- Owner ratify (R11) or enabling decisions (R12)
- Training JSONL / Track E eval_integrity_gate
- Automatic DWG→DXF
- Merging #586 (branch protection BLOCKED; needs human review)
- Inventing Track C contacts / measured pilot

## 6. Sequence

1. Precision pass + file gate + hashed store (landed).
2. Live geometric + JWT e2e + `--file` (landed).
3. Sol 5.6 loop: hashed-vs-legacy list, cleanup matching, DXF extract, mixed refuse, EvidencePack verdict (landed through `c01bc02b`).
4. This document + verification MD + operator list `mixed` flag.
5. CI babysit (Python 3.10 job on #586).
6. Unattended waves: fix CI, Sol re-review, no new L3 runtime track.

## 7. Risk

| Risk | Mitigation |
|---|---|
| Fake L4 from visual | never copy visual→geometric; unverified numeric geom does not raise confidence |
| Cross-tenant delete | mixed legacy dirs refused (`refused_mixed`, exit 1) |
| Temp-file race on `tenant_meta.json` | `mkstemp` unique name; skip rewrite when sidecar valid |
| DXF parse cost on offline insufficient | extract only when a candidate can be scored |
| 12h unattended overreach | scheduler stops ~2026-09-16 17:00 UTC; never merge; never enable decisions |
