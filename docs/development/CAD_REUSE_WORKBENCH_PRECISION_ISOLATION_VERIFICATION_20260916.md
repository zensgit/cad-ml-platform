# ReviewReuse precision + tenant isolation — Verification

**Date**: 2026-09-16  
**Branch**: `eng/workbench-precision-isolation-20260915`  
**HEAD at writing**: see latest commit (`957546f6` 3.10/3.11 green; create_task off event loop)  
**PR**: https://github.com/zensgit/cad-ml-platform/pull/586  
**Plan**: `CAD_REUSE_WORKBENCH_PRECISION_ISOLATION_DEVELOPMENT_20260916.md`

---

## 1. Acceptance map

| Plan item | Evidence | Status |
|---|---|---|
| Local L4 only with geom-json pairs | `precision.py` `_try_l4_score`; tests `test_precision_scores_json_query_and_candidate_geom`, `test_precision_extracts_dxf_query_geom` | pass locally |
| No visual→geometric copy | `dedup_live.py` + `test_precision_labels_vision_only_without_copying_visual` | pass |
| Unverified numeric geom does not raise confidence | `evidence._top_confidence` + `test_unverified_numeric_geom_does_not_raise_confidence` | pass |
| Low L4 downgrades duplicate→different | `test_local_l4_reject_downgrades_duplicate_verdict`, `test_low_precision_reason` | pass |
| Stale precision-l4 stripped on vision-only | `test_stale_l4_method_stripped_on_vision_only` | pass |
| File gate includes PDF, rejects exe | `test_filename_gate`, `test_create_rejects_unsupported_file_type` | pass |
| Hashed tenant dirs, no `a/b` vs `a_b` collision | `test_filesystem_store_tenant_path_no_collision` | pass |
| list prefers hashed over stale legacy | `test_list_for_tenant_prefers_hashed_over_stale_legacy` | pass |
| cleanup `--tenant` by original id / hash | `test_cleanup_matches_hashed_tenant_by_original_id` | pass |
| Mixed legacy dir not rmtree'd | `test_cleanup_refuses_mixed_legacy_tenant_dir` | pass |
| JWT missing/invalid → 401 | `tests/unit/test_review_reuse_api.py` | pass |
| Isolated `--file` fail-closed | isolated-archive tests | pass |
| Decisions default-off | existing workbench + r2_hold tests | pass |
| R2 HOLD no feedback JSONL | `test_review_reuse_r2_hold.py` | pass |
| Track C / R11 / R12 not claimed | this PR body + board residual_human | n/a (human) |

**Not claimed:** owner design-lock ratification; production decision enable; customer pilot C1–C5; Track E model-release metrics.

## 2. Commands run

```bash
python3 -m flake8 --max-line-length=100 \
  src/core/review_reuse/precision.py \
  src/core/review_reuse/evidence.py \
  src/core/review_reuse/store.py \
  src/core/review_reuse/files.py \
  scripts/review_reuse_store_ops.py \
  tests/unit/test_review_reuse_*.py

make test-review-reuse
```

### Local results (Python 3.11, 2026-09-16)

| Run | Result |
|---|---|
| flake8 on changed ReviewReuse files | clean |
| `make test-review-reuse` | **115 passed**, 7 ezdxf warnings |

### CI (PR #586)

| Check | Result |
|---|---|
| tests (3.11) on `c01bc02b` | success |
| tests (3.10) on `957546f6` | **success** |
| tests (3.11) on `957546f6` | **success** |
| lint-all-report / openapi-fast / e2e-smoke | success on `c01bc02b` |
| mergeable_state | `blocked` (review required; do not merge) |

Re-run `make test-review-reuse` after each follow-up commit and record the count here.

## 3. Honesty probes (must stay true)

```text
# vision-only must not become high confidence
seed: visual=0.99, no geometric, methods=dedup2d-vision
→ rejection_reasons contains vision_only_unverified
→ evidence_pack.confidence.band == low, score == 0.0

# low L4 must not keep duplicate + high visual confidence
seed: geometric=0.1, visual=0.99, methods=precision-l4
→ low_precision_score; state=different; confidence from geometric only (0.1)

# mixed legacy a/b + a_b in one sanitized dir
cleanup --tenant a/b --apply → exit 1, mixed dir remains
```

## 4. Operator commands (decisions off)

```bash
# list (hashed tenants show original id)
python scripts/review_reuse_store_ops.py list --store-dir data/review_reuse_tasks --json

# cleanup is dry-run by default
python scripts/review_reuse_store_ops.py cleanup --store-dir data/review_reuse_tasks --older-than-days 30

# isolated archive, decisions stay off
make review-reuse-isolated-archive
# or FILE=path.dxf
```

## 5. Boundary check (this PR family)

Diff vs `origin/main` must **not** include:

- `src/core/feedback.py` writes / training JSONL
- `eval_integrity_gate` replacement
- `cost_cap` revive
- `REVIEW_REUSE_DECISIONS_ENABLED` default true

Confirm with:

```bash
git diff origin/main...HEAD --name-only
```

## 6. Residual after this wave

| Item | Owner |
|---|---|
| Python 3.10 CI job | **success** on `027c3bd5` |
| Human review + merge of #586 | owner / reviewer |
| R11 ratify, R12 decision enable | residual_human |
| Track C C1–C5 | residual_human |
