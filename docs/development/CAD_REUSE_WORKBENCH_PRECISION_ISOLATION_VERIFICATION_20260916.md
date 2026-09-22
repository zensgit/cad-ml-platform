# ReviewReuse precision + tenant isolation — Verification

**Date**: 2026-09-16  
**Branch**: `eng/workbench-precision-isolation-20260915`  
**HEAD at writing**: window 4 start `dc9267e0` (product `adcd3b6e`)  
**PR**: https://github.com/zensgit/cad-ml-platform/pull/586  
**Plan**: `CAD_REUSE_WORKBENCH_PRECISION_ISOLATION_DEVELOPMENT_20260916.md`

---

## 1. Acceptance map

| Plan item | Evidence | Status |
|---|---|---|
| Local L4 only with geom-json pairs | `precision.py` `_try_l4_score`; tests `test_precision_scores_json_query_and_candidate_geom`, `test_precision_extracts_dxf_query_geom` | pass locally |
| Pre-scored L4 exports level >= 4 | `test_seeded_l4_without_match_level_exports_level_4` | pass locally |
| Local L4 drops stale unverified reasons | `test_local_l4_clears_stale_vision_only_reason` | pass locally |
| No visual→geometric copy | `dedup_live.py` + `test_precision_labels_vision_only_without_copying_visual` | pass |
| Unverified numeric geom does not raise confidence | `evidence._top_confidence` + `test_unverified_numeric_geom_does_not_raise_confidence` | pass |
| Low L4 downgrades duplicate→different | `test_local_l4_reject_downgrades_duplicate_verdict`, `test_low_precision_reason` | pass |
| Stale precision-l4 stripped on vision-only | `test_stale_l4_method_stripped_on_vision_only` | pass |
| File gate includes PDF, rejects exe | `test_filename_gate`, `test_create_rejects_unsupported_file_type` | pass |
| File gate includes geometry JSON | `test_create_accepts_geom_json_filename` | pass locally |
| Occupied hashed dir not overwritten | `test_filesystem_put_refuses_hashed_dir_occupied_by_legacy_tenant` | pass locally |
| Hashed tenant dirs, no `a/b` vs `a_b` collision | `test_filesystem_store_tenant_path_no_collision` | pass |
| list prefers hashed over stale legacy | `test_list_for_tenant_prefers_hashed_over_stale_legacy` | pass |
| cleanup `--tenant` by original id / hash | `test_cleanup_matches_hashed_tenant_by_original_id` | pass |
| Mixed legacy dir not rmtree'd | `test_cleanup_refuses_mixed_legacy_tenant_dir` | pass |
| JWT missing/invalid → 401 | `tests/unit/test_review_reuse_api.py` | pass |
| Isolated `--file` fail-closed | isolated-archive tests | pass |
| Decisions default-off | existing workbench + r2_hold tests | pass |
| R2 HOLD no feedback JSONL | `test_review_reuse_r2_hold.py` | pass |
| Track C / R11 / R12 not claimed | this PR body + board residual_human | n/a (human) |
| Pipeline commit cannot clobber cancel/decision | `store.update_atomically`; `test_pipeline_honors_mid_flight_cancel`; `test_filesystem_update_atomically_keeps_terminal_status` | pass locally |
| FS updates atomic across worker processes | `_StoreFileLock` flock; `test_filesystem_update_atomically_serializes_processes`; `test_filesystem_put_new_idempotent_serializes_processes` | pass locally |
| Unique tmp for task/idem/meta writes | `_atomic_write_text`; `test_filesystem_atomic_write_uses_unique_tmp` | pass locally |
| Empty DXF extract is not L4 geom | `test_empty_dxf_extract_is_not_l4_geometry` | pass locally |
| Malformed entity dict is not L4 geom | `test_malformed_entity_dict_is_not_l4_geometry` | pass locally |
| Negative-radius circle is not L4 geom | `test_negative_radius_circle_is_not_l4_geometry` | pass locally |
| Zero-length polyline is not L4 geom | `test_zero_length_polyline_is_not_l4_geometry` | pass locally |
| Block INSERT is L4 geom | `test_insert_block_geom_is_l4_geometry` | pass locally |
| Zero-ratio ellipse is not L4 geom | `test_zero_ratio_ellipse_is_not_l4_geometry` | pass locally |
| Non-finite / out-of-range pre-scored L4 | `test_nonfinite_prescored_l4_is_not_trusted`, `test_out_of_range_prescored_l4_is_not_trusted` | pass locally |
| Cancel on occupied hashed dir is 409 | `test_cancel_translates_occupied_hashed_dir` | pass locally |
| Mid-flight decision pack rebuilt with candidates | `test_pipeline_rebuilds_pack_after_mid_flight_decision` | pass locally |
| Terminal merge keeps pipeline events | same tests assert `recall_completed` / `precision_completed` / `evidence_pack_ready` | pass locally |
| Live recall timeout without running loop | `test_run_coro_applies_timeout_without_running_loop` | pass locally |
| Legacy dir named like another hash not cleaned | `test_cleanup_legacy_dir_named_like_other_hash_is_not_selected` | pass locally |
| Meta vs task tenant disagreement refuses cleanup | `test_cleanup_refuses_when_meta_disagrees_with_task_tenant` | pass locally |
| Decision-before-evidence not counted as 0s review | mid-flight decision test asserts `median_review_time_seconds is None` | pass locally |
| High L4 restores stale low-precision `different` | `test_rescore_clears_stale_low_precision_reason` | pass locally |
| High L4 keeps independent version-gate `different` | `test_high_l4_preserves_version_gate_different`, `test_high_prescored_l4_preserves_version_gate_different` | pass locally |
| High L4 keeps reasonless adapter `different` | `test_high_l4_preserves_reasonless_different`, `test_high_prescored_l4_preserves_reasonless_different` | pass locally |
| Mixed valid + degenerate supported entities not L4 | `test_mixed_valid_line_and_zero_radius_circle_is_not_l4` | pass locally |
| Unknown TEXT beside a valid LINE still L4 | `test_valid_line_with_unknown_type_still_l4` | pass locally |
| Non-finite / out-of-range local L4 not trusted | `test_nonfinite_local_l4_score_is_not_trusted`, `test_out_of_range_local_l4_score_is_not_trusted` | pass locally |
| Unreadable tenant_meta.json is occupied | `test_filesystem_put_refuses_unreadable_tenant_meta` | pass locally |

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
| `make test-review-reuse` (pre-P2, `dc2c8fec`) | **115 passed**, 7 ezdxf warnings |
| `make test-review-reuse` (Sol P2 wave) | **121 passed**, 7 ezdxf warnings |
| `make test-review-reuse` (Sol P1/P2 follow-up) | **122 passed**, 7 ezdxf warnings |
| `make test-review-reuse` (L4 level + stale reasons) | **124 passed**, 7 ezdxf warnings |
| `make test-review-reuse` (meta-vs-task refuse + review-time) | **125 passed**, 7 ezdxf warnings |
| tests (3.10) on `19545bdf` | **fail** — `asyncio.TimeoutError` is not `TimeoutError` on 3.10 |
| tests (3.11) / lint-type on `19545bdf` | **pass** |
| `make test-review-reuse` after sanitizing pipeline errors | **125 passed**, 7 ezdxf warnings |
| `make test-review-reuse` at `18b497f2` (wave 11:28 UTC) | **125 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) / lint-type on `f80d06a3` | **pass** |
| `make test-review-reuse` after geometric-only confidence P1 | **127 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) on `2bd32083` | **pass** |
| `make test-review-reuse` after occupied-dir + json gate | **129 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) on `2f76c87c` | **pass** |
| `make test-review-reuse` after geom-json schema + loop timeout | **132 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) / lint-type on `4d4bb285` | **pass** |
| `make test-review-reuse` at `4d4bb285` (wave 14:30 UTC) | **132 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) on `33a8315c` | **pass** |
| `make test-review-reuse` after insufficient-evidence confidence skip | **133 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) on `5fb2096f` | **pass** |
| `make test-review-reuse` after store-geom + grouped cleanup | **135 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) on `6b29ab81` | **pass** |
| `make test-review-reuse` after inline-geom store fallback | **136 passed**, 7 ezdxf warnings |
| Unattended deadline 2026-09-16 17:00 UTC | stopped at `704fd1c6`; last green 3.10/3.11 was `6b29ab81` |
| tests (3.10) / tests (3.11) / lint-type on `7002bfa2` | **pass** (2026-09-17 01:00 UTC) |
| Sol wave18 vs `origin/main` on `7002bfa2` | no P1/P2 |
| Sol wave19 vs `origin/main` on `f58e80ee` | P1 FS inter-process lock; P2 empty DXF `_is_geom_json` |
| `make test-review-reuse` after FS flock + empty DXF gate | **140 passed**, 7 ezdxf warnings |
| Sol wave20 vs `origin/main` on `ac2ef01d` | P2 malformed entity L4; P2 cancel occupied-dir 409 |
| `make test-review-reuse` after entity-shape + cancel 409 | **142 passed**, 7 ezdxf warnings |
| Sol wave21 vs `origin/main` on `f049a14c` | P2 negative/NaN/inf degenerate primitives still L4 |
| `make test-review-reuse` after finite positive radius | **143 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) / lint-type on `52c820b5` | **pass** |
| Sol wave22 vs `origin/main` on `52c820b5` | P2 zero-length polyline `[[0,0],[0,0]]` still L4 |
| `make test-review-reuse` after distinct polyline points | **144 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) on `4a9a390a` | **pass** |
| Sol wave23 vs `origin/main` on `4a9a390a` | P2 INSERT; P2 ellipse ratio; P2 pre-scored L4 range |
| `make test-review-reuse` after INSERT + ellipse + pre-scored L4 | **148 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) on `d2d0beb5` | **pass** |
| Sol wave24 vs `origin/main` on `d2d0beb5` | P2 drop out-of-range geom score; zero-scale INSERT; zero-sweep ARC |
| `make test-review-reuse` after scale/sweep/score-clear | **150 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) on `a999cc63` | **pass** |
| Sol wave25 vs `origin/main` on `a999cc63` | P2 boolean coords (`true`/`false`) still L4 |
| `make test-review-reuse` after rejecting bool numerics | **151 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) / lint-type on `c9a7ce4b` | **pass** |
| Sol wave26 vs `origin/main` on `c9a7ce4b` | no P1/P2 |
| tests (3.10) / tests (3.11) / e2e-smoke on `1ae5be3b` | **pass** |
| Sol wave27 vs `origin/main` on `1ae5be3b` | P2 lowercase entity type; P2 boolean precision_score |
| `make test-review-reuse` after type canonicalize + bool scores | **153 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) / e2e-smoke on `d6d9a1c9` | **pass** |
| Sol wave28 vs `origin/main` on `d6d9a1c9` | P2 stale `low_precision_score` after high L4 rescore |
| `make test-review-reuse` after clearing stale low-precision | **154 passed**, 7 ezdxf warnings |
| High L4 rescore restores state/verdict | `test_rescore_clears_stale_low_precision_reason` asserts not `different` |
| Version-gate `different` does not raise pack confidence | `test_high_l4_preserves_version_gate_different` asserts confidence 0.0 |
| Zero-sweep ELLIPSE is not L4 geom | `test_zero_sweep_ellipse_is_not_l4_geometry` |
| Window 3 start `make test-review-reuse` after verdict restore | **154 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) / e2e-smoke on `23a35ec9` | **pass** |
| Sol wave30 vs `origin/main` on `23a35ec9` | P2 high L4 must not promote independent `different` |
| `make test-review-reuse` after preserving version-gate different | **156 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) / e2e-smoke on `b98e23a0` | **pass** |
| Sol wave31 vs `origin/main` on `b98e23a0` | P2 restore similar only if stale `low_precision_score` existed |
| `make test-review-reuse` after reasonless-different preserve | **158 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) / e2e-smoke on `d46c6156` | **pass** |
| Sol wave32 vs `origin/main` on `d46c6156` | **no P1/P2** |
| tests (3.10) / tests (3.11) / e2e-smoke on `1211c01f` | **pass** |
| Sol wave33 vs `origin/main` on `1211c01f` | **blocked** (Codex usage limit until 2026-09-23 22:47 UTC) |
| tests (3.10) / tests (3.11) / e2e-smoke on `58757c69` | **pass** |
| Sol wave34 vs `origin/main` on `58757c69` | **blocked** (same Codex usage limit) |
| tests (3.10) / tests (3.11) / e2e-smoke on `800bb4fd` | **pass** |
| Sol wave35 vs `origin/main` on `800bb4fd` | **blocked** (same Codex usage limit) |
| tests (3.10) / tests (3.11) / e2e-smoke on `c231f79c` | **pass** |
| Sol wave36 vs `origin/main` on `c231f79c` | **blocked** (same Codex usage limit) |
| tests (3.10) / tests (3.11) / e2e-smoke on `a04f3b91` | **pass** |
| Sol wave37 vs `origin/main` on `a04f3b91` | **blocked** (same Codex usage limit) |
| tests (3.10) / tests (3.11) / e2e-smoke on `9a686f13` | **pass** |
| Sol wave38 vs `origin/main` on `9a686f13` | **blocked** (same Codex usage limit) |
| tests (3.10) / tests (3.11) / e2e-smoke on `cbd987f3` | **pass** |
| Sol wave39 vs `origin/main` on `cbd987f3` | **blocked** (same Codex usage limit) |
| tests (3.10) / tests (3.11) / e2e-smoke on `aae99b6e` | **pass** |
| Sol wave40 vs `origin/main` on `aae99b6e` | **blocked** (same Codex usage limit) |
| tests (3.10) / tests (3.11) / e2e-smoke on `077c188a` | **pass** |
| Sol wave41 vs `origin/main` on `077c188a` | **blocked** (same Codex usage limit) |
| tests (3.10) / tests (3.11) / e2e-smoke on `472042c7` | **pass** |
| Sol wave42 vs `origin/main` on `472042c7` | **blocked** (same Codex usage limit) |
| tests (3.10) / tests (3.11) / e2e-smoke on `715401d0` | **pass** |
| Sol wave43 vs `origin/main` on `715401d0` | **blocked** (same Codex usage limit) |
| tests (3.10) / tests (3.11) / e2e-smoke on `e5da5144` | **pass** |
| Sol wave44 vs `origin/main` on `e5da5144` | **blocked** (same Codex usage limit) |
| tests (3.10) / tests (3.11) / e2e-smoke on `f2acff4f` | **pass** |
| Sol wave45 vs `origin/main` on `f2acff4f` | **blocked** (same Codex usage limit) |
| `make test-review-reuse` after independent-reject confidence skip | **158 passed**, 7 ezdxf warnings |
| Zero-sweep ELLIPSE is not L4 geom | `test_zero_sweep_ellipse_is_not_l4_geometry` |
| Omitted ellipse params match DXF full span | `test_omitted_ellipse_params_match_full_span` |
| Non-finite INSERT rotation is not L4 geom | `test_nonfinite_insert_rotation_is_not_l4_geometry` |
| Sol wave50 vs `origin/main` on `2c32cae0` | P2 omitted ellipse params; P2 NaN INSERT rotation |
| `make test-review-reuse` after ellipse/INSERT P2 | **161 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) / e2e-smoke on `4858606b` | **pass** |
| Sol wave51 vs `origin/main` on `4858606b` | **no P1/P2** |
| Sol wave53 vs `origin/main` on `b76c67d8` | P2 mixed valid + degenerate supported entities still L4 |
| `make test-review-reuse` after mixed-entity L4 gate | **163 passed**, 7 ezdxf warnings |
| Sol wave54 vs `origin/main` on `21c0dfbc` | P2 live hits drop inline `geom_json` |
| Live match inline geom_json still L4 | `test_vision_response_forwards_inline_geom_json` |
| `make test-review-reuse` after live geom_json forward | **164 passed**, 7 ezdxf warnings |
| Sol wave55 vs `origin/main` on `d31a9942` | P1 fused semantic score stored as geometric L4 |
| Shared TEXT does not certify different geometry | `test_shared_text_does_not_certify_different_geometry` |
| `make test-review-reuse` after geometry-only L4 | **168 passed**, 7 ezdxf warnings |
| Sol wave54 remaining vs `origin/main` on `d31a9942` | P2 non-finite local L4; P2 unreadable tenant meta fail-open |
| `make test-review-reuse` after local L4 range + unreadable meta | **167 passed**, 7 ezdxf warnings |
| Sol wave56 vs `origin/main` on `fa3eb91d` | P1 bag-of-features re-enabled; P1 live fused score as L4 |
| Layout-shifted clones not certified | `test_layout_shifted_same_primitives_are_not_certified` |
| Live fused score does not skip local L4 | `test_live_fused_precision_score_does_not_skip_local_l4` |
| `make test-review-reuse` after wave56 P1s | **170 passed**, 7 ezdxf warnings |
| Sol wave57 vs `origin/main` on `a3d97c35` | P1 `CAD_ML_PLATFORM_L4_ENTITIES_GEOM_HASH=1` re-enables bag-of-features |
| Env-on layout-shifted clones not certified | `test_layout_shifted_clones_not_certified_when_geom_hash_env_on` |
| `make test-review-reuse` after env geom-hash pin | **171 passed**, 7 ezdxf warnings |
| Sol wave58 vs `origin/main` on `b86733a4` | P2 cleanup deletes dirs with unreadable tenant_meta.json |
| Cleanup refuses unreadable sidecar | `test_cleanup_refuses_unreadable_tenant_meta` |
| `make test-review-reuse` after cleanup meta P2 | **172 passed**, 7 ezdxf warnings |
| Sol wave59 vs `origin/main` on `b86733a4` | P2 sub-0.001 primitives collapse after 3-decimal rounding |
| Tiny orthogonal LINEs are not L4 | `test_sub_millimeter_orthogonal_lines_are_not_l4_geometry` |
| `make test-review-reuse` after quantized geom P2 | **173 passed**, 7 ezdxf warnings |
| Sol wave60 vs `origin/main` on `9806f46b` | P2 HATCH proxies inflate geometry-only L4 |
| Shared HATCH does not certify different geometry | `test_shared_hatch_does_not_certify_different_geometry` |
| `make test-review-reuse` after HATCH exclusion | **174 passed**, 7 ezdxf warnings |
| Sol wave61 vs `origin/main` on `256068ce` | P2 per-entity layer penalty rejects identical geometry |
| Layer mismatch does not reject identical LINEs | `test_layer_mismatch_does_not_reject_identical_geometry` |
| `make test-review-reuse` after layer-penalty pin | **175 passed**, 7 ezdxf warnings |
| Sol wave62 vs `origin/main` on `48ec1243` | P1 translated polylines certified via canonical matching |
| Shifted polylines are not L4 | `test_layout_shifted_polylines_are_not_certified` |
| `make test-review-reuse` after polyline explode | **177 passed**, 7 ezdxf warnings |
| Sol wave63 vs `origin/main` on `4362c886` | P1 exploded polylines truncated at 64 entities |
| Long polylines diverging after cap are not L4 | `test_long_polylines_diverging_after_matcher_cap_are_not_certified` |
| `make test-review-reuse` after matcher-cap raise | **178 passed**, 7 ezdxf warnings |
| Sol wave64 vs `origin/main` on `1a99996f` | P1 unmatched extra entities; P1 unbounded matcher cap |
| Extra unmatched LINE is not L4 1.0 | `test_extra_unmatched_line_is_not_certified` |
| Over-cap geometry is not prefix-certified | `test_over_cap_geometry_is_not_certified_as_l4` |
| `make test-review-reuse` after unmatched penalty + hard cap | **180 passed**, 7 ezdxf warnings |
| Sol wave65 vs `origin/main` on `167831f6` | P1 bulged polyline exploded to chord; P1 INSERT hash 0.5556 L4 |
| Bulged polyline is not straight L4 | `test_bulged_polyline_is_not_certified_as_straight_l4` |
| Mismatched INSERT block_hash is not L4 | `test_mismatched_insert_block_hash_is_not_certified` |
| `make test-review-reuse` after bulge/hash P1s | **182 passed**, 7 ezdxf warnings |
| Sol wave66 vs `origin/main` on `66205333` | P1 DXF extract drops bulge; P1 spline matcher truncates at 16 |
| DXF-extracted bulge is not chord L4 | `test_dxf_extracted_bulge_is_not_certified_as_straight_l4` |
| Long splines are not prefix-certified | `test_long_splines_truncated_by_matcher_are_not_l4` |
| `make test-review-reuse` after DXF bulge + spline cap | **184 passed**, 7 ezdxf warnings |
| Sol wave67 vs `origin/main` on `0414daca` | P1 quantized bulge; spline prefix; swapped INSERT hashes; half-ellipses |
| `make test-review-reuse` after wave67 P1s | **188 passed**, 7 ezdxf warnings |
| Sol wave68 vs `origin/main` on `0414daca` | P1 INSERT without hash; P1 stale extract cache; P2 mixed-group cleanup |
| INSERT without block_hash is not L4 | `test_insert_without_block_hash_is_not_l4_geometry` |
| Legacy extract cache is re-extracted | `test_legacy_dxf_extract_cache_without_version_is_reextracted` |
| Mixed sibling refuses whole tenant group | `test_cleanup_refuses_whole_group_when_sibling_is_mixed` |
| `make test-review-reuse` after hash/cache/cleanup P1s | **191 passed**, 7 ezdxf warnings |
| Sol wave70 vs `origin/main` on `89a6667c` | P1 mixed INSERT dropped; P1 incomplete SPLINE identity |
| LINE+INSERT is not L4 | `test_line_plus_insert_is_not_certified_as_l4` |
| `make test-review-reuse` after unscored INSERT/SPLINE | **194 passed**, 7 ezdxf warnings |
| Sol wave72 vs `origin/main` on `9dfc3b87` | P1 leftover geom types; P2 similar+independent reject confidence |
| LINE+POINT is not L4 | `test_line_plus_point_is_not_certified_as_l4` |
| similar version-gate confidence stays 0 | `test_similar_version_gate_does_not_raise_confidence` |
| `make test-review-reuse` after leftover-geom + confidence P2 | **198 passed**, 7 ezdxf warnings |
| Sol wave73 vs `origin/main` on `01ec8379` | P1 live geom_json tenant leak; P2 polyline width explode; P2 $INSUNITS |
| Live geom_json is L4-only then stripped | `test_precision_strips_inline_geom_json_after_l4` |
| GET/EvidencePack/audit have no geom_json | `test_task_export_does_not_include_candidate_geom_json` |
| Thick polyline is not zero-width L4 | `test_wide_polyline_is_not_certified_as_zero_width_l4` |
| Inch vs mm same numbers are not L4 | `test_dxf_inch_vs_mm_same_numbers_are_not_l4` |
| v2 extract cache without width is re-extracted | `test_legacy_v2_extract_cache_without_width_is_reextracted` |
| `make test-review-reuse` after geom-strip/width/units | **208 passed**, 7 ezdxf warnings |
| Sol wave74 vs `origin/main` on `e3357abd` | P1 wrapped ARC sweep; P2 fractional INSUNITS |
| Wrapped ARC sweeps are not L4 | `test_wrapped_arc_sweeps_are_not_certified` |
| Fractional INSUNITS is not L4 | `test_fractional_insunits_is_not_certified` |
| `make test-review-reuse` after ARC sweep + integral units | **210 passed**, 7 ezdxf warnings |
| Sol wave75 vs `origin/main` on `f45c6586` | P2 swapped multi-arc sweeps |
| Swapped multi-arc sweeps are not L4 | `test_swapped_multi_arc_sweeps_are_not_certified` |
| `make test-review-reuse` after center-bound ARC sweeps | **212 passed**, 7 ezdxf warnings |
| Sol wave76 vs `origin/main` on `04e97e8d` | P1 concentric ARC sweep swap |
| Concentric multi-arc sweep swaps are not L4 | `test_swapped_concentric_arc_sweeps_are_not_certified` |
| `make test-review-reuse` after radius-bound ARC sweeps | **214 passed**, 7 ezdxf warnings |
| Sol wave77 vs `origin/main` on `cc28e09b` | P2 greedy ARC reorder false reject |
| Reordered nearby ARCs still L4 | `test_reordered_nearby_arcs_still_l4` |
| `make test-review-reuse` after bipartite ARC pairing | **215 passed**, 7 ezdxf warnings |
| Sol wave79 vs `origin/main` on `dfb0efe9` | P1 nearby ARC sweep swap; P2 string polyline closed |
| Nearby ARC sweep swaps are not L4 | `test_swapped_nearby_arc_sweeps_are_not_certified` |
| String ``closed: "false"`` is not L4 | `test_string_false_polyline_closed_is_not_certified` |
| `make test-review-reuse` after nearest-center ARC + closed bool | **217 passed**, 7 ezdxf warnings |
| Sol wave80 vs `origin/main` on `0604807b` | P1 cap before explode; P2 ARC radius in assignment |
| Reordered concentric near-radii still L4 | `test_reordered_concentric_near_radii_still_l4` |
| `make test-review-reuse` after pre-explode cap + radius cost | **218 passed**, 7 ezdxf warnings |
| Sol wave81 vs `origin/main` on `02fa35dd` | P2 same-center ARC reorder; P2 flock reopen after fork |
| Reordered same-center ARCs still L4 | `test_reordered_same_center_arcs_still_l4` |
| Flock fd reopens after PID change | `test_store_file_lock_reopens_fd_after_pid_change` |
| `make test-review-reuse` after ARC angle tie-break + flock PID | **220 passed**, 7 ezdxf warnings |
| Sol wave82 vs `origin/main` on `bcc9cc76` | P2 decided+pipeline_failed; P2 live-recall worker cancel |
| Decision then pipeline boom keeps error | `test_pipeline_failed_after_mid_flight_decision_keeps_error` |
| `make test-review-reuse` after pipeline-fail + worker cancel | **221 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) on `37aac287` | **pass** |
| Sol wave84 vs `origin/main` on `37aac287` | P1 ARC angle only on equal center+radius; P1 mixed cleanup grouping |
| Quantized 0.001 ARC sweep swaps are not L4 | `test_swapped_quantized_step_arc_sweeps_are_not_certified` |
| `--tenant A --apply` keeps hashed A when mixed B holds A | `test_cleanup_tenant_apply_keeps_hashed_when_mixed_other_holds_tasks` |
| `make test-review-reuse` after ARC tie-break + mixed grouping | **223 passed**, 7 ezdxf warnings |
| Sol wave85 vs `origin/main` on `d1c3a93e` | P2 equal-spatial ARC angle; P2 DXF size cap |
| Uniform-offset same-center ARC reorder still L4 | `test_reordered_uniform_offset_same_center_arcs_still_l4` |
| Oversized DXF skips local extract | `test_oversized_dxf_extract_is_skipped` |
| `make test-review-reuse` after equal-spatial ARC + DXF cap | **225 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) on `ea4c612f` | **pass** |
| Sol wave87 vs `origin/main` on `ea4c612f` | P2 two-stage ARC assignment |
| Near-equal 3-decimal ARC sweep swaps are not L4 | `test_near_equal_spatial_arc_sweep_swap_is_not_certified` |
| `make test-review-reuse` after two-stage ARC assignment | **226 passed**, 7 ezdxf warnings |
| Sol wave88 vs `origin/main` on `e5cdc765` | P2 min-total ARC tight edges |
| Colinear equal-total ARC shift still L4 | `test_shifted_colinear_arcs_matching_sweeps_still_l4` |
| `make test-review-reuse` after min-total ARC tight edges | **227 passed**, 7 ezdxf warnings |
| Sol wave89 vs `origin/main` on `ee1d21c6` | P2 JSON size cap; P2 live-recall start stop |
| Oversized JSON skips decode | `test_oversized_json_geom_is_skipped` |
| Nested live-recall start timeout stops worker | `test_run_coro_stops_worker_when_startup_times_out` |
| `make test-review-reuse` after JSON cap + live-recall start stop | **229 passed**, 7 ezdxf warnings |
| Sol window4-w1 vs `origin/main` on `a6f3e2b5` | P2 stale running idempotency replay |
| Stale running idempotency resumes pipeline | `test_stale_running_idempotency_resumes_pipeline` |
| Fresh running idempotency is not resumed | `test_fresh_running_idempotency_is_not_resumed` |
| `make test-review-reuse` after stale running resume | **231 passed**, 7 ezdxf warnings |
| Sol window4-w2 vs `origin/main` on `0edbaf22` | P2 resume input bind; P2 atomic stale claim |
| Mismatched stale retry is 409 | `test_stale_running_idempotency_rejects_mismatched_input` |
| Concurrent stale resume is single-winner | `test_stale_running_idempotency_claim_is_single_winner` |
| `make test-review-reuse` after stale claim/bind | **233 passed**, 7 ezdxf warnings |
| Sol window4-w3 vs `origin/main` on `b611a21f` | P2 INSUNITS 1–24; P2 pipeline claim token |
| Out-of-range INSUNITS is not L4 | `test_out_of_range_insunits_is_not_certified` |
| Stolen claim cannot commit evidence | `test_stale_claim_blocks_previous_owner_commit` |
| `make test-review-reuse` after INSUNITS + claim fence | **235 passed**, 7 ezdxf warnings |
| Sol window4-w4 vs `origin/main` on `bacbb539` | P2 POLYLINE is_closed; P2 raw fractional INSUNITS; P2 FILE idem key |
| Classic closed POLYLINE is not open L4 | `test_dxf_extracted_closed_polyline_is_not_open_l4` |
| Fractional DXF $INSUNITS is not mm | `test_dxf_fractional_insunits_header_is_not_certified` |
| FILE isolated run hashes idempotency key | `test_isolated_file_run_uses_content_idempotency_key` |
| `make test-review-reuse` after extract closed/units + FILE key | **238 passed**, 7 ezdxf warnings |
| Sol window4-w5 vs `origin/main` on `1192d5bf` | P2 CRLF fractional $INSUNITS scan |
| CRLF fractional INSUNITS is not mm | `test_dxf_crlf_fractional_insunits_header_is_not_certified` |
| `make test-review-reuse` after CRLF INSUNITS scan | **239 passed**, 7 ezdxf warnings |
| Sol window4-w6 vs `origin/main` on `22de1b87` | P2 omitted units on both sides still L4 |
| Omitted INSUNITS both sides is not L4 | `test_omitted_insunits_on_both_sides_is_not_certified` |
| `make test-review-reuse` after omitted-units fail-closed | **240 passed**, 7 ezdxf warnings |
| Sol window4-w8 vs `origin/main` on `ccc2f3af` | P2 failed idempotent replay returned HTTP 200 |
| Failed idempotent replay stays 500 | `test_idempotent_replay_of_failed_task_raises_pipeline_failed` |
| `make test-review-reuse` after failed-replay 500 | **241 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) / lint-type / e2e-smoke on `eed168f0` | **pass** |
| Sol window4-w11 vs `origin/main` on `eed168f0` | P2 classic POLYLINE parent default widths dropped |
| Thick classic POLYLINE is not thin LINE L4 | `test_dxf_extracted_polyline_default_width_is_not_certified_as_l4` |
| `make test-review-reuse` after POLYLINE default-width extract | **242 passed**, 7 ezdxf warnings |
| Sol window4-w12 vs `origin/main` on `7f15b214` | P2 nested-loop timeout stopped before canceled cleanup |
| Nested live-recall timeout runs CancelledError cleanup | `test_run_coro_timeout_runs_canceled_cleanup` |
| `make test-review-reuse` after live-recall drain | **243 passed**, 7 ezdxf warnings |
| Sol window4-w13 vs `origin/main` on `9fa69cac` | P2 drain `loop.stop()` blocked the thread-safe future |
| Nested live-recall success does not wait 1s | `test_run_coro_nested_success_does_not_wait_drain_timeout` |
| `make test-review-reuse` after drain-stop-after-future | **244 passed**, 7 ezdxf warnings |
| Sol window4-w14 vs `origin/main` on `b6598ccb` | P2 overflow int coords 500; P2 non-list polyline bulges L4 |
| Overflow JSON ints are not L4 | `test_overflow_integer_coords_are_not_l4_geometry` |
| Non-list bulges are not chord L4 | `test_non_list_polyline_bulges_is_not_certified_as_straight_l4` |
| `make test-review-reuse` after overflow + bulge-shape | **246 passed**, 7 ezdxf warnings |
| Sol window4-w15 vs `origin/main` on `e70f6e9f` | P2 tenant DXF extract_sig cache; P2 overflow adapter scores |
| ReviewReuse DXF extract does not write shared cache | `test_review_reuse_dxf_extract_does_not_write_shared_cache` |
| Overflow adapter scores are not unit scores | `test_overflow_integer_adapter_score_is_not_a_unit_score` |
| `make test-review-reuse` after uncached DXF + score overflow | **248 passed**, 7 ezdxf warnings |
| Sol window4-w16 vs `origin/main` on `be0e60ae` | P2 pre-scored L4 overrode local units/shape refuse |
| Pre-scored L4 does not certify inch vs mm | `test_prescored_l4_does_not_override_units_conflict` |
| `make test-review-reuse` after local-conflict L4 | **249 passed**, 7 ezdxf warnings |
| Sol window4-w17 vs `origin/main` on `c22d8f42` | P2 conflicting INSUNITS aliases; P2 non-boolean has_width |
| Conflicting insunits aliases are not L4 | `test_conflicting_insunits_aliases_are_not_certified` |
| Integer has_width is not thin LINE L4 | `test_integer_has_width_marker_is_not_certified_as_thin_l4` |
| `make test-review-reuse` after alias-units + has_width | **251 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) / lint-type / e2e-smoke on `88b0706d` | **pass** |
| Sol window4-w20 vs `origin/main` on `88b0706d` | P2 non-list polyline `widths` exploded as thin LINE |
| Non-list widths are not thin LINE L4 | `test_non_list_polyline_widths_is_not_certified_as_thin_l4` |
| `make test-review-reuse` after widths-shape | **252 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) / lint-type / e2e-smoke on `1b3afc29` | **pass** |
| Sol window4-w22 vs `origin/main` on `1b3afc29` | P2 put() rescanned every task JSON under flock |
| Valid tenant_meta skips task scan | `test_filesystem_put_skips_task_scan_when_tenant_meta_matches` |
| Missing meta still scans foreign tasks | `test_filesystem_put_scans_tasks_when_tenant_meta_missing` |
| `make test-review-reuse` after store meta fast-path | **254 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) / lint-type / e2e-smoke on `27399d99` | **pass** |
| Sol window4-w23 vs `origin/main` on `27399d99` | **no P1/P2** |
| tests (3.10) / lint-type / e2e-smoke on `5ce7c2df` | **pass** |
| Sol window4-w24 vs `origin/main` on `5ce7c2df` | P2 Windows flock no-op left only RLock |
| Missing fcntl+msvcrt fails closed | `test_store_file_lock_fails_closed_without_fcntl_or_msvcrt` |
| msvcrt path when fcntl missing | `test_store_file_lock_uses_msvcrt_when_fcntl_missing` |
| `make test-review-reuse` after Windows lock | **256 passed**, 7 ezdxf warnings |
| Sol window4-w25 vs `origin/main` on `52565375` | P2 msvcrt retried permanent OSError |
| Permanent msvcrt errors fail closed | `test_store_file_lock_msvcrt_permanent_error_fails_closed` |
| Contention EACCES still retries | `test_store_file_lock_msvcrt_retries_contention` |
| `make test-review-reuse` after msvcrt contention gate | **258 passed**, 7 ezdxf warnings |
| tests (3.10) / tests (3.11) / lint-type / e2e-smoke on `26a727e0` | **pass** |
| Sol window4-w26 vs `origin/main` on `26a727e0` | **no P1/P2** |
| tests (3.10) / lint-type / e2e-smoke on `65b317ba` | **pass** |
| Sol window4-w27 vs `origin/main` on `65b317ba` | **no P1/P2** |

### CI (PR #586)

| Check | Result |
|---|---|
| tests (3.11) on `c01bc02b` | success |
| tests (3.10) on `957546f6` | **success** |
| tests (3.11) on `957546f6` | **success** |
| lint-type on `4d4bb285` | **pass** |
| tests (3.10) / tests (3.11) on `4d4bb285` | **pass** |
| tests (3.10) / tests (3.11) on `ea4c612f` | **pass** |
| e2e-smoke on `2f76c87c` | pass |
| Evaluation Report | fail (Track E; out of scope) |
| tests (3.10) / tests (3.11) / lint-type on `c9a7ce4b` | **pass** |
| tests (3.10) / tests (3.11) / lint-type / core-fast-gate on `dc9267e0` | **pass** (Window 4 start) |
| tests (3.10) / tests (3.11) / lint-type / e2e-smoke on `eed168f0` | **pass** |
| mergeable_state | `unstable` (Evaluation Report Track E; do not merge) |

Re-run `make test-review-reuse` after each follow-up commit and record the count here.

### Window 4 (72h, 2026-09-21 15:03 UTC → 2026-09-24 15:00 UTC)

| Check | Result |
|---|---|
| Window 4 start HEAD | `dc9267e0` |
| tests (3.10) / tests (3.11) / lint-type / core-fast-gate | **pass** on `dc9267e0` |
| Evaluation Report | fail (Track E; out of scope) |
| mergeable_state | do not merge (review required) |

## 3. Honesty probes (must stay true)

```text
# vision-only must not become high confidence
seed: visual=0.99, no geometric, methods=dedup2d-vision
→ rejection_reasons contains vision_only_unverified
→ evidence_pack.confidence.band == low, score == 0.0

# low L4 must not keep duplicate + high visual confidence
seed: geometric=0.1, visual=0.99, methods=precision-l4
→ low_precision_score; state=different; confidence from geometric only (0.1)

# adapter L4 score without match_level
→ verification.level >= 4 (not 0)

# local L4 after upstream vision_only_unverified
→ reason cleared; evidence_pack.confidence uses the geometric score

# mixed legacy a/b + a_b in one sanitized dir
cleanup --tenant a/b --apply → exit 1, mixed dir remains

# mid-flight candidate-less decision, then pipeline finishes
→ status stays decided (not evidence_ready)
→ evidence_pack.candidates matches stored candidates
→ evidence_pack.human_decision.state is the submitted action
→ events include recall_completed, precision_completed, evidence_pack_ready, decision_submitted

# legacy dir named sha256(pilot-tenant)[:24] with tenant_id=that hash
cleanup --tenant pilot-tenant --apply → must not delete that dir

# empty DXF extract (entities=[]) with candidate geom present
→ not precision-l4; missing_geom_json (same as empty JSON)

# two worker processes update_atomically the same running task
→ event count is the sum; no lost terminal overwrite

# JSON {"entities":[{}]} on both query and candidate
→ not precision-l4; missing_geom_json

# legacy task readable, hashed dir occupied by another tenant
cancel → ReviewReuseError store_conflict; legacy file unchanged

# version_gate_filtered + matching geom (geometric 1.0)
→ state stays different; version_gate_filtered remains; not similar

# adapter different with no rejection_reasons + matching geom
→ state stays different; not similar (restore only if low_precision_score existed)
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
| Python 3.10 CI job | **pass** on `65b317ba` (last fully green HEAD) |
| Human review + merge of #586 | owner / reviewer |
| R11 ratify, R12 decision enable | residual_human |
| Track C C1–C5 | residual_human |
| Decisions-before-evidence as a product gate | residual_human (this wave rebuilds the pack instead of blocking) |
