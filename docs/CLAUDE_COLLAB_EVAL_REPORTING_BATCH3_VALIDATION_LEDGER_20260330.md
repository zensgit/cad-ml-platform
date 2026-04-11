# Claude Collaboration Batch 3 Validation Ledger

日期：2026-03-30

本文件只记录执行事实，不记录设计理由。

填写规则：

- 每批完成后由 Claude 回填
- 后续批次只能追加，不能覆盖上一批内容
- 未完成的字段保留 `TBD`

---

## Batch 3A Status

- `status`: `complete`
- `implementation_scope`: `top-level eval reporting shared helper + interactive report explicit failure semantics`
- `changed_files`:
  - `scripts/ci/generate_eval_reporting_bundle.py`
  - `scripts/generate_eval_report_v2.py`
  - `tests/unit/test_generate_eval_reporting_bundle.py`
- `new_files`:
  - `scripts/eval_reporting_bundle_helpers.py`
  - `tests/unit/test_eval_reporting_bundle_helpers.py`
  - `docs/DEDUP_TOP_LEVEL_EVAL_REPORTING_HELPER_AND_FAILURE_SEMANTICS_ALIGNMENT_DESIGN_20260330.md`
  - `docs/DEDUP_TOP_LEVEL_EVAL_REPORTING_HELPER_AND_FAILURE_SEMANTICS_ALIGNMENT_VALIDATION_20260330.md`
- `tests_run`: `22`
- `test_results`: `22 passed in 14.41s`
- `open_risks`: `none identified`
- `handoff_ready`: `yes`

### Batch 3A Evidence

- `top_level_helper_py`: `scripts/eval_reporting_bundle_helpers.py`
- `top_level_bundle_json`: `reports/eval_history/eval_reporting_bundle.json`
- `static_report_html`: `reports/eval_history/report_static/index.html`
- `interactive_report_html`: `reports/eval_history/report_interactive/index.html`
- `interactive_failure_handling_proof`: `test_interactive_report_failure_causes_bundle_failure — monkeypatches v2 main to return 1, asserts top-level rc != 0 and manifest not written`
- `make_target_proof`: `make -n eval-reporting-bundle` outputs expected flags (unchanged from Batch 2)
- `design_md`: `docs/DEDUP_TOP_LEVEL_EVAL_REPORTING_HELPER_AND_FAILURE_SEMANTICS_ALIGNMENT_DESIGN_20260330.md`
- `validation_md`: `docs/DEDUP_TOP_LEVEL_EVAL_REPORTING_HELPER_AND_FAILURE_SEMANTICS_ALIGNMENT_VALIDATION_20260330.md`

### Batch 3A Command Log

```text
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  scripts/eval_reporting_bundle_helpers.py \
  scripts/ci/generate_eval_reporting_bundle.py \
  scripts/generate_eval_report.py \
  scripts/generate_eval_report_v2.py \
  tests/unit/test_eval_reporting_bundle_helpers.py \
  tests/unit/test_generate_eval_reporting_bundle.py \
  tests/unit/test_generate_eval_report.py \
  tests/unit/test_generate_eval_report_v2.py \
  tests/unit/test_eval_history_make_targets.py

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_eval_reporting_bundle_helpers.py \
  tests/unit/test_generate_eval_reporting_bundle.py \
  tests/unit/test_generate_eval_report.py \
  tests/unit/test_generate_eval_report_v2.py \
  tests/unit/test_eval_history_make_targets.py -q
```

### Batch 3A Result Log

```text
py_compile: success (no output, all 9 files compile cleanly)

pytest: 22 passed in 14.41s
```

---

## Batch 3B Status

- `status`: `complete`
- `implementation_scope`: `default materialization chain integration + persisted artifact pointer + schema/validator alignment`
- `changed_files`:
  - `scripts/eval_with_history.sh`
  - `scripts/validate_eval_history.py`
  - `docs/eval_history.schema.json`
  - `tests/unit/test_eval_with_history_script_history_sequence.py`
  - `tests/unit/test_validate_eval_history_history_sequence.py`
- `new_files`:
  - `docs/DEDUP_TOP_LEVEL_EVAL_REPORTING_DEFAULT_MATERIALIZATION_AND_ARTIFACT_POINTER_ALIGNMENT_DESIGN_20260330.md`
  - `docs/DEDUP_TOP_LEVEL_EVAL_REPORTING_DEFAULT_MATERIALIZATION_AND_ARTIFACT_POINTER_ALIGNMENT_VALIDATION_20260330.md`
- `tests_run`: `18 (batch 3B) + 37 (full regression)`
- `test_results`: `18 passed in 20.93s (batch 3B), 37 passed in 15.68s (full regression)`
- `open_risks`: `none identified`
- `handoff_ready`: `yes`

### Batch 3B Evidence

- `default_materialization_entrypoint`: `scripts/eval_with_history.sh` — materializes `eval_reporting_bundle` after `history_sequence_reporting_bundle`, before explainability guard
- `eval_reporting_bundle_json`: `reports/eval_history/eval_reporting_bundle.json`
- `eval_reporting_bundle_md`: `reports/eval_history/eval_reporting_bundle.md`
- `persisted_artifact_pointer_proof`: `history_report["artifacts"]["eval_reporting_bundle_json"]` verified in test to equal `<REPORT_DIR>/eval_reporting_bundle.json`
- `validator_schema_alignment_proof`: `test_validate_eval_history_accepts_eval_reporting_bundle_artifact` passes — validator and schema both accept `eval_reporting_bundle_json` as `string|null`
- `make_target_proof`: `make -n eval-reporting-bundle` outputs expected flags (unchanged)
- `design_md`: `docs/DEDUP_TOP_LEVEL_EVAL_REPORTING_DEFAULT_MATERIALIZATION_AND_ARTIFACT_POINTER_ALIGNMENT_DESIGN_20260330.md`
- `validation_md`: `docs/DEDUP_TOP_LEVEL_EVAL_REPORTING_DEFAULT_MATERIALIZATION_AND_ARTIFACT_POINTER_ALIGNMENT_VALIDATION_20260330.md`

### Batch 3B Command Log

```text
bash -n scripts/eval_with_history.sh

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  scripts/validate_eval_history.py \
  tests/unit/test_eval_with_history_script_history_sequence.py \
  tests/unit/test_validate_eval_history_history_sequence.py \
  tests/unit/test_eval_history_make_targets.py \
  tests/unit/test_generate_eval_reporting_bundle.py

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_eval_with_history_script_history_sequence.py \
  tests/unit/test_validate_eval_history_history_sequence.py \
  tests/unit/test_eval_history_make_targets.py \
  tests/unit/test_generate_eval_reporting_bundle.py -q

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_eval_reporting_bundle_helpers.py \
  tests/unit/test_generate_eval_reporting_bundle.py \
  tests/unit/test_generate_eval_report.py \
  tests/unit/test_generate_eval_report_v2.py \
  tests/unit/test_eval_with_history_script_history_sequence.py \
  tests/unit/test_validate_eval_history_history_sequence.py \
  tests/unit/test_eval_history_make_targets.py \
  tests/unit/test_generate_history_sequence_reporting_bundle.py \
  tests/unit/test_generate_eval_signal_reporting_bundle.py -q
```

### Batch 3B Result Log

```text
bash -n: success (no output)

py_compile: success (no output, all 5 files compile cleanly)

pytest batch 3B: 18 passed in 20.93s

pytest full regression: 37 passed in 15.68s
```

---

## Verifier Notes

这一节由人工验证者填写。

### Batch 3A Verifier Decision

- `decision`: `accepted`
- `notes`: `人工复核通过。实际复跑 py_compile 通过，pytest 为 22 passed in 7.55s。顶层 helper 仅承担 bundle/sub-bundle discovery 和 discovery context 归一化，没有越权成为新的 metrics owner。generate_eval_report_v2.main() 已显式 return 0，generate_eval_reporting_bundle.py 也已对 interactive report 做 rc 检查并 fail-closed。非阻塞说明：helper 目前主要为 Batch 3B 和后续顶层 discovery 消费做准备，当前尚未强制接入更多 report consumer，这不影响 3A 验收。`

### Batch 3B Verifier Decision

- `decision`: `accepted`
- `notes`: `人工复核通过。实际复跑 bash -n 通过，Batch 3B pytest 为 18 passed in 18.93s，合并回归为 37 passed in 16.15s。eval_with_history.sh 已按要求在 history_sequence_reporting_bundle 之后、named-command explainability guard 之前 materialize eval_reporting_bundle，失败时以 exit 5 fail-closed；persisted artifacts 已新增 eval_reporting_bundle_json 指针；validator 与 schema 已同步接受该字段。非阻塞说明：artifact 指针是在 materialization 之前按默认路径预写入 row，这是本批目标内可接受行为，因为脚本失败时会 fail-closed，成功路径下目标文件也会随即被 materialize。`
