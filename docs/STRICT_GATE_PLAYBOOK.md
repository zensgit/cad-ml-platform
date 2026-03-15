# Strict Gate Playbook

This playbook maps strict-gate failure signals to concrete remediation steps.

## graph2d-review-generic

- Re-run review-pack export and gate locally:
  - `make graph2d-review-pack`
  - `make graph2d-review-pack-gate`
- Inspect gate report:
  - `reports/history_sequence_eval/graph2d_review_pack_gate_report.json`

## graph2d-review-gate-failed-under-strict-mode

- Root signal: review-pack gate status is not `passed` while strict is enabled.
- Actions:
  - Validate review input CSV quality and row counts.
  - Tune thresholds in `config/graph2d_review_pack_gate.yaml` or dispatch overrides.
  - Re-run:
    - `make graph2d-review-pack`
    - `make graph2d-review-pack-gate`

## graph2d-review-strict-mode-disabled

- Root signal: strict mode is disabled for review gate.
- Actions:
  - For release gating, set strict mode to `true`.
  - Re-run strict e2e check:
    - `make graph2d-review-pack-gate-strict-e2e`

## hybrid-blind-generic

- Re-run blind benchmark and gate:
  - `make hybrid-blind-eval`
  - `make hybrid-blind-gate`
- Inspect:
  - `reports/history_sequence_eval/hybrid_blind_gate_report.json`

## hybrid-blind-strict-mode-requires-real-dataset

- Root signal: strict-real mode requested but real dataset path missing/unavailable.
- Actions:
  - Set valid `HYBRID_BLIND_DXF_DIR` and optional manifest.
  - Keep `HYBRID_BLIND_STRICT_REQUIRE_REAL_DATA=true`.
  - Re-run strict-real flow:
    - `make hybrid-blind-strict-real`

## hybrid-blind-gate-failed-under-strict-mode

- Root signal: blind gate status not `passed` under strict mode.
- Actions:
  - Compare hybrid vs Graph2D deltas in gate report.
  - Fix calibration/model/data drift before enabling hard strict.
  - Re-run:
    - `make hybrid-blind-eval`
    - `make hybrid-blind-gate`

## hybrid-calibration-generic

- Re-run calibration and gate:
  - `make hybrid-calibrate-confidence`
  - `make hybrid-calibration-gate`

## hybrid-calibration-gate-failed-under-strict-mode

- Root signal: calibration gate status not `passed` with strict enabled.
- Actions:
  - Check ECE/Brier/MCE against baseline thresholds.
  - Refresh baseline only after confirming intended behavior:
    - `make update-hybrid-calibration-baseline`
  - Re-run gate after baseline/threshold review.

## hybrid-superpass-generic

- Re-run superpass gate:
  - `make hybrid-superpass-gate`
- Validate structural report:
  - `python3 scripts/ci/validate_hybrid_superpass_reports.py --superpass-json ...`

## hybrid-superpass-superpass-failed-under-strict-mode

- Root signal: superpass target status is not `passed` and strict fail is enabled.
- Actions:
  - Verify blind-gate and calibration inputs for superpass checker.
  - Confirm target config in `config/hybrid_superpass_targets.yaml`.
  - Re-run:
    - `make hybrid-superpass-gate`

## hybrid-superpass-validation-nonzero-exit

- Root signal: superpass structure validation step returned nonzero.
- Actions:
  - Inspect validation JSON and headline:
    - `reports/history_sequence_eval/hybrid_superpass_validation.json`
  - Ensure referenced artifacts exist and schema mode is compatible.
  - Re-run validation script directly for pinpoint diagnostics.

## generic-strict-gate

- Unknown strict gate reason.
- Actions:
  - Inspect workflow summary and strict reason fields in PR comment.
  - Search reason source in workflow:
    - `.github/workflows/evaluation-report.yml`
  - Add explicit mapping in:
    - `scripts/ci/comment_evaluation_report_pr.js`
