from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sample_ok_payload() -> dict:
    return {
        "status": "ok",
        "recommended_thresholds": {
            "min_reports": 4,
            "max_hybrid_accuracy_drop": 0.03,
            "max_gain_drop": 0.03,
            "max_coverage_drop": 0.05,
            "label_slice_enable": True,
            "label_slice_min_common": 2,
            "label_slice_auto_cap_min_common": True,
            "label_slice_min_support": 3,
            "label_slice_max_hybrid_accuracy_drop": 0.15,
            "label_slice_max_gain_drop": 0.15,
            "family_slice_enable": True,
            "family_slice_min_common": 2,
            "family_slice_auto_cap_min_common": True,
            "family_slice_min_support": 5,
            "family_slice_max_hybrid_accuracy_drop": 0.20,
            "family_slice_max_gain_drop": 0.20,
        },
    }


def test_apply_hybrid_blind_drift_suggestion_builds_var_map() -> None:
    from scripts.ci import apply_hybrid_blind_drift_suggestion_to_gh_vars as mod

    var_map = mod._build_var_map(_sample_ok_payload())
    assert var_map["HYBRID_BLIND_DRIFT_ALERT_MIN_REPORTS"] == "4"
    assert var_map["HYBRID_BLIND_DRIFT_ALERT_MAX_ACC_DROP"] == "0.03"
    assert var_map["HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_AUTO_CAP_MIN_COMMON"] == "true"
    assert var_map["HYBRID_BLIND_DRIFT_ALERT_FAMILY_SLICE_MIN_SUPPORT"] == "5"


def test_apply_hybrid_blind_drift_suggestion_main_print_only(tmp_path: Path) -> None:
    from scripts.ci import apply_hybrid_blind_drift_suggestion_to_gh_vars as mod

    suggestion = tmp_path / "suggestion.json"
    _write_json(suggestion, _sample_ok_payload())

    rc = mod.main(
        [
            "--suggestion-json",
            str(suggestion),
            "--repo",
            "zensgit/cad-ml-platform",
        ]
    )
    assert rc == 0


def test_apply_hybrid_blind_drift_suggestion_requires_ok_status(tmp_path: Path) -> None:
    from scripts.ci import apply_hybrid_blind_drift_suggestion_to_gh_vars as mod

    suggestion = tmp_path / "suggestion_bad.json"
    _write_json(suggestion, {"status": "insufficient"})

    try:
        mod.main(
            [
                "--suggestion-json",
                str(suggestion),
                "--repo",
                "zensgit/cad-ml-platform",
            ]
        )
    except ValueError as exc:
        assert "status must be 'ok'" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-ok status")
