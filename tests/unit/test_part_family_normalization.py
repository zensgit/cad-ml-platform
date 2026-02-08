from src.core.classification.part_family import normalize_part_family_prediction


def test_normalize_part_family_prediction_non_dict_sets_invalid_payload_error():
    out = normalize_part_family_prediction("nope", provider_name="v16")
    assert out["part_family"] is None
    assert out["part_family_confidence"] is None
    assert out["part_family_source"] == "provider:v16"
    assert (out.get("part_family_error") or {}).get("code") == "invalid_payload"


def test_normalize_part_family_prediction_non_ok_status_sets_error_code_and_message():
    out = normalize_part_family_prediction(
        {"status": "timeout", "error": "timeout after 0.01s"},
        provider_name="v16",
    )
    assert out["part_family"] is None
    assert (out.get("part_family_error") or {}).get("code") == "timeout"
    assert (out.get("part_family_error") or {}).get("message") == "timeout after 0.01s"


def test_normalize_part_family_prediction_ok_missing_label_sets_missing_label():
    out = normalize_part_family_prediction({"status": "ok", "confidence": 0.9}, provider_name="v16")
    assert out["part_family"] is None
    assert (out.get("part_family_error") or {}).get("code") == "missing_label"


def test_normalize_part_family_prediction_ok_populates_fields_and_clamps_confidence():
    out = normalize_part_family_prediction(
        {
            "status": "ok",
            "label": "壳体类",
            "confidence": 1.5,
            "model_version": 123,
            "needs_review": True,
            "review_reason": "edge_case",
            "top2_category": "轴类",
            "top2_confidence": -0.2,
        },
        provider_name=" v16 ",
    )
    assert out["part_family"] == "壳体类"
    assert out["part_family_confidence"] == 1.0
    assert out["part_family_source"] == "provider:v16"
    assert out["part_family_model_version"] == "123"
    assert out["part_family_needs_review"] is True
    assert out["part_family_review_reason"] == "edge_case"
    assert (out.get("part_family_top2") or {}).get("label") == "轴类"
    assert (out.get("part_family_top2") or {}).get("confidence") == 0.0


def test_normalize_part_family_prediction_top2_requires_confidence_value():
    out = normalize_part_family_prediction(
        {"status": "ok", "label": "连接件", "confidence": 0.5, "top2_category": "其他"},
        provider_name="v6",
    )
    assert out["part_family"] == "连接件"
    assert out["part_family_top2"] is None

