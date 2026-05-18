from src.core.knowledge.analysis_summary import (
    KNOWLEDGE_RULE_VERSION,
    build_knowledge_summary,
)


def _checks_for(payload: dict, category: str) -> list[dict]:
    return [
        item for item in payload["knowledge_checks"] if item.get("category") == category
    ]


def _assert_rule_metadata(rows: list[dict]) -> None:
    assert rows
    assert all(item.get("rule_source") for item in rows)
    assert all(item.get("rule_version") == KNOWLEDGE_RULE_VERSION for item in rows)


def test_build_knowledge_summary_extracts_checks_and_conflicts() -> None:
    payload = build_knowledge_summary(
        text_signals="M10x1.25 ISO 2768-mK GB/T 1804-M IT7 位置度 0.2 M A C B "
        "材料: 304 表面粗糙度 Ra3.2 N8 人孔",
        geometric_features={"complexity_score": 12.0},
        entity_counts={"CIRCLE": 2, "LINE": 4},
        fine_part_type="人孔",
        coarse_part_type="开孔件",
    )

    categories = {item["category"] for item in payload["knowledge_checks"]}
    assert "thread_standard" in categories
    assert "general_tolerance" in categories
    assert "it_grade" in categories
    assert "gdt" in categories
    assert "material" in categories
    assert "surface_finish" in categories
    checks_by_category = {
        item["category"]: item for item in payload["knowledge_checks"]
    }
    assert checks_by_category["material"]["rule_source"] == "materials_catalog"
    assert checks_by_category["surface_finish"]["rule_source"] == (
        "iso1302_surface_finish_catalog"
    )
    assert all(
        item.get("rule_version") == KNOWLEDGE_RULE_VERSION
        for item in payload["knowledge_checks"]
    )

    standards = payload["standards_candidates"]
    assert any(item.get("designation") == "M10x1.25" for item in standards)
    assert any(item.get("designation") == "ISO 2768-MK" for item in standards)
    assert any(item.get("designation") == "GB/T 1804-M" for item in standards)
    assert any(
        item.get("type") == "material" and item.get("grade") == "S30408"
        for item in standards
    )
    assert any(
        item.get("type") == "surface_finish" and item.get("grade") == "N8"
        for item in standards
    )
    assert all(item.get("rule_source") for item in standards)
    assert all(item.get("rule_version") == KNOWLEDGE_RULE_VERSION for item in standards)

    violations = payload["violations"]
    assert violations
    assert violations[0]["category"] == "datum_sequence"
    assert all(item.get("rule_source") for item in violations)
    assert all(
        item.get("rule_version") == KNOWLEDGE_RULE_VERSION for item in violations
    )
    assert payload["knowledge_hints"]
    assert all(
        item.get("rule_source") == "knowledge_manager"
        for item in payload["knowledge_hints"]
    )
    assert all(
        item.get("rule_version") == KNOWLEDGE_RULE_VERSION
        for item in payload["knowledge_hints"]
    )


def test_knowledge_summary_fixture_material_substitution() -> None:
    payload = build_knowledge_summary(
        text_signals="材料替代: 304 -> 316L，用于耐蚀升级",
    )

    checks = _checks_for(payload, "material_substitution")
    _assert_rule_metadata(checks)
    substitution = checks[0]["value"]
    assert substitution["source_material"]["grade"] == "S30408"
    assert substitution["target_material"]["grade"] == "S31603"
    assert substitution["target_material"]["equivalents"]["US"] == "316L"


def test_knowledge_summary_fixture_h7_g6_fit_validation() -> None:
    payload = build_knowledge_summary(
        text_signals="公差配合: 直径25 H7/g6 精密滑动配合",
    )

    checks = _checks_for(payload, "fit_validation")
    _assert_rule_metadata(checks)
    assert checks[0]["item"] == "H7/g6"
    assert checks[0]["value"]["fit_type"] == "clearance"
    assert checks[0]["value"]["fit_class"] == "normal_running"
    assert any(
        item.get("type") == "iso_fit" and item.get("designation") == "H7/g6"
        for item in payload["standards_candidates"]
    )


def test_knowledge_summary_fixture_surface_finish_recommendation() -> None:
    payload = build_knowledge_summary(
        text_signals="密封面表面粗糙度建议 Ra1.6，关键接触面要求 N7",
    )

    checks = _checks_for(payload, "surface_finish_recommendation")
    _assert_rule_metadata(checks)
    assert checks[0]["value"]["recommended_finish"]["designation"] == "Ra 1.6"
    assert checks[0]["value"]["candidate_finishes"]


def test_knowledge_summary_fixture_machining_process_route() -> None:
    payload = build_knowledge_summary(
        text_signals="加工路线: 车削 -> 铣削 -> 钻孔 -> 磨削",
    )

    checks = _checks_for(payload, "machining_process_route")
    _assert_rule_metadata(checks)
    assert checks[0]["value"]["processes"] == [
        "turning",
        "milling",
        "drilling",
        "grinding",
    ]


def test_knowledge_summary_fixture_manufacturability_risk() -> None:
    payload = build_knowledge_summary(
        text_signals="薄壁区域需复核，存在深孔加工风险",
        geometric_features={
            "thin_walls_detected": True,
            "min_thickness_estimate": 0.4,
            "stock_removal_ratio": 0.9,
        },
    )

    checks = _checks_for(payload, "manufacturability_risk")
    _assert_rule_metadata(checks)
    risk_codes = {item["code"] for item in checks[0]["value"]["risks"]}
    assert {"THIN_WALL", "DEEP_HOLE", "HIGH_STOCK_REMOVAL"}.issubset(risk_codes)

    violations = [
        item
        for item in payload["violations"]
        if item.get("category") == "manufacturability_risk"
    ]
    _assert_rule_metadata(violations)
