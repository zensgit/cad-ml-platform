from __future__ import annotations

from src.core.knowledge.dynamic.manager import KnowledgeManager
from src.core.knowledge.dynamic.models import GeometryPattern, KnowledgeCategory, KnowledgeEntry
from src.core.knowledge.dynamic.store import JSONKnowledgeStore


def _build_manager(tmp_path, rules):
    KnowledgeManager._instance = None
    store = JSONKnowledgeStore(data_dir=tmp_path)
    for rule in rules:
        store.save(rule)
    return KnowledgeManager(store=store)


def test_keyword_and_pattern_hints(tmp_path):
    rule = KnowledgeEntry(
        category=KnowledgeCategory.PART_TYPE,
        name="Bracket rule",
        keywords=["bracket"],
        ocr_patterns=[r"BRK-\d+"],
        part_hints={"bracket": 0.6},
    )
    km = _build_manager(tmp_path, [rule])
    hints = km.get_part_hints(text="BRK-123 bracket", geometric_features=None, entity_counts=None)
    assert hints["bracket"] == 1.0


def test_geometry_pattern_ratio_match(tmp_path):
    rule = GeometryPattern(
        category=KnowledgeCategory.GEOMETRY,
        name="Washer ratio rule",
        conditions={"circle_ratio": {"min": 0.5}},
        part_hints={"washer": 0.7},
    )
    km = _build_manager(tmp_path, [rule])
    hints = km.get_part_hints(
        text="",
        geometric_features={},
        entity_counts={"CIRCLE": 6, "LINE": 4},
    )
    assert hints["washer"] == 0.7


def test_geometry_pattern_empty_conditions_skipped(tmp_path):
    rule = GeometryPattern(
        category=KnowledgeCategory.GEOMETRY,
        name="Empty conditions rule",
        conditions={},
        part_hints={"bolt": 0.5},
    )
    km = _build_manager(tmp_path, [rule])
    hints = km.get_part_hints(text="", geometric_features={}, entity_counts={"CIRCLE": 1})
    assert "bolt" not in hints
