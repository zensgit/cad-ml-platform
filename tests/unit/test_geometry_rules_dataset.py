from src.core.knowledge.dynamic.manager import get_knowledge_manager


def test_dataset_geometry_rules_keyword_match():
    km = get_knowledge_manager()
    hints = km.get_part_hints(text="上封头组件", geometric_features=None, entity_counts=None)
    assert "上封头组件" in hints


def test_dataset_geometry_rules_variant_match():
    km = get_knowledge_manager()
    hints = km.get_part_hints(text="过滤托架", geometric_features=None, entity_counts=None)
    assert "过滤托架" in hints
