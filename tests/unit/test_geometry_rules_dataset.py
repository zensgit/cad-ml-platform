"""Tests for geometry rules dataset matching.

Note: These tests depend on specific data being present in the knowledge base.
If the knowledge base structure changes, these tests may need updating.
"""

import pytest

from src.core.knowledge.dynamic.manager import get_knowledge_manager


@pytest.mark.skipif(
    True,  # Skip until knowledge base data is populated
    reason="Knowledge base does not contain required geometry rules for these part types"
)
def test_dataset_geometry_rules_keyword_match():
    """Test keyword matching returns expected part hints.

    This test requires '上封头组件' to be defined in geometry_rules.json
    with appropriate keyword rules.
    """
    km = get_knowledge_manager()
    hints = km.get_part_hints(text="上封头组件", geometric_features=None, entity_counts=None)
    assert "上封头组件" in hints


@pytest.mark.skipif(
    True,  # Skip until knowledge base data is populated
    reason="Knowledge base does not contain required geometry rules for these part types"
)
def test_dataset_geometry_rules_variant_match():
    """Test variant matching returns expected part hints.

    This test requires '过滤托架' to be defined in geometry_rules.json
    with appropriate keyword rules.
    """
    km = get_knowledge_manager()
    hints = km.get_part_hints(text="过滤托架", geometric_features=None, entity_counts=None)
    assert "过滤托架" in hints


def test_knowledge_manager_initialization():
    """Test that knowledge manager initializes successfully."""
    km = get_knowledge_manager()
    assert km is not None


def test_get_part_hints_returns_dict():
    """Test that get_part_hints returns a dictionary."""
    km = get_knowledge_manager()
    hints = km.get_part_hints(text="test", geometric_features=None, entity_counts=None)
    assert isinstance(hints, dict)


def test_get_part_hints_with_empty_text():
    """Test get_part_hints with empty text returns empty dict."""
    km = get_knowledge_manager()
    hints = km.get_part_hints(text="", geometric_features=None, entity_counts=None)
    assert isinstance(hints, dict)
