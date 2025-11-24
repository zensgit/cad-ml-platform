"""Test data fixtures for v4 feature extraction.

Provides various CAD document test cases for v4 feature validation.
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class MockEntity:
    """Mock CAD entity for testing."""
    type: str
    id: str
    properties: Dict[str, Any]


@dataclass
class MockCadDocument:
    """Mock CAD document for testing."""
    entities: List[MockEntity]
    format: str = "DXF"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def create_empty_document() -> MockCadDocument:
    """空文档 - 无实体."""
    return MockCadDocument(entities=[], format="DXF")


def create_single_cube_document() -> MockCadDocument:
    """单立方体 - 6个面."""
    return MockCadDocument(
        entities=[
            MockEntity(type="SOLID", id="cube1", properties={"faces": 6}),
        ],
        format="DXF",
    )


def create_simple_document() -> MockCadDocument:
    """简单文档 - 少量实体."""
    return MockCadDocument(
        entities=[
            MockEntity(type="LINE", id="l1", properties={}),
            MockEntity(type="CIRCLE", id="c1", properties={}),
            MockEntity(type="ARC", id="a1", properties={}),
        ],
        format="DXF",
    )


def create_complex_document() -> MockCadDocument:
    """复杂文档 - 多种实体类型，高多样性."""
    entities = []

    # 各种类型实体
    types = ["LINE", "CIRCLE", "ARC", "POLYLINE", "SOLID", "SPLINE", "ELLIPSE", "HATCH"]
    for i, entity_type in enumerate(types):
        for j in range(3):  # 每种类型3个实体
            entities.append(
                MockEntity(
                    type=entity_type,
                    id=f"{entity_type.lower()}_{i}_{j}",
                    properties={}
                )
            )

    return MockCadDocument(entities=entities, format="DWG")


def create_uniform_document() -> MockCadDocument:
    """均匀分布文档 - 所有类型数量相同."""
    entities = []
    types = ["LINE", "CIRCLE", "ARC", "POLYLINE"]

    for entity_type in types:
        for i in range(5):
            entities.append(
                MockEntity(
                    type=entity_type,
                    id=f"{entity_type.lower()}_{i}",
                    properties={}
                )
            )

    return MockCadDocument(entities=entities, format="DXF")


def create_single_type_document() -> MockCadDocument:
    """单一类型文档 - 熵值应为0."""
    entities = []
    for i in range(10):
        entities.append(
            MockEntity(type="LINE", id=f"line_{i}", properties={})
        )

    return MockCadDocument(entities=entities, format="DXF")


def create_document_with_solids() -> MockCadDocument:
    """带实体几何的文档 - 用于surface_count测试."""
    return MockCadDocument(
        entities=[
            MockEntity(type="SOLID", id="box", properties={"faces": 6}),
            MockEntity(type="SOLID", id="cylinder", properties={"faces": 3}),
            MockEntity(type="SOLID", id="sphere", properties={"faces": 1}),
        ],
        format="STEP",
    )


# Test data registry
V4_TEST_CASES = {
    "empty": {
        "doc": create_empty_document(),
        "expected_entropy": 0.0,
        "expected_surface_count": 0,
        "description": "空文档，无实体",
    },
    "single_cube": {
        "doc": create_single_cube_document(),
        "expected_entropy": 0.0,  # 单一类型
        "expected_surface_count": 6,  # 立方体6个面
        "description": "单个立方体",
    },
    "simple": {
        "doc": create_simple_document(),
        "expected_entropy_range": (0.5, 1.0),  # 多种类型
        "expected_surface_count": 3,  # 简单估算
        "description": "简单文档，3个实体",
    },
    "complex": {
        "doc": create_complex_document(),
        "expected_entropy_range": (0.8, 1.0),  # 高多样性
        "expected_surface_count": 24,  # 8类型 * 3实体
        "description": "复杂文档，24个实体，8种类型",
    },
    "uniform": {
        "doc": create_uniform_document(),
        "expected_entropy": 1.0,  # 完全均匀分布
        "expected_surface_count": 20,  # 4类型 * 5实体
        "description": "均匀分布文档",
    },
    "single_type": {
        "doc": create_single_type_document(),
        "expected_entropy": 0.0,  # 无多样性
        "expected_surface_count": 10,
        "description": "单一类型文档，10个LINE",
    },
    "with_solids": {
        "doc": create_document_with_solids(),
        "expected_entropy": 0.0,  # 单一SOLID类型
        "expected_surface_count": 10,  # 6 + 3 + 1
        "description": "包含实体几何的文档",
    },
}


def get_test_case(name: str) -> Dict[str, Any]:
    """获取测试用例."""
    if name not in V4_TEST_CASES:
        raise ValueError(f"Unknown test case: {name}. Available: {list(V4_TEST_CASES.keys())}")
    return V4_TEST_CASES[name]


def get_all_test_cases() -> List[Dict[str, Any]]:
    """获取所有测试用例."""
    return list(V4_TEST_CASES.values())


__all__ = [
    "MockEntity",
    "MockCadDocument",
    "create_empty_document",
    "create_single_cube_document",
    "create_simple_document",
    "create_complex_document",
    "create_uniform_document",
    "create_single_type_document",
    "create_document_with_solids",
    "V4_TEST_CASES",
    "get_test_case",
    "get_all_test_cases",
]
