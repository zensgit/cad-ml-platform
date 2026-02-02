"""
Tests for CAD modules C1-C2.

Covers:
- C1: DWG native support
- C2: DXF enhanced parsing
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# C1: DWG Native Support Tests
# ============================================================================

class TestC1DWGSupport:
    """Tests for C1 DWG support module."""

    def test_dwg_imports(self):
        """Test DWG module imports."""
        from src.core.cad.dwg import (
            DWGConverter,
            ConverterConfig,
            ConversionResult,
            DWGParser,
            DWGHeader,
            DWGManager,
        )
        assert DWGConverter is not None
        assert DWGParser is not None

    def test_converter_config_creation(self):
        """Test ConverterConfig creation."""
        from src.core.cad.dwg.converter import ConverterConfig, DXFVersion

        config = ConverterConfig(
            output_version=DXFVersion.R2018,
            output_format="DXF",
            audit=True,
            timeout=60,
        )
        assert config.output_version == DXFVersion.R2018
        assert config.timeout == 60

    def test_dxf_version_enum(self):
        """Test DXFVersion enum."""
        from src.core.cad.dwg.converter import DXFVersion

        assert DXFVersion.R2018.value == "ACAD2018"
        assert DXFVersion.R2000.value == "ACAD2000"

    def test_conversion_result_creation(self):
        """Test ConversionResult creation."""
        from src.core.cad.dwg.converter import ConversionResult, ConversionStatus

        result = ConversionResult(
            input_path="/path/to/file.dwg",
            output_path="/path/to/file.dxf",
            status=ConversionStatus.SUCCESS,
            conversion_time=1.5,
        )
        assert result.success is True
        assert result.conversion_time == 1.5

    def test_dwg_converter_creation(self):
        """Test DWGConverter creation."""
        from src.core.cad.dwg.converter import DWGConverter, ConverterConfig

        config = ConverterConfig()
        converter = DWGConverter(config)
        assert converter is not None

    def test_dwg_version_enum(self):
        """Test DWGVersion enum."""
        from src.core.cad.dwg.parser import DWGVersion

        assert DWGVersion.R2018.value == "AC1032"
        assert DWGVersion.R2000.value == "AC1015"

    def test_dwg_header_creation(self):
        """Test DWGHeader creation."""
        from src.core.cad.dwg.parser import DWGHeader, DWGVersion

        header = DWGHeader(
            version=DWGVersion.R2018,
            version_string="AC1032",
            file_size=1024,
        )
        assert header.version == DWGVersion.R2018

    def test_dwg_parser_creation(self):
        """Test DWGParser creation."""
        from src.core.cad.dwg.parser import DWGParser

        parser = DWGParser()
        assert parser is not None

    def test_dwg_manager_config(self):
        """Test ManagerConfig creation."""
        from src.core.cad.dwg.manager import ManagerConfig

        config = ManagerConfig(
            cache_dir="/tmp/dwg_cache",
            cache_ttl_hours=24,
            auto_convert=True,
        )
        assert config.cache_ttl_hours == 24
        assert config.auto_convert is True

    def test_dwg_file_creation(self):
        """Test DWGFile creation."""
        from src.core.cad.dwg.manager import DWGFile
        from pathlib import Path

        dwg_file = DWGFile(
            path=Path("/path/to/file.dwg"),
        )
        assert dwg_file.name == "file.dwg"
        assert dwg_file.stem == "file"


# ============================================================================
# C2: DXF Enhanced Parsing Tests
# ============================================================================

class TestC2DXFEnhanced:
    """Tests for C2 DXF enhanced parsing module."""

    def test_dxf_enhanced_imports(self):
        """Test DXF enhanced module imports."""
        from src.core.cad.dxf import (
            LayerHierarchy,
            LayerNode,
            BlockExpander,
            AttributeExtractor,
            EntityFilter,
        )
        assert LayerHierarchy is not None
        assert BlockExpander is not None

    # ---------- Layer Hierarchy Tests ----------

    def test_layer_node_creation(self):
        """Test LayerNode creation."""
        from src.core.cad.dxf.hierarchy import LayerNode

        node = LayerNode(
            name="GEOMETRY",
            full_name="LAYER-GEOMETRY",
            color=1,
            is_on=True,
        )
        assert node.name == "GEOMETRY"
        assert node.is_leaf is True

    def test_layer_hierarchy_creation(self):
        """Test LayerHierarchy creation."""
        from src.core.cad.dxf.hierarchy import LayerHierarchy

        hierarchy = LayerHierarchy()
        assert hierarchy is not None
        assert hierarchy.root is not None

    def test_layer_hierarchy_add_layers(self):
        """Test adding layers to hierarchy."""
        from src.core.cad.dxf.hierarchy import LayerHierarchy

        hierarchy = LayerHierarchy()
        node1 = hierarchy.add_layer("MAIN-GEOMETRY")
        node2 = hierarchy.add_layer("MAIN-TEXT")
        node3 = hierarchy.add_layer("DETAIL-DIMS")

        assert len(hierarchy.layers) == 3
        assert hierarchy.separator == "-"

    def test_layer_hierarchy_tree(self):
        """Test layer hierarchy tree structure."""
        from src.core.cad.dxf.hierarchy import LayerHierarchy

        hierarchy = LayerHierarchy()
        hierarchy.add_layer("A-B-C")
        hierarchy.add_layer("A-B-D")
        hierarchy.add_layer("A-E")

        # Check tree structure
        a_node = hierarchy.root.find_child("A")
        assert a_node is not None
        assert len(a_node.children) == 2  # B and E

    def test_layer_analyzer_creation(self):
        """Test LayerAnalyzer creation."""
        from src.core.cad.dxf.hierarchy import LayerAnalyzer

        analyzer = LayerAnalyzer()
        assert analyzer is not None

    # ---------- Block Expansion Tests ----------

    def test_block_reference_creation(self):
        """Test BlockReference creation."""
        from src.core.cad.dxf.blocks import BlockReference

        ref = BlockReference(
            name="TEST_BLOCK",
            handle="ABC123",
            layer="0",
            insert_point=(100.0, 200.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            rotation=45.0,
        )
        assert ref.name == "TEST_BLOCK"
        assert ref.rotation == 45.0
        assert ref.uniform_scale is True

    def test_block_reference_transform_matrix(self):
        """Test transform matrix generation."""
        from src.core.cad.dxf.blocks import BlockReference
        import math

        ref = BlockReference(
            name="TEST",
            handle="ABC",
            layer="0",
            insert_point=(10.0, 20.0, 0.0),
            scale=(2.0, 2.0, 1.0),
            rotation=90.0,
        )
        matrix = ref.get_transform_matrix()
        assert len(matrix) == 4
        assert len(matrix[0]) == 4

    def test_block_expander_creation(self):
        """Test BlockExpander creation."""
        from src.core.cad.dxf.blocks import BlockExpander

        expander = BlockExpander(max_depth=10)
        assert expander is not None

    def test_expanded_block_creation(self):
        """Test ExpandedBlock creation."""
        from src.core.cad.dxf.blocks import ExpandedBlock

        expanded = ExpandedBlock(name="TEST_BLOCK")
        assert expanded.name == "TEST_BLOCK"
        assert expanded.total_entity_count == 0

    # ---------- Attribute Extraction Tests ----------

    def test_attribute_definition_creation(self):
        """Test AttributeDefinition creation."""
        from src.core.cad.dxf.attributes import AttributeDefinition

        attdef = AttributeDefinition(
            tag="PART_NO",
            prompt="Enter part number:",
            default_value="",
            insert_point=(0.0, 0.0, 0.0),
            height=2.5,
            rotation=0.0,
            layer="0",
        )
        assert attdef.tag == "PART_NO"

    def test_attrib_data_creation(self):
        """Test ATTRIBData creation."""
        from src.core.cad.dxf.attributes import ATTRIBData

        attrib = ATTRIBData(
            tag="PART_NO",
            value="12345",
            insert_point=(0.0, 0.0, 0.0),
            height=2.5,
            rotation=0.0,
            layer="0",
            style="Standard",
            block_name="TITLEBLOCK",
            block_handle="ABC",
        )
        assert attrib.tag == "PART_NO"
        assert attrib.value == "12345"

    def test_attribute_extractor_creation(self):
        """Test AttributeExtractor creation."""
        from src.core.cad.dxf.attributes import AttributeExtractor

        extractor = AttributeExtractor()
        assert extractor is not None

    def test_titleblock_patterns(self):
        """Test titleblock field pattern matching."""
        from src.core.cad.dxf.attributes import AttributeExtractor
        import re

        extractor = AttributeExtractor()

        # Test various tag formats
        test_cases = [
            ("DWG_NO", "drawing_number"),
            ("TITLE", "drawing_title"),
            ("REV", "revision"),
            ("DATE", "date"),
            ("DRAWN_BY", "drawn_by"),
            ("SCALE", "scale"),
        ]

        for tag, expected_field in test_cases:
            matched_field = None
            for field, patterns in extractor.TITLEBLOCK_PATTERNS.items():
                for pattern in patterns:
                    if re.match(pattern, tag):
                        matched_field = field
                        break
                if matched_field:
                    break
            assert matched_field == expected_field, f"Tag {tag} should match {expected_field}"

    # ---------- Entity Filter Tests ----------

    def test_filter_config_creation(self):
        """Test FilterConfig creation."""
        from src.core.cad.dxf.filters import FilterConfig

        config = FilterConfig(
            layers=["GEOMETRY", "TEXT"],
            entity_types=["LINE", "CIRCLE"],
            min_x=0,
            max_x=1000,
        )
        assert config.layers == ["GEOMETRY", "TEXT"]
        assert config.entity_types == ["LINE", "CIRCLE"]

    def test_entity_filter_creation(self):
        """Test EntityFilter creation."""
        from src.core.cad.dxf.filters import EntityFilter, FilterConfig

        config = FilterConfig(layers=["0"])
        filter_obj = EntityFilter(config)
        assert filter_obj.config == config

    def test_filter_result_creation(self):
        """Test FilterResult creation."""
        from src.core.cad.dxf.filters import FilterResult

        result = FilterResult(
            total_scanned=100,
            total_matched=50,
        )
        assert result.match_rate == 0.5

    def test_entity_filter_type_groups(self):
        """Test entity type groups."""
        from src.core.cad.dxf.filters import EntityFilter

        groups = EntityFilter.TYPE_GROUPS
        assert "geometry" in groups
        assert "text" in groups
        assert "dimensions" in groups
        assert "LINE" in groups["geometry"]
        assert "TEXT" in groups["text"]


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
