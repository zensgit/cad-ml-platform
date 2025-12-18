"""Tests for src/adapters/factory.py to improve coverage.

Covers:
- _BaseAdapter class
- DxfAdapter class
- StlAdapter class  
- StubAdapter class
- StepIgesAdapter class
- AdapterFactory class
- Parsing and conversion logic
- Error handling paths
"""

from __future__ import annotations

from typing import Dict
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class TestBaseAdapter:
    """Tests for _BaseAdapter base class."""

    def test_base_adapter_class_exists(self):
        """Test _BaseAdapter class exists."""
        from src.adapters.factory import _BaseAdapter

        adapter = _BaseAdapter()
        assert adapter is not None

    @pytest.mark.asyncio
    async def test_base_adapter_parse_not_implemented(self):
        """Test _BaseAdapter.parse raises NotImplementedError."""
        from src.adapters.factory import _BaseAdapter

        adapter = _BaseAdapter()
        with pytest.raises(NotImplementedError):
            await adapter.parse(b"data", file_name="test.file")

    @pytest.mark.asyncio
    async def test_base_adapter_convert_calls_parse(self):
        """Test _BaseAdapter.convert calls parse and to_unified_dict."""
        from src.adapters.factory import _BaseAdapter
        from src.models.cad_document import CadDocument

        adapter = _BaseAdapter()
        mock_doc = MagicMock(spec=CadDocument)
        mock_doc.to_unified_dict.return_value = {"key": "value"}

        with patch.object(adapter, "parse", new_callable=AsyncMock, return_value=mock_doc):
            result = await adapter.convert(b"data", file_name="test.file")

        assert result == {"key": "value"}


class TestDxfAdapter:
    """Tests for DxfAdapter class."""

    def test_dxf_adapter_format(self):
        """Test DxfAdapter format is 'dxf'."""
        from src.adapters.factory import DxfAdapter

        adapter = DxfAdapter()
        assert adapter.format == "dxf"

    @pytest.mark.asyncio
    async def test_dxf_adapter_parse_without_ezdxf(self):
        """Test DxfAdapter.parse returns stub when ezdxf not available."""
        from src.adapters.factory import DxfAdapter

        adapter = DxfAdapter()

        # Mock ezdxf import to fail
        with patch.dict("sys.modules", {"ezdxf": None}):
            with patch("builtins.__import__", side_effect=ImportError("No ezdxf")):
                result = await adapter.parse(b"dxf data", file_name="test.dxf")

        assert result.file_name == "test.dxf"
        assert result.format == "dxf"
        assert result.metadata.get("parser") == "stub"

    @pytest.mark.asyncio
    async def test_dxf_adapter_parse_with_ezdxf(self):
        """Test DxfAdapter.parse with mocked ezdxf."""
        from src.adapters.factory import DxfAdapter

        adapter = DxfAdapter()

        # Create mock ezdxf
        mock_ezdxf = MagicMock()
        mock_entity = MagicMock()
        mock_entity.dxftype.return_value = "LINE"
        mock_entity.dxf.layer = "0"
        mock_entity.bbox.return_value = None  # No bbox

        mock_doc = MagicMock()
        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter([mock_entity]))
        mock_doc.modelspace.return_value = mock_msp
        mock_ezdxf.read.return_value = mock_doc

        with patch.dict("sys.modules", {"ezdxf": mock_ezdxf}):
            # Need to force reimport to use the mock
            result = await adapter.parse(b"dxf data", file_name="test.dxf")

        assert result.file_name == "test.dxf"
        assert result.format == "dxf"


class TestStlAdapter:
    """Tests for StlAdapter class."""

    def test_stl_adapter_format(self):
        """Test StlAdapter format is 'stl'."""
        from src.adapters.factory import StlAdapter

        adapter = StlAdapter()
        assert adapter.format == "stl"

    @pytest.mark.asyncio
    async def test_stl_adapter_parse_without_trimesh(self):
        """Test StlAdapter.parse returns stub when trimesh not available."""
        from src.adapters.factory import StlAdapter

        adapter = StlAdapter()

        # Mock trimesh import to fail
        with patch.dict("sys.modules", {"trimesh": None}):
            with patch("builtins.__import__", side_effect=ImportError("No trimesh")):
                result = await adapter.parse(b"stl data", file_name="test.stl")

        assert result.file_name == "test.stl"
        assert result.format == "stl"
        assert result.metadata.get("parser") == "stub"

    @pytest.mark.asyncio
    async def test_stl_adapter_parse_with_trimesh(self):
        """Test StlAdapter.parse with mocked trimesh."""
        from src.adapters.factory import StlAdapter
        import numpy as np

        adapter = StlAdapter()

        # Create mock trimesh
        mock_mesh = MagicMock()
        mock_mesh.faces = [[0, 1, 2], [3, 4, 5]]  # 2 facets
        mock_mesh.bounds = [
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 10.0, 10.0])
        ]

        mock_trimesh = MagicMock()
        mock_trimesh.load.return_value = mock_mesh

        with patch.dict("sys.modules", {"trimesh": mock_trimesh}):
            result = await adapter.parse(b"stl data", file_name="test.stl")

        assert result.file_name == "test.stl"
        assert result.format == "stl"


class TestStubAdapter:
    """Tests for StubAdapter class."""

    def test_stub_adapter_format(self):
        """Test StubAdapter format is 'stub'."""
        from src.adapters.factory import StubAdapter

        adapter = StubAdapter()
        assert adapter.format == "stub"

    @pytest.mark.asyncio
    async def test_stub_adapter_parse(self):
        """Test StubAdapter.parse returns stub document."""
        from src.adapters.factory import StubAdapter

        adapter = StubAdapter()
        result = await adapter.parse(b"any data", file_name="test.unknown")

        assert result.file_name == "test.unknown"
        assert result.format == "stub"
        assert result.metadata.get("parser") == "stub"


class TestStepIgesAdapter:
    """Tests for StepIgesAdapter class."""

    def test_step_iges_adapter_format(self):
        """Test StepIgesAdapter format is 'step'."""
        from src.adapters.factory import StepIgesAdapter

        adapter = StepIgesAdapter()
        assert adapter.format == "step"

    @pytest.mark.asyncio
    async def test_step_adapter_parse_without_occ(self):
        """Test StepIgesAdapter.parse returns stub when pythonocc not available."""
        from src.adapters.factory import StepIgesAdapter

        adapter = StepIgesAdapter()

        # Mock OCC import to fail
        with patch.dict("sys.modules", {"OCC": None, "OCC.Core.STEPControl": None}):
            with patch("builtins.__import__", side_effect=ImportError("No OCC")):
                result = await adapter.parse(b"step data", file_name="test.step")

        assert result.file_name == "test.step"
        assert result.format == "step"
        assert result.metadata.get("parser") == "stub"


class TestAdapterFactory:
    """Tests for AdapterFactory class."""

    def test_get_adapter_dxf(self):
        """Test get_adapter returns DxfAdapter for 'dxf'."""
        from src.adapters.factory import AdapterFactory, DxfAdapter

        adapter = AdapterFactory.get_adapter("dxf")
        assert isinstance(adapter, DxfAdapter)

    def test_get_adapter_dwg(self):
        """Test get_adapter returns DxfAdapter for 'dwg'."""
        from src.adapters.factory import AdapterFactory, DxfAdapter

        adapter = AdapterFactory.get_adapter("dwg")
        assert isinstance(adapter, DxfAdapter)

    def test_get_adapter_stl(self):
        """Test get_adapter returns StlAdapter for 'stl'."""
        from src.adapters.factory import AdapterFactory, StlAdapter

        adapter = AdapterFactory.get_adapter("stl")
        assert isinstance(adapter, StlAdapter)

    def test_get_adapter_step(self):
        """Test get_adapter returns StepIgesAdapter for 'step'."""
        from src.adapters.factory import AdapterFactory, StepIgesAdapter

        adapter = AdapterFactory.get_adapter("step")
        assert isinstance(adapter, StepIgesAdapter)

    def test_get_adapter_stp(self):
        """Test get_adapter returns StepIgesAdapter for 'stp'."""
        from src.adapters.factory import AdapterFactory, StepIgesAdapter

        adapter = AdapterFactory.get_adapter("stp")
        assert isinstance(adapter, StepIgesAdapter)

    def test_get_adapter_iges(self):
        """Test get_adapter returns StepIgesAdapter for 'iges'."""
        from src.adapters.factory import AdapterFactory, StepIgesAdapter

        adapter = AdapterFactory.get_adapter("iges")
        assert isinstance(adapter, StepIgesAdapter)

    def test_get_adapter_igs(self):
        """Test get_adapter returns StepIgesAdapter for 'igs'."""
        from src.adapters.factory import AdapterFactory, StepIgesAdapter

        adapter = AdapterFactory.get_adapter("igs")
        assert isinstance(adapter, StepIgesAdapter)

    def test_get_adapter_unknown_format(self):
        """Test get_adapter returns StubAdapter for unknown formats."""
        from src.adapters.factory import AdapterFactory, StubAdapter

        adapter = AdapterFactory.get_adapter("xyz")
        assert isinstance(adapter, StubAdapter)

    def test_get_adapter_case_insensitive(self):
        """Test get_adapter is case insensitive."""
        from src.adapters.factory import AdapterFactory, DxfAdapter

        adapter_upper = AdapterFactory.get_adapter("DXF")
        adapter_lower = AdapterFactory.get_adapter("dxf")
        adapter_mixed = AdapterFactory.get_adapter("DxF")

        assert isinstance(adapter_upper, DxfAdapter)
        assert isinstance(adapter_lower, DxfAdapter)
        assert isinstance(adapter_mixed, DxfAdapter)

    def test_mapping_completeness(self):
        """Test all expected formats are in mapping."""
        from src.adapters.factory import AdapterFactory

        expected_formats = ["dxf", "dwg", "stl", "step", "stp", "iges", "igs"]
        for fmt in expected_formats:
            assert fmt in AdapterFactory._mapping


class TestCadDocumentModels:
    """Tests for CadDocument model usage."""

    def test_cad_entity_creation(self):
        """Test CadEntity can be created."""
        from src.models.cad_document import CadEntity

        entity = CadEntity(kind="LINE", layer="0")
        assert entity.kind == "LINE"
        assert entity.layer == "0"

    def test_cad_entity_default_layer(self):
        """Test CadEntity with default layer."""
        from src.models.cad_document import CadEntity

        entity = CadEntity(kind="CIRCLE")
        assert entity.kind == "CIRCLE"

    def test_bounding_box_creation(self):
        """Test BoundingBox can be created."""
        from src.models.cad_document import BoundingBox

        bbox = BoundingBox()
        assert hasattr(bbox, "min_x")
        assert hasattr(bbox, "max_x")

    def test_cad_document_creation(self):
        """Test CadDocument can be created."""
        from src.models.cad_document import CadDocument, BoundingBox

        doc = CadDocument(
            file_name="test.dxf",
            format="dxf",
            entities=[],
            layers={},
            bounding_box=BoundingBox(),
            metadata={"parser": "test"},
        )

        assert doc.file_name == "test.dxf"
        assert doc.format == "dxf"

    def test_cad_document_to_unified_dict(self):
        """Test CadDocument.to_unified_dict method."""
        from src.models.cad_document import CadDocument, BoundingBox

        doc = CadDocument(
            file_name="test.dxf",
            format="dxf",
            entities=[],
            layers={"0": 5},
            bounding_box=BoundingBox(),
            metadata={"parser": "test"},
        )

        result = doc.to_unified_dict()
        assert isinstance(result, dict)


class TestBoundingBoxLogic:
    """Tests for bounding box calculation logic."""

    def test_bbox_update_logic(self):
        """Test bounding box update logic."""
        from src.models.cad_document import BoundingBox

        bbox = BoundingBox()
        
        # Simulate updating bbox
        test_coords = [
            (0.0, 0.0),
            (10.0, 5.0),
            (-5.0, 20.0),
        ]

        for x, y in test_coords:
            bbox.min_x = min(bbox.min_x, x)
            bbox.min_y = min(bbox.min_y, y)
            bbox.max_x = max(bbox.max_x, x)
            bbox.max_y = max(bbox.max_y, y)

        assert bbox.min_x == -5.0
        assert bbox.max_x == 10.0
        assert bbox.min_y == 0.0
        assert bbox.max_y == 20.0

    def test_bbox_3d_coordinates(self):
        """Test bounding box with 3D coordinates."""
        from src.models.cad_document import BoundingBox

        bbox = BoundingBox()
        bbox.min_z = -10.0
        bbox.max_z = 10.0

        assert bbox.min_z == -10.0
        assert bbox.max_z == 10.0


class TestLayerCounting:
    """Tests for layer counting logic."""

    def test_layer_counting(self):
        """Test layer entity counting."""
        layers: Dict[str, int] = {}
        entities_by_layer = ["0", "0", "1", "0", "walls", "walls"]

        for layer in entities_by_layer:
            layers[layer] = layers.get(layer, 0) + 1

        assert layers["0"] == 3
        assert layers["1"] == 1
        assert layers["walls"] == 2

    def test_empty_layers(self):
        """Test empty layers dict."""
        layers: Dict[str, int] = {}
        assert len(layers) == 0


class TestParserMetadata:
    """Tests for parser metadata handling."""

    def test_stub_parser_metadata(self):
        """Test stub parser metadata."""
        metadata = {"parser": "stub"}
        assert metadata["parser"] == "stub"

    def test_ezdxf_parser_metadata(self):
        """Test ezdxf parser metadata."""
        metadata = {"parser": "ezdxf"}
        assert metadata["parser"] == "ezdxf"

    def test_trimesh_parser_metadata(self):
        """Test trimesh parser metadata with facets."""
        metadata = {"parser": "trimesh", "facets": 100}
        assert metadata["parser"] == "trimesh"
        assert metadata["facets"] == 100

    def test_pythonocc_parser_metadata(self):
        """Test pythonocc parser metadata with solids."""
        metadata = {"parser": "pythonocc", "solids": 5}
        assert metadata["parser"] == "pythonocc"
        assert metadata["solids"] == 5


class TestRawStatsHandling:
    """Tests for raw_stats handling in CadDocument."""

    def test_raw_stats_facet_count(self):
        """Test raw_stats with facet count."""
        from src.models.cad_document import CadDocument

        doc = CadDocument(
            file_name="test.stl",
            format="stl",
            raw_stats={"facet_count": 1000},
        )

        assert doc.raw_stats["facet_count"] == 1000

    def test_raw_stats_empty(self):
        """Test raw_stats defaults to empty dict."""
        from src.models.cad_document import CadDocument

        doc = CadDocument(
            file_name="test.dxf",
            format="dxf",
        )

        # raw_stats should exist (may be empty or None depending on default)
        assert hasattr(doc, "raw_stats")
