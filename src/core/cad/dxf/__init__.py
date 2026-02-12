"""
Enhanced DXF Parsing Module (C2).

Provides advanced DXF parsing capabilities:
- Layer hierarchy analysis
- Block reference expansion
- ATTRIB attribute extraction
- Entity filtering and querying
"""

from src.core.cad.dxf.hierarchy import (
    LayerHierarchy,
    LayerNode,
    LayerAnalyzer,
    analyze_layers,
)
from src.core.cad.dxf.blocks import (
    BlockExpander,
    BlockReference,
    ExpandedBlock,
    expand_blocks,
    get_block_tree,
)
from src.core.cad.dxf.attributes import (
    AttributeExtractor,
    ATTRIBData,
    AttributeDefinition,
    extract_attributes,
    extract_titleblock_attributes,
)
from src.core.cad.dxf.filters import (
    EntityFilter,
    FilterConfig,
    FilterResult,
    filter_entities,
    filter_by_layer,
    filter_by_type,
)

__all__ = [
    # Hierarchy
    "LayerHierarchy",
    "LayerNode",
    "LayerAnalyzer",
    "analyze_layers",
    # Blocks
    "BlockExpander",
    "BlockReference",
    "ExpandedBlock",
    "expand_blocks",
    "get_block_tree",
    # Attributes
    "AttributeExtractor",
    "ATTRIBData",
    "AttributeDefinition",
    "extract_attributes",
    "extract_titleblock_attributes",
    # Filters
    "EntityFilter",
    "FilterConfig",
    "FilterResult",
    "filter_entities",
    "filter_by_layer",
    "filter_by_type",
]
