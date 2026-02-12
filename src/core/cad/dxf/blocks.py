"""
Block reference expansion for DXF files.

Handles INSERT entities and nested block references.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class BlockReference:
    """Block reference (INSERT) information."""
    name: str
    handle: str
    layer: str
    insert_point: Tuple[float, float, float]
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotation: float = 0.0  # Degrees
    col_count: int = 1
    row_count: int = 1
    col_spacing: float = 0.0
    row_spacing: float = 0.0
    attributes: Dict[str, str] = field(default_factory=dict)

    @property
    def is_array(self) -> bool:
        """Check if this is an array insert."""
        return self.col_count > 1 or self.row_count > 1

    @property
    def uniform_scale(self) -> bool:
        """Check if scale is uniform."""
        return self.scale[0] == self.scale[1] == self.scale[2]

    def get_transform_matrix(self) -> List[List[float]]:
        """Get 4x4 transformation matrix."""
        # Rotation in radians
        rad = math.radians(self.rotation)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)

        sx, sy, sz = self.scale
        tx, ty, tz = self.insert_point

        # Combined scale-rotation-translation matrix
        return [
            [sx * cos_r, -sy * sin_r, 0, tx],
            [sx * sin_r, sy * cos_r, 0, ty],
            [0, 0, sz, tz],
            [0, 0, 0, 1],
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "handle": self.handle,
            "layer": self.layer,
            "insert_point": self.insert_point,
            "scale": self.scale,
            "rotation": self.rotation,
            "col_count": self.col_count,
            "row_count": self.row_count,
            "attributes": self.attributes,
            "is_array": self.is_array,
        }


@dataclass
class ExpandedEntity:
    """Entity from expanded block."""
    original_entity: Any
    entity_type: str
    layer: str
    block_path: List[str]  # Path of block names
    transform: List[List[float]]  # Combined transformation
    depth: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "layer": self.layer,
            "block_path": self.block_path,
            "depth": self.depth,
        }


@dataclass
class ExpandedBlock:
    """Expanded block with all entities."""
    name: str
    entities: List[ExpandedEntity] = field(default_factory=list)
    nested_blocks: List["ExpandedBlock"] = field(default_factory=list)
    total_entity_count: int = 0
    max_depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entity_count": len(self.entities),
            "total_entity_count": self.total_entity_count,
            "nested_blocks": [b.to_dict() for b in self.nested_blocks],
            "max_depth": self.max_depth,
        }


class BlockExpander:
    """
    Expands block references in DXF files.

    Handles nested blocks, transformations, and arrays.
    """

    def __init__(self, max_depth: int = 10):
        """
        Initialize block expander.

        Args:
            max_depth: Maximum nesting depth
        """
        self._max_depth = max_depth
        self._blocks: Dict[str, Any] = {}
        self._references: List[BlockReference] = []
        self._expanded_entities: List[ExpandedEntity] = []

    @property
    def blocks(self) -> Dict[str, Any]:
        return self._blocks.copy()

    @property
    def references(self) -> List[BlockReference]:
        return self._references.copy()

    def load_blocks(self, dxf_doc: Any) -> None:
        """
        Load block definitions from DXF document.

        Args:
            dxf_doc: ezdxf document
        """
        self._blocks.clear()

        if hasattr(dxf_doc, "blocks"):
            for block in dxf_doc.blocks:
                self._blocks[block.name] = block

        logger.info(f"Loaded {len(self._blocks)} block definitions")

    def find_references(self, dxf_doc: Any) -> List[BlockReference]:
        """
        Find all block references (INSERTs) in modelspace.

        Args:
            dxf_doc: ezdxf document

        Returns:
            List of BlockReference
        """
        self._references.clear()

        if hasattr(dxf_doc, "modelspace"):
            for entity in dxf_doc.modelspace():
                if entity.dxftype() == "INSERT":
                    ref = self._create_reference(entity)
                    self._references.append(ref)

        logger.info(f"Found {len(self._references)} block references")
        return self._references

    def _create_reference(self, insert_entity: Any) -> BlockReference:
        """Create BlockReference from INSERT entity."""
        # Get basic properties
        name = insert_entity.dxf.name
        handle = insert_entity.dxf.handle
        layer = insert_entity.dxf.layer

        # Get insert point
        insert_point = (
            insert_entity.dxf.insert.x,
            insert_entity.dxf.insert.y,
            insert_entity.dxf.insert.z if hasattr(insert_entity.dxf.insert, "z") else 0.0,
        )

        # Get scale
        scale = (
            getattr(insert_entity.dxf, "xscale", 1.0),
            getattr(insert_entity.dxf, "yscale", 1.0),
            getattr(insert_entity.dxf, "zscale", 1.0),
        )

        # Get rotation
        rotation = getattr(insert_entity.dxf, "rotation", 0.0)

        # Get array properties
        col_count = getattr(insert_entity.dxf, "column_count", 1)
        row_count = getattr(insert_entity.dxf, "row_count", 1)
        col_spacing = getattr(insert_entity.dxf, "column_spacing", 0.0)
        row_spacing = getattr(insert_entity.dxf, "row_spacing", 0.0)

        # Get attributes
        attributes = {}
        if hasattr(insert_entity, "attribs"):
            for attrib in insert_entity.attribs:
                tag = attrib.dxf.tag
                value = attrib.dxf.text
                attributes[tag] = value

        return BlockReference(
            name=name,
            handle=handle,
            layer=layer,
            insert_point=insert_point,
            scale=scale,
            rotation=rotation,
            col_count=col_count,
            row_count=row_count,
            col_spacing=col_spacing,
            row_spacing=row_spacing,
            attributes=attributes,
        )

    def expand_reference(
        self,
        reference: BlockReference,
        include_nested: bool = True,
    ) -> ExpandedBlock:
        """
        Expand a block reference to its constituent entities.

        Args:
            reference: Block reference to expand
            include_nested: Include nested block references

        Returns:
            ExpandedBlock
        """
        block_def = self._blocks.get(reference.name)
        if block_def is None:
            logger.warning(f"Block definition not found: {reference.name}")
            return ExpandedBlock(name=reference.name)

        expanded = ExpandedBlock(name=reference.name)
        transform = reference.get_transform_matrix()

        self._expand_block_recursive(
            block_def=block_def,
            expanded=expanded,
            transform=transform,
            block_path=[reference.name],
            depth=0,
            include_nested=include_nested,
        )

        return expanded

    def _expand_block_recursive(
        self,
        block_def: Any,
        expanded: ExpandedBlock,
        transform: List[List[float]],
        block_path: List[str],
        depth: int,
        include_nested: bool,
    ) -> None:
        """Recursively expand block entities."""
        if depth > self._max_depth:
            logger.warning(f"Max depth {self._max_depth} reached at {block_path}")
            return

        expanded.max_depth = max(expanded.max_depth, depth)

        for entity in block_def:
            entity_type = entity.dxftype()

            if entity_type == "INSERT" and include_nested:
                # Nested block reference
                nested_ref = self._create_reference(entity)
                nested_block_def = self._blocks.get(nested_ref.name)

                if nested_block_def:
                    nested_expanded = ExpandedBlock(name=nested_ref.name)
                    nested_transform = self._multiply_matrices(
                        transform,
                        nested_ref.get_transform_matrix(),
                    )

                    self._expand_block_recursive(
                        block_def=nested_block_def,
                        expanded=nested_expanded,
                        transform=nested_transform,
                        block_path=block_path + [nested_ref.name],
                        depth=depth + 1,
                        include_nested=include_nested,
                    )

                    expanded.nested_blocks.append(nested_expanded)
                    expanded.total_entity_count += nested_expanded.total_entity_count
            else:
                # Regular entity
                layer = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"

                expanded_entity = ExpandedEntity(
                    original_entity=entity,
                    entity_type=entity_type,
                    layer=layer,
                    block_path=block_path,
                    transform=transform,
                    depth=depth,
                )

                expanded.entities.append(expanded_entity)
                expanded.total_entity_count += 1

    def _multiply_matrices(
        self,
        a: List[List[float]],
        b: List[List[float]],
    ) -> List[List[float]]:
        """Multiply two 4x4 matrices."""
        result = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    result[i][j] += a[i][k] * b[k][j]
        return result

    def expand_all(self, dxf_doc: Any) -> List[ExpandedBlock]:
        """
        Expand all block references in document.

        Args:
            dxf_doc: ezdxf document

        Returns:
            List of ExpandedBlock
        """
        self.load_blocks(dxf_doc)
        references = self.find_references(dxf_doc)

        expanded_blocks = []
        for ref in references:
            expanded = self.expand_reference(ref)
            expanded_blocks.append(expanded)

        return expanded_blocks

    def get_block_tree(self, dxf_doc: Any) -> Dict[str, Any]:
        """
        Get block dependency tree.

        Args:
            dxf_doc: ezdxf document

        Returns:
            Block tree structure
        """
        self.load_blocks(dxf_doc)

        # Analyze dependencies
        dependencies: Dict[str, Set[str]] = {}

        for name, block_def in self._blocks.items():
            deps = set()
            for entity in block_def:
                if entity.dxftype() == "INSERT":
                    deps.add(entity.dxf.name)
            dependencies[name] = deps

        # Build tree
        tree = {
            "total_blocks": len(self._blocks),
            "blocks": {},
        }

        for name, deps in dependencies.items():
            tree["blocks"][name] = {
                "dependencies": list(deps),
                "is_leaf": len(deps) == 0,
            }

        return tree

    def get_statistics(self) -> Dict[str, Any]:
        """Get expansion statistics."""
        return {
            "total_blocks": len(self._blocks),
            "total_references": len(self._references),
            "unique_blocks_used": len(set(r.name for r in self._references)),
            "array_references": sum(1 for r in self._references if r.is_array),
            "references_with_attributes": sum(1 for r in self._references if r.attributes),
        }


def expand_blocks(dxf_source: Union[str, Path, Any]) -> List[ExpandedBlock]:
    """
    Convenience function to expand all blocks.

    Args:
        dxf_source: DXF file path or ezdxf document

    Returns:
        List of ExpandedBlock
    """
    try:
        import ezdxf
    except ImportError:
        raise ImportError("ezdxf required: pip install ezdxf")

    if isinstance(dxf_source, (str, Path)):
        doc = ezdxf.readfile(str(dxf_source))
    else:
        doc = dxf_source

    expander = BlockExpander()
    return expander.expand_all(doc)


def get_block_tree(dxf_source: Union[str, Path, Any]) -> Dict[str, Any]:
    """
    Convenience function to get block dependency tree.

    Args:
        dxf_source: DXF file path or ezdxf document

    Returns:
        Block tree structure
    """
    try:
        import ezdxf
    except ImportError:
        raise ImportError("ezdxf required: pip install ezdxf")

    if isinstance(dxf_source, (str, Path)):
        doc = ezdxf.readfile(str(dxf_source))
    else:
        doc = dxf_source

    expander = BlockExpander()
    return expander.get_block_tree(doc)
