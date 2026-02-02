"""
Layer hierarchy analysis for DXF files.

Analyzes layer structure, relationships, and properties.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class LayerNode:
    """Node in layer hierarchy tree."""
    name: str
    full_name: str  # Full layer name including parent path
    parent: Optional["LayerNode"] = None
    children: List["LayerNode"] = field(default_factory=list)

    # Layer properties
    color: int = 7  # White/black
    linetype: str = "Continuous"
    lineweight: int = -1  # Default
    is_on: bool = True
    is_frozen: bool = False
    is_locked: bool = False
    plot: bool = True

    # Statistics
    entity_count: int = 0
    entity_types: Dict[str, int] = field(default_factory=dict)

    @property
    def depth(self) -> int:
        """Get depth in hierarchy."""
        if self.parent is None:
            return 0
        return self.parent.depth + 1

    @property
    def path(self) -> List[str]:
        """Get path from root."""
        if self.parent is None:
            return [self.name]
        return self.parent.path + [self.name]

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    def add_child(self, child: "LayerNode") -> None:
        """Add a child node."""
        child.parent = self
        self.children.append(child)

    def find_child(self, name: str) -> Optional["LayerNode"]:
        """Find immediate child by name."""
        for child in self.children:
            if child.name == name:
                return child
        return None

    def get_all_descendants(self) -> List["LayerNode"]:
        """Get all descendant nodes."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "full_name": self.full_name,
            "depth": self.depth,
            "color": self.color,
            "linetype": self.linetype,
            "is_on": self.is_on,
            "is_frozen": self.is_frozen,
            "entity_count": self.entity_count,
            "entity_types": self.entity_types,
            "children": [c.to_dict() for c in self.children],
        }


class LayerHierarchy:
    """
    Layer hierarchy representation.

    Organizes layers into a tree structure based on naming conventions.
    """

    # Common layer name separators
    SEPARATORS = ["-", "_", "|", "/", "\\", "$"]

    def __init__(self, separator: Optional[str] = None):
        """
        Initialize layer hierarchy.

        Args:
            separator: Layer name separator (auto-detected if None)
        """
        self._root = LayerNode(name="ROOT", full_name="")
        self._layers: Dict[str, LayerNode] = {}
        self._separator = separator
        self._detected_separator: Optional[str] = None

    @property
    def root(self) -> LayerNode:
        return self._root

    @property
    def separator(self) -> Optional[str]:
        return self._separator or self._detected_separator

    @property
    def layers(self) -> Dict[str, LayerNode]:
        return self._layers.copy()

    def add_layer(
        self,
        name: str,
        color: int = 7,
        linetype: str = "Continuous",
        is_on: bool = True,
        is_frozen: bool = False,
        **kwargs,
    ) -> LayerNode:
        """
        Add a layer to the hierarchy.

        Args:
            name: Layer name
            color: Layer color
            linetype: Layer linetype
            is_on: Layer visibility
            is_frozen: Layer frozen state

        Returns:
            LayerNode
        """
        if name in self._layers:
            return self._layers[name]

        # Detect separator if not set
        if self._separator is None and self._detected_separator is None:
            self._detected_separator = self._detect_separator(name)

        # Parse hierarchy from name
        sep = self.separator
        if sep:
            parts = name.split(sep)
        else:
            parts = [name]

        # Build hierarchy
        current = self._root
        current_path = []

        for part in parts:
            current_path.append(part)
            full_name = sep.join(current_path) if sep else part

            existing = current.find_child(part)
            if existing:
                current = existing
            else:
                node = LayerNode(
                    name=part,
                    full_name=full_name,
                    color=color if full_name == name else 7,
                    linetype=linetype if full_name == name else "Continuous",
                    is_on=is_on if full_name == name else True,
                    is_frozen=is_frozen if full_name == name else False,
                )
                current.add_child(node)
                current = node

        # Store reference to actual layer
        self._layers[name] = current
        return current

    def _detect_separator(self, name: str) -> Optional[str]:
        """Detect layer name separator."""
        for sep in self.SEPARATORS:
            if sep in name:
                return sep
        return None

    def get_layer(self, name: str) -> Optional[LayerNode]:
        """Get layer by name."""
        return self._layers.get(name)

    def get_layers_by_prefix(self, prefix: str) -> List[LayerNode]:
        """Get all layers with given prefix."""
        return [
            node for name, node in self._layers.items()
            if name.startswith(prefix)
        ]

    def get_layers_by_pattern(self, pattern: str) -> List[LayerNode]:
        """Get layers matching regex pattern."""
        regex = re.compile(pattern, re.IGNORECASE)
        return [
            node for name, node in self._layers.items()
            if regex.search(name)
        ]

    def iter_layers(self, depth_first: bool = True) -> Iterator[LayerNode]:
        """Iterate over all layers."""
        def _iter_node(node: LayerNode) -> Iterator[LayerNode]:
            if node != self._root:
                yield node
            for child in node.children:
                yield from _iter_node(child)

        yield from _iter_node(self._root)

    def get_statistics(self) -> Dict[str, Any]:
        """Get hierarchy statistics."""
        depths = [node.depth for node in self._layers.values()]

        return {
            "total_layers": len(self._layers),
            "max_depth": max(depths) if depths else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "separator": self.separator,
            "root_children": len(self._root.children),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "separator": self.separator,
            "statistics": self.get_statistics(),
            "hierarchy": self._root.to_dict(),
        }

    def print_tree(self, max_depth: Optional[int] = None) -> str:
        """Generate tree representation."""
        lines = []

        def _print_node(node: LayerNode, prefix: str = "", is_last: bool = True):
            if node == self._root:
                for i, child in enumerate(node.children):
                    _print_node(child, "", i == len(node.children) - 1)
            else:
                if max_depth and node.depth > max_depth:
                    return

                connector = "└── " if is_last else "├── "
                status = ""
                if not node.is_on:
                    status = " [OFF]"
                elif node.is_frozen:
                    status = " [FROZEN]"

                lines.append(f"{prefix}{connector}{node.name}{status} ({node.entity_count})")

                child_prefix = prefix + ("    " if is_last else "│   ")
                for i, child in enumerate(node.children):
                    _print_node(child, child_prefix, i == len(node.children) - 1)

        _print_node(self._root)
        return "\n".join(lines)


class LayerAnalyzer:
    """
    Analyzes DXF layer structure and usage.
    """

    def __init__(self):
        """Initialize layer analyzer."""
        self._hierarchy: Optional[LayerHierarchy] = None

    def analyze(self, dxf_doc: Any) -> LayerHierarchy:
        """
        Analyze layers in a DXF document.

        Args:
            dxf_doc: ezdxf document

        Returns:
            LayerHierarchy
        """
        hierarchy = LayerHierarchy()

        # Add all layers from layer table
        if hasattr(dxf_doc, "layers"):
            for layer in dxf_doc.layers:
                hierarchy.add_layer(
                    name=layer.dxf.name,
                    color=layer.dxf.color,
                    linetype=layer.dxf.linetype,
                    is_on=layer.is_on(),
                    is_frozen=layer.is_frozen(),
                    is_locked=layer.is_locked(),
                )

        # Count entities per layer
        if hasattr(dxf_doc, "modelspace"):
            for entity in dxf_doc.modelspace():
                layer_name = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"
                entity_type = entity.dxftype()

                node = hierarchy.get_layer(layer_name)
                if node is None:
                    # Layer referenced but not in table
                    node = hierarchy.add_layer(layer_name)

                node.entity_count += 1
                if entity_type not in node.entity_types:
                    node.entity_types[entity_type] = 0
                node.entity_types[entity_type] += 1

        self._hierarchy = hierarchy
        return hierarchy

    def analyze_file(self, file_path: Union[str, Path]) -> LayerHierarchy:
        """
        Analyze layers in a DXF file.

        Args:
            file_path: Path to DXF file

        Returns:
            LayerHierarchy
        """
        try:
            import ezdxf
        except ImportError:
            raise ImportError("ezdxf required: pip install ezdxf")

        doc = ezdxf.readfile(str(file_path))
        return self.analyze(doc)

    def get_layer_usage_report(self) -> Dict[str, Any]:
        """Generate layer usage report."""
        if not self._hierarchy:
            return {}

        layers = list(self._hierarchy.iter_layers())

        # Identify unused layers
        unused = [l for l in layers if l.entity_count == 0]

        # Identify hidden layers with content
        hidden_with_content = [
            l for l in layers
            if (not l.is_on or l.is_frozen) and l.entity_count > 0
        ]

        # Entity distribution
        total_entities = sum(l.entity_count for l in layers)
        type_totals: Dict[str, int] = {}
        for layer in layers:
            for etype, count in layer.entity_types.items():
                type_totals[etype] = type_totals.get(etype, 0) + count

        return {
            "total_layers": len(layers),
            "unused_layers": [l.full_name for l in unused],
            "hidden_with_content": [l.full_name for l in hidden_with_content],
            "total_entities": total_entities,
            "entity_type_distribution": type_totals,
            "top_layers_by_entities": sorted(
                [(l.full_name, l.entity_count) for l in layers],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }


def analyze_layers(dxf_source: Union[str, Path, Any]) -> LayerHierarchy:
    """
    Convenience function to analyze DXF layers.

    Args:
        dxf_source: DXF file path or ezdxf document

    Returns:
        LayerHierarchy
    """
    analyzer = LayerAnalyzer()

    if isinstance(dxf_source, (str, Path)):
        return analyzer.analyze_file(dxf_source)
    else:
        return analyzer.analyze(dxf_source)
