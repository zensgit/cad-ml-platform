"""
Entity filtering for DXF files.

Provides flexible filtering and querying of DXF entities.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Entity filter configuration."""
    # Layer filters
    layers: Optional[List[str]] = None
    layer_pattern: Optional[str] = None
    exclude_layers: Optional[List[str]] = None

    # Entity type filters
    entity_types: Optional[List[str]] = None
    exclude_types: Optional[List[str]] = None

    # Property filters
    color: Optional[int] = None
    linetype: Optional[str] = None
    lineweight: Optional[int] = None

    # Spatial filters
    min_x: Optional[float] = None
    max_x: Optional[float] = None
    min_y: Optional[float] = None
    max_y: Optional[float] = None

    # Block filters
    in_block: Optional[str] = None
    exclude_blocks: Optional[List[str]] = None

    # Visibility
    include_invisible: bool = False
    include_frozen_layers: bool = False

    # Custom filter
    custom_filter: Optional[Callable[[Any], bool]] = None


@dataclass
class FilterResult:
    """Result of entity filtering."""
    entities: List[Any] = field(default_factory=list)
    total_scanned: int = 0
    total_matched: int = 0
    by_layer: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)

    @property
    def match_rate(self) -> float:
        if self.total_scanned == 0:
            return 0.0
        return self.total_matched / self.total_scanned

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_scanned": self.total_scanned,
            "total_matched": self.total_matched,
            "match_rate": round(self.match_rate, 4),
            "by_layer": self.by_layer,
            "by_type": self.by_type,
        }


class EntityFilter:
    """
    Flexible entity filter for DXF documents.

    Supports filtering by layer, type, properties, and spatial bounds.
    """

    # Common entity type groups
    TYPE_GROUPS = {
        "geometry": ["LINE", "CIRCLE", "ARC", "ELLIPSE", "SPLINE", "POLYLINE", "LWPOLYLINE"],
        "text": ["TEXT", "MTEXT"],
        "dimensions": ["DIMENSION", "LEADER", "TOLERANCE"],
        "hatches": ["HATCH", "SOLID"],
        "blocks": ["INSERT"],
        "points": ["POINT"],
        "images": ["IMAGE", "WIPEOUT"],
        "3d": ["3DFACE", "3DSOLID", "BODY", "MESH"],
    }

    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize entity filter.

        Args:
            config: Filter configuration
        """
        self._config = config or FilterConfig()
        self._layer_states: Dict[str, Dict[str, bool]] = {}

    @property
    def config(self) -> FilterConfig:
        return self._config

    def set_config(self, config: FilterConfig) -> "EntityFilter":
        """Set filter configuration."""
        self._config = config
        return self

    def filter(
        self,
        dxf_doc: Any,
        space: str = "modelspace",
    ) -> FilterResult:
        """
        Filter entities in a DXF document.

        Args:
            dxf_doc: ezdxf document
            space: Space to filter ("modelspace" or layout name)

        Returns:
            FilterResult
        """
        # Load layer states
        self._load_layer_states(dxf_doc)

        # Get entity source
        if space == "modelspace":
            entities = dxf_doc.modelspace()
        else:
            try:
                layout = dxf_doc.layouts.get(space)
                entities = layout if layout else []
            except Exception:
                entities = []

        # Filter entities
        result = FilterResult()

        for entity in entities:
            result.total_scanned += 1

            if self._matches(entity):
                result.entities.append(entity)
                result.total_matched += 1

                # Track statistics
                layer = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"
                etype = entity.dxftype()

                result.by_layer[layer] = result.by_layer.get(layer, 0) + 1
                result.by_type[etype] = result.by_type.get(etype, 0) + 1

        logger.debug(f"Filtered {result.total_matched}/{result.total_scanned} entities")
        return result

    def _load_layer_states(self, dxf_doc: Any) -> None:
        """Load layer visibility states."""
        self._layer_states.clear()

        if hasattr(dxf_doc, "layers"):
            for layer in dxf_doc.layers:
                self._layer_states[layer.dxf.name] = {
                    "is_on": layer.is_on(),
                    "is_frozen": layer.is_frozen(),
                    "is_locked": layer.is_locked(),
                }

    def _matches(self, entity: Any) -> bool:
        """Check if entity matches filter criteria."""
        # Get entity properties
        layer = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"
        etype = entity.dxftype()

        # Check layer visibility
        if not self._config.include_frozen_layers:
            layer_state = self._layer_states.get(layer, {})
            if layer_state.get("is_frozen", False):
                return False

        if not self._config.include_invisible:
            layer_state = self._layer_states.get(layer, {})
            if not layer_state.get("is_on", True):
                return False

        # Check layer filters
        if self._config.layers is not None:
            if layer not in self._config.layers:
                return False

        if self._config.layer_pattern is not None:
            if not re.search(self._config.layer_pattern, layer, re.IGNORECASE):
                return False

        if self._config.exclude_layers is not None:
            if layer in self._config.exclude_layers:
                return False

        # Check entity type filters
        if self._config.entity_types is not None:
            # Expand type groups
            types = set()
            for t in self._config.entity_types:
                if t.lower() in self.TYPE_GROUPS:
                    types.update(self.TYPE_GROUPS[t.lower()])
                else:
                    types.add(t.upper())

            if etype not in types:
                return False

        if self._config.exclude_types is not None:
            exclude = set()
            for t in self._config.exclude_types:
                if t.lower() in self.TYPE_GROUPS:
                    exclude.update(self.TYPE_GROUPS[t.lower()])
                else:
                    exclude.add(t.upper())

            if etype in exclude:
                return False

        # Check property filters
        if self._config.color is not None:
            entity_color = getattr(entity.dxf, "color", None)
            if entity_color != self._config.color:
                return False

        if self._config.linetype is not None:
            entity_lt = getattr(entity.dxf, "linetype", "").upper()
            if entity_lt != self._config.linetype.upper():
                return False

        # Check spatial filters
        if self._has_spatial_filter():
            if not self._check_spatial(entity):
                return False

        # Check block filter
        if self._config.in_block is not None:
            # This would require tracking block context
            pass

        # Check custom filter
        if self._config.custom_filter is not None:
            if not self._config.custom_filter(entity):
                return False

        return True

    def _has_spatial_filter(self) -> bool:
        """Check if any spatial filter is set."""
        return any([
            self._config.min_x is not None,
            self._config.max_x is not None,
            self._config.min_y is not None,
            self._config.max_y is not None,
        ])

    def _check_spatial(self, entity: Any) -> bool:
        """Check if entity is within spatial bounds."""
        try:
            # Get bounding box
            if hasattr(entity, "bounding_box"):
                bbox = entity.bounding_box()
                if bbox:
                    min_pt, max_pt = bbox
                    min_x, min_y = min_pt.x, min_pt.y
                    max_x, max_y = max_pt.x, max_pt.y
                else:
                    return True  # Can't determine, include it
            else:
                # Try to get representative point
                if hasattr(entity.dxf, "insert"):
                    pt = entity.dxf.insert
                    min_x = max_x = pt.x
                    min_y = max_y = pt.y
                elif hasattr(entity.dxf, "start"):
                    pt = entity.dxf.start
                    min_x = max_x = pt.x
                    min_y = max_y = pt.y
                elif hasattr(entity.dxf, "center"):
                    pt = entity.dxf.center
                    min_x = max_x = pt.x
                    min_y = max_y = pt.y
                else:
                    return True  # Can't determine, include it

            # Check bounds
            if self._config.min_x is not None and max_x < self._config.min_x:
                return False
            if self._config.max_x is not None and min_x > self._config.max_x:
                return False
            if self._config.min_y is not None and max_y < self._config.min_y:
                return False
            if self._config.max_y is not None and min_y > self._config.max_y:
                return False

            return True

        except Exception:
            return True  # Can't determine, include it

    def iter_filtered(
        self,
        dxf_doc: Any,
        space: str = "modelspace",
    ) -> Iterator[Any]:
        """
        Iterate over filtered entities.

        Args:
            dxf_doc: ezdxf document
            space: Space to filter

        Yields:
            Matching entities
        """
        self._load_layer_states(dxf_doc)

        if space == "modelspace":
            entities = dxf_doc.modelspace()
        else:
            try:
                layout = dxf_doc.layouts.get(space)
                entities = layout if layout else []
            except Exception:
                entities = []

        for entity in entities:
            if self._matches(entity):
                yield entity


def filter_entities(
    dxf_source: Union[str, Path, Any],
    config: Optional[FilterConfig] = None,
    **kwargs,
) -> FilterResult:
    """
    Convenience function to filter DXF entities.

    Args:
        dxf_source: DXF file path or ezdxf document
        config: Filter configuration
        **kwargs: Additional filter options

    Returns:
        FilterResult
    """
    try:
        import ezdxf
    except ImportError:
        raise ImportError("ezdxf required: pip install ezdxf")

    if isinstance(dxf_source, (str, Path)):
        doc = ezdxf.readfile(str(dxf_source))
    else:
        doc = dxf_source

    if config is None:
        config = FilterConfig(**kwargs)

    filter_obj = EntityFilter(config)
    return filter_obj.filter(doc)


def filter_by_layer(
    dxf_source: Union[str, Path, Any],
    layers: List[str],
    exclude: bool = False,
) -> FilterResult:
    """
    Filter entities by layer.

    Args:
        dxf_source: DXF file path or ezdxf document
        layers: Layer names
        exclude: If True, exclude these layers

    Returns:
        FilterResult
    """
    if exclude:
        config = FilterConfig(exclude_layers=layers)
    else:
        config = FilterConfig(layers=layers)

    return filter_entities(dxf_source, config)


def filter_by_type(
    dxf_source: Union[str, Path, Any],
    entity_types: List[str],
    exclude: bool = False,
) -> FilterResult:
    """
    Filter entities by type.

    Args:
        dxf_source: DXF file path or ezdxf document
        entity_types: Entity types (can use groups like "geometry", "text")
        exclude: If True, exclude these types

    Returns:
        FilterResult
    """
    if exclude:
        config = FilterConfig(exclude_types=entity_types)
    else:
        config = FilterConfig(entity_types=entity_types)

    return filter_entities(dxf_source, config)
