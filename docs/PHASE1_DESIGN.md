# Phase 1: Core Rendering Engine Upgrade Design

## 1. SVG Rendering Support (Task 1.1)

### Objective
Implement a robust SVG rendering engine to support vector-based CAD formats (DXF, DWG) with high fidelity, enabling better feature extraction and visualization.

### Architecture

#### `src/core/svg_renderer.py`
A new module dedicated to SVG generation.

*   **Class `SVGRenderer`**:
    *   `__init__(self, config: Dict)`: Initialize with rendering configuration (stroke width, colors, etc.).
    *   `render(self, entity: Any) -> str`: Main entry point to render a CAD entity to an SVG string.
    *   `_render_line(self, line: Any) -> str`: Render a line entity.
    *   `_render_circle(self, circle: Any) -> str`: Render a circle entity.
    *   `_render_arc(self, arc: Any) -> str`: Render an arc entity.
    *   `_render_polyline(self, polyline: Any) -> str`: Render a polyline.
    *   `save(self, svg_content: str, path: str)`: Save SVG to file.

#### Integration with `src/core/renderer.py`
The existing `Renderer` class will be updated to use `SVGRenderer` when the output format is SVG or when vector-based feature extraction is required.

### Key Features
*   **Coordinate System Transformation**: Map CAD coordinates to SVG viewport coordinates.
*   **Style Management**: Configurable stroke width, color, and fill.
*   **Entity Support**: Basic primitives (Line, Circle, Arc, Polyline, Text).

## 2. Advanced Rasterization (Task 1.2)

### Objective
Improve rasterization quality using anti-aliasing and high-resolution rendering from the SVG source.

### Implementation
*   Use `cairosvg` or similar library (if available, otherwise fallback to custom rasterizer) to convert generated SVG to PNG/JPEG.
*   Implement `render_to_image(self, svg_content: str, format: str = 'png') -> bytes`.

## 3. Color-Coded Layer Rendering (Task 1.3)

### Objective
Render different CAD layers with distinct colors to aid in semantic segmentation and feature extraction.

### Implementation
*   Extend `SVGRenderer` to accept a layer-to-color mapping.
*   `_get_layer_color(self, layer_name: str) -> str`: Helper to resolve colors.

## Testing Strategy
*   **Unit Tests**: Verify SVG string generation for individual entities.
*   **Integration Tests**: Convert a sample DXF file to SVG and verify structure.
*   **Visual Inspection**: Manually check generated SVGs for correctness (during dev).
