from __future__ import annotations

from src.core.dedupcad_precision.verifier import PrecisionVerifier


def test_precision_verifier_self_match_returns_one():
    """Self-match should be perfect even with polyline/dimension/hatch entities."""
    v2 = {
        "layers": {"0": {"color": 7, "linetype": "CONTINUOUS"}},
        "entities": [
            {"type": "LINE", "layer": "0", "start": [0.0, 0.0], "end": [100.0, 0.0]},
            {
                "type": "ARC",
                "layer": "0",
                "center": [10.0, 20.0],
                "radius": 5.0,
                "start_angle": 0.0,
                "end_angle": 90.0,
            },
            {
                "type": "LWPOLYLINE",
                "layer": "0",
                "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
                "closed": False,
            },
            # value=None is common when DXF dimension text is "<>" and parsing fails;
            # self-match must not be penalized.
            {
                "type": "DIMENSION",
                "layer": "0",
                "text": "<>",
                "dimstyle": "STANDARD",
                "value": None,
                "tol": None,
                "unit": "mm",
            },
            {"type": "HATCH", "layer": "0", "pattern": "ANSI31", "color": 1, "loops": 2},
        ],
        # Optional sections should also be stable for self-match.
        "dimensions": [
            {
                "dimension_type": "linear",
                "measurement_text": "10.00",
                "actual_measurement": 10.0,
                "override_text": None,
                "points": [[0.0, 0.0], [10.0, 0.0]],
                "text_midpoint": [5.0, 1.0],
                "text_rotation": 0.0,
                "layer": "0",
                "style": "STANDARD",
                "bbox": {"min_x": 0.0, "min_y": 0.0, "max_x": 10.0, "max_y": 1.0},
                "text_matches_value": True,
            }
        ],
        "hatches": [
            {
                "hatch_type": "pattern",
                "pattern_name": "ANSI31",
                "pattern_scale": 1.0,
                "pattern_angle": 0.0,
                "boundary_paths": [{"type": "external", "vertices_count": 4, "is_polyline": True}],
                "area": 2.0,
                "islands_count": 0,
                "layer": "0",
                "color": 256,
                "fill_color": None,
            }
        ],
        "blocks": {},
    }

    res = PrecisionVerifier().score_pair(v2, v2)
    assert res.score == 1.0
    assert res.breakdown["entities"] == 1.0
    assert res.breakdown["layers"] == 1.0


def test_version_profile_entities_spatial_signature_reduces_false_positive():
    """Version profile should not rely on pure bag-of-features when layouts differ."""
    def square_lines(x0: float, y0: float, *, size: float = 1.0, segments: int = 4):
        seg = max(1, int(segments))
        step = size / seg
        lines = []
        # bottom
        for i in range(seg):
            lines.append(
                {"type": "LINE", "layer": "0", "start": [x0 + i * step, y0], "end": [x0 + (i + 1) * step, y0]}
            )
        # right
        for i in range(seg):
            lines.append(
                {
                    "type": "LINE",
                    "layer": "0",
                    "start": [x0 + size, y0 + i * step],
                    "end": [x0 + size, y0 + (i + 1) * step],
                }
            )
        # top
        for i in range(seg):
            lines.append(
                {
                    "type": "LINE",
                    "layer": "0",
                    "start": [x0 + size - i * step, y0 + size],
                    "end": [x0 + size - (i + 1) * step, y0 + size],
                }
            )
        # left
        for i in range(seg):
            lines.append(
                {
                    "type": "LINE",
                    "layer": "0",
                    "start": [x0, y0 + size - i * step],
                    "end": [x0, y0 + size - (i + 1) * step],
                }
            )
        return lines

    # Two drawings with identical line-length/angle distributions but different spatial layouts.
    # Layout A: two unit squares side-by-side (32 line entities).
    left = {
        "layers": {"0": {"color": 7, "linetype": "CONTINUOUS"}},
        "entities": square_lines(0.0, 0.0) + square_lines(3.0, 0.0),
        "dimensions": [],
        "hatches": [],
        "blocks": {},
    }

    # Layout B: two unit squares stacked vertically (32 line entities).
    right = {
        "layers": {"0": {"color": 7, "linetype": "CONTINUOUS"}},
        "entities": square_lines(0.0, 0.0) + square_lines(0.0, 3.0),
        "dimensions": [],
        "hatches": [],
        "blocks": {},
    }

    res = PrecisionVerifier().score_pair(left, right, profile="version")
    # Entities similarity should be significantly below perfect match due to spatial mismatch.
    assert res.breakdown["entities"] < 0.95
