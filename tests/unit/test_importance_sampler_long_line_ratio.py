from __future__ import annotations

import ezdxf

from src.ml.importance_sampler import ImportanceSampler


def test_importance_sampler_caps_non_frame_long_lines(monkeypatch) -> None:
    monkeypatch.setenv("DXF_MAX_NODES", "10")
    monkeypatch.setenv("DXF_SAMPLING_STRATEGY", "importance")
    monkeypatch.setenv("DXF_SAMPLING_SEED", "42")
    monkeypatch.setenv("DXF_TEXT_PRIORITY_RATIO", "0.0")
    monkeypatch.setenv("DXF_FRAME_PRIORITY_RATIO", "0.0")
    monkeypatch.setenv("DXF_LONG_LINE_RATIO", "0.3")  # => max_long_lines=3

    doc = ezdxf.new()
    msp = doc.modelspace()

    # 7 non-long entities (ensure we still need some long lines to fill to max_nodes).
    msp.add_circle((20.0, 80.0), radius=2.0)
    msp.add_circle((30.0, 80.0), radius=2.0)
    msp.add_arc((40.0, 70.0), radius=5.0, start_angle=0.0, end_angle=180.0)
    msp.add_arc((50.0, 70.0), radius=5.0, start_angle=0.0, end_angle=180.0)
    msp.add_line((20.0, 60.0), (28.0, 60.0))  # short line
    msp.add_line((20.0, 62.0), (28.0, 62.0))  # short line
    msp.add_line((20.0, 64.0), (28.0, 64.0))  # short line

    # Many long lines (non-frame): length >= 0.5 * max_dim where max_dim=100.
    for i in range(20):
        y = 60.0 + float(i) * 0.5
        msp.add_line((10.0, y), (90.0, y))

    entities = list(msp)
    sampler = ImportanceSampler()
    result = sampler.sample(entities, bbox=(0.0, 0.0, 100.0, 100.0))

    assert result.sampled_count == 10

    long_lines = 0
    dtypes = []
    for ent in result.sampled_entities:
        dtype = ent.dxftype()
        dtypes.append(dtype)
        if dtype != "LINE":
            continue
        start = ent.dxf.start
        end = ent.dxf.end
        length = float(start.distance(end))
        if length >= 50.0:
            long_lines += 1

    assert long_lines <= 3
    assert "CIRCLE" in dtypes

