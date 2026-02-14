from __future__ import annotations

import ezdxf

from src.ml.importance_sampler import ImportanceSampler


def test_importance_sampler_caps_frame_entities(monkeypatch) -> None:
    # Keep the sample small so we can assert caps deterministically.
    monkeypatch.setenv("DXF_MAX_NODES", "10")
    monkeypatch.setenv("DXF_SAMPLING_STRATEGY", "importance")
    monkeypatch.setenv("DXF_SAMPLING_SEED", "42")
    monkeypatch.setenv("DXF_TEXT_PRIORITY_RATIO", "0.0")
    monkeypatch.setenv("DXF_FRAME_PRIORITY_RATIO", "0.2")  # => max_frame=2

    doc = ezdxf.new()
    msp = doc.modelspace()

    # Frame-like entities: many title-block lines (right-bottom region).
    for i in range(20):
        y = 5.0 + i * 0.5  # <= 40.0 for i < 70, so always in title block
        msp.add_line((70.0, y), (80.0, y))

    # Non-frame entities: circles far from title block.
    for i in range(20):
        msp.add_circle((10.0 + float(i), 90.0), radius=1.0)

    entities = list(msp)
    sampler = ImportanceSampler()
    result = sampler.sample(entities, bbox=(0.0, 0.0, 100.0, 100.0))

    assert result.sampled_count == 10
    assert result.stats["frame_count"] <= 2
    assert result.stats["title_block_count"] <= 2
    dtypes = [e.dxftype() for e in result.sampled_entities]
    assert "CIRCLE" in dtypes

