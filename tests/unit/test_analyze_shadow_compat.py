from __future__ import annotations

from src.api.v1 import analyze
from src.api.v1 import analyze_shadow_compat
from src.core.classification import shadow_pipeline


def test_analyze_shadow_compat_reexports_shadow_pipeline_helpers() -> None:
    assert (
        analyze_shadow_compat._build_graph2d_soft_override_suggestion
        is shadow_pipeline._build_graph2d_soft_override_suggestion
    )
    assert (
        analyze_shadow_compat._enrich_graph2d_prediction
        is shadow_pipeline._enrich_graph2d_prediction
    )
    assert (
        analyze_shadow_compat._graph2d_is_drawing_type
        is shadow_pipeline._graph2d_is_drawing_type
    )
    assert (
        analyze_shadow_compat._resolve_history_sequence_file_path
        is shadow_pipeline._resolve_history_sequence_file_path
    )


def test_analyze_reexports_shadow_compat_helpers() -> None:
    assert (
        analyze._build_graph2d_soft_override_suggestion
        is analyze_shadow_compat._build_graph2d_soft_override_suggestion
    )
    assert analyze._enrich_graph2d_prediction is analyze_shadow_compat._enrich_graph2d_prediction
    assert analyze._graph2d_is_drawing_type is analyze_shadow_compat._graph2d_is_drawing_type
    assert (
        analyze._resolve_history_sequence_file_path
        is analyze_shadow_compat._resolve_history_sequence_file_path
    )
