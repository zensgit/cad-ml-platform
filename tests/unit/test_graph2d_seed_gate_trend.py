from __future__ import annotations


def test_graph2d_seed_gate_trend_contains_rows_and_bars() -> None:
    from scripts.ci.render_graph2d_seed_gate_trend import build_trend_markdown

    summary = {
        "strict_accuracy_mean": 0.36,
        "strict_top_pred_ratio_mean": 0.60,
        "strict_top_pred_ratio_max": 0.71,
        "strict_low_conf_threshold": 0.2,
        "strict_low_conf_ratio_mean": 0.05,
        "strict_low_conf_ratio_max": 0.09,
    }
    rows = [
        {
            "seed": 21,
            "status": "ok",
            "strict_accuracy": 0.29,
            "strict_top_pred_ratio": 0.71,
            "strict_low_conf_ratio": 0.09,
        },
        {
            "seed": 7,
            "status": "ok",
            "strict_accuracy": 0.43,
            "strict_top_pred_ratio": 0.49,
            "strict_low_conf_ratio": 0.05,
        },
    ]
    text = build_trend_markdown(
        summary=summary, rows=rows, title="Graph2D Seed Gate (CI Tests)"
    )
    assert "Graph2D Seed Gate (CI Tests) Trend" in text
    assert "| Seed | Status | Strict acc | Top-pred ratio | Low-conf ratio |" in text
    assert "`[########........] / [#...............]`" in text
    assert "`strict_top_pred_ratio_mean/max=0.600000/0.710000`" in text
    assert "`strict_low_conf_ratio_mean/max=0.050000/0.090000`" in text

