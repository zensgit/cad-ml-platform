from __future__ import annotations

from pathlib import Path

from scripts.sweep_graph2d_strict_mode import _extract_strict_metrics, _read_eval_overall


def test_read_eval_overall_extracts_overall_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "eval_metrics.csv"
    csv_path.write_text(
        "label_cn,total,correct,accuracy,precision,recall,f1,top2_accuracy,share,macro_f1,weighted_f1\n"
        "__overall__,80,10,0.125,,,,0.250,1.000,0.300,0.400\n",
        encoding="utf-8",
    )

    payload = _read_eval_overall(csv_path)
    assert payload["eval_samples"] == 80
    assert payload["eval_accuracy"] == 0.125
    assert payload["eval_top2_accuracy"] == 0.25
    assert payload["eval_macro_f1"] == 0.3
    assert payload["eval_weighted_f1"] == 0.4


def test_extract_strict_metrics_parses_confidence_and_top_pred() -> None:
    summary = {
        "accuracy": 0.175,
        "confidence": {"p50": 0.125, "p90": 0.2},
        "top_pred_labels_canon": [["label_a", 40], ["label_b", 10]],
    }

    payload = _extract_strict_metrics(summary)
    assert payload["strict_accuracy"] == 0.175
    assert payload["strict_conf_p50"] == 0.125
    assert payload["strict_conf_p90"] == 0.2
    assert payload["strict_top_pred_label"] == "label_a"
    assert payload["strict_top_pred_count"] == 40
