from __future__ import annotations

import json
from pathlib import Path


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    lines.extend(",".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _success_fixture(tmp_path: Path) -> dict[str, Path]:
    golden_train = tmp_path / "data" / "manifests" / "golden_train_set.csv"
    golden_val = tmp_path / "data" / "manifests" / "golden_val_set.csv"
    _write_csv(
        golden_train,
        ["file_path", "cache_path", "taxonomy_v2_class"],
        [["/train/a.dxf", "/cache/a.pt", "A"]],
    )
    _write_csv(
        golden_val,
        ["file_path", "cache_path", "taxonomy_v2_class"],
        [["/val/b.dxf", "/cache/b.pt", "B"]],
    )

    auto_retrain = tmp_path / "scripts" / "auto_retrain.sh"
    _write_text(
        auto_retrain,
        """
human_verified
eligible_for_training
--val-manifest
backfill_manifest_cache_paths.py
Backfilling cache_path into manifest
missing cache_path after backfill
exit 1
""".strip()
        + "\n",
    )

    append_reviewed = tmp_path / "scripts" / "append_reviewed_to_manifest.py"
    _write_text(
        append_reviewed,
        """
human_verified
--include-unverified
cache_path
will need preprocess
""".strip()
        + "\n",
    )

    finetune_augmented = tmp_path / "scripts" / "finetune_graph2d_v2_augmented.py"
    _write_text(
        finetune_augmented,
        """
val_paths = set()
train_indices = [i for i, (cp, _) in enumerate(ds.samples) if cp not in val_paths]
print("Leakage prevention: removed")
""".strip()
        + "\n",
    )

    finetune_pretrained = tmp_path / "scripts" / "finetune_graph2d_from_pretrained.py"
    _write_text(
        finetune_pretrained,
        """
val_paths = set()
train_indices = [i for i, (cp, _) in enumerate(dataset.samples) if cp not in val_paths]
logger.info("Leakage prevention: removed")
""".strip()
        + "\n",
    )

    active_learning_api = tmp_path / "src" / "api" / "v1" / "active_learning.py"
    _write_text(
        active_learning_api,
        'label_source: Optional[str] = Field(default="human_feedback")\n',
    )

    active_learning_core = tmp_path / "src" / "core" / "active_learning.py"
    _write_text(
        active_learning_core,
        """
eligible_for_training = True
eligible_count = 1
human_feedback
""".strip()
        + "\n",
    )

    backfill_helper = tmp_path / "scripts" / "backfill_manifest_cache_paths.py"
    _write_text(
        backfill_helper,
        """
cache_manifest.csv
missing cache_path after backfill
hashlib.md5
""".strip()
        + "\n",
    )

    return {
        "golden_train": golden_train,
        "golden_val": golden_val,
        "auto_retrain": auto_retrain,
        "append_reviewed": append_reviewed,
        "finetune_augmented": finetune_augmented,
        "finetune_pretrained": finetune_pretrained,
        "active_learning_api": active_learning_api,
        "active_learning_core": active_learning_core,
        "backfill_helper": backfill_helper,
    }


def test_check_training_data_governance_success(tmp_path: Path) -> None:
    from scripts.ci.check_training_data_governance import check_training_data_governance

    paths = _success_fixture(tmp_path)
    report = check_training_data_governance(
        golden_train_manifest=paths["golden_train"],
        golden_val_manifest=paths["golden_val"],
        auto_retrain_script=paths["auto_retrain"],
        append_reviewed_script=paths["append_reviewed"],
        finetune_augmented_script=paths["finetune_augmented"],
        finetune_pretrained_script=paths["finetune_pretrained"],
        active_learning_api=paths["active_learning_api"],
        active_learning_core=paths["active_learning_core"],
        backfill_helper=paths["backfill_helper"],
    )
    assert report["status"] == "ok"
    assert report["violations_count"] == 0


def test_check_training_data_governance_detects_manifest_overlap(tmp_path: Path) -> None:
    from scripts.ci.check_training_data_governance import check_training_data_governance

    paths = _success_fixture(tmp_path)
    _write_csv(
        paths["golden_val"],
        ["file_path", "cache_path", "taxonomy_v2_class"],
        [["/train/a.dxf", "/cache/a.pt", "A"]],
    )
    report = check_training_data_governance(
        golden_train_manifest=paths["golden_train"],
        golden_val_manifest=paths["golden_val"],
        auto_retrain_script=paths["auto_retrain"],
        append_reviewed_script=paths["append_reviewed"],
        finetune_augmented_script=paths["finetune_augmented"],
        finetune_pretrained_script=paths["finetune_pretrained"],
        active_learning_api=paths["active_learning_api"],
        active_learning_core=paths["active_learning_core"],
        backfill_helper=paths["backfill_helper"],
    )
    assert report["status"] == "error"
    assert any(item["label"] == "golden_train_val_overlap" for item in report["violations"])


def test_main_writes_output_json_and_fails_for_missing_fail_closed_guard(
    tmp_path: Path,
) -> None:
    from scripts.ci import check_training_data_governance as mod

    paths = _success_fixture(tmp_path)
    _write_text(
        paths["auto_retrain"],
        """
human_verified
eligible_for_training
--val-manifest
Backfilling cache_path into manifest
missing cache_path after backfill
""".strip()
        + "\n",
    )
    output_json = tmp_path / "report.json"

    rc = mod.main(
        [
            "--golden-train-manifest",
            str(paths["golden_train"]),
            "--golden-val-manifest",
            str(paths["golden_val"]),
            "--auto-retrain-script",
            str(paths["auto_retrain"]),
            "--append-reviewed-script",
            str(paths["append_reviewed"]),
            "--finetune-augmented-script",
            str(paths["finetune_augmented"]),
            "--finetune-pretrained-script",
            str(paths["finetune_pretrained"]),
            "--active-learning-api",
            str(paths["active_learning_api"]),
            "--active-learning-core",
            str(paths["active_learning_core"]),
            "--backfill-helper",
            str(paths["backfill_helper"]),
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 1
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert any(
        item["label"] == "auto_retrain_script"
        and "exit 1" in item["missing_tokens"]
        for item in payload["violations"]
    )
