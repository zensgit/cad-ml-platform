import json
from pathlib import Path

from src.utils.knowledge_seed import seed_knowledge_if_empty


def _write_seed_rule(path: Path) -> None:
    data = {
        "version": "2025-12-22T00:00:00Z",
        "category": "geometry",
        "count": 1,
        "updated_at": "2025-12-22T00:00:00Z",
        "rules": [
            {
                "id": "seed_geometry_round_part",
                "category": "geometry",
                "name": "Seed round part hint",
                "chinese_name": "",
                "description": "Seed rule for geometry matching",
                "keywords": ["round", "circular"],
                "ocr_patterns": [],
                "part_hints": {"round_part": 0.6},
                "enabled": True,
                "priority": 10,
                "source": "seed",
                "created_at": "2025-12-22T00:00:00Z",
                "updated_at": "2025-12-22T00:00:00Z",
                "metadata": {},
                "conditions": {"circle_ratio": {"min": 0.3}},
            }
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_seed_knowledge_if_empty(monkeypatch, tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    target_dir = tmp_path / "target"
    seed_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)

    seed_file = seed_dir / "geometry_rules.json"
    _write_seed_rule(seed_file)

    monkeypatch.setenv("KNOWLEDGE_AUTO_SEED", "1")

    seeded = seed_knowledge_if_empty(seed_dir=seed_dir, target_dir=target_dir)
    assert seeded is True
    assert (target_dir / "geometry_rules.json").exists()

    seeded_again = seed_knowledge_if_empty(seed_dir=seed_dir, target_dir=target_dir)
    assert seeded_again is False
