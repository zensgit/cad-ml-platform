import json
import logging
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional, Union

logger = logging.getLogger(__name__)


def _iter_rule_files(path: Path) -> Iterable[Path]:
    return sorted(path.glob("*_rules.json"))


def _has_rules(path: Path) -> bool:
    for rule_file in _iter_rule_files(path):
        try:
            data = json.loads(rule_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        rules = data.get("rules", [])
        if isinstance(rules, list) and rules:
            return True
    return False


def seed_knowledge_if_empty(
    seed_dir: Optional[Union[str, Path]] = None,
    target_dir: Optional[Union[str, Path]] = None,
) -> bool:
    if os.getenv("KNOWLEDGE_AUTO_SEED", "0") != "1":
        return False

    seed_path = Path(seed_dir or os.getenv("KNOWLEDGE_SEED_DIR", "seed/knowledge"))
    target_path = Path(target_dir or os.getenv("KNOWLEDGE_DATA_DIR", "data/knowledge"))
    target_path.mkdir(parents=True, exist_ok=True)

    if _has_rules(target_path):
        return False

    if not seed_path.exists():
        logger.warning("Knowledge seed dir not found: %s", seed_path)
        return False

    copied = 0
    for rule_file in _iter_rule_files(seed_path):
        dest = target_path / rule_file.name
        shutil.copy2(rule_file, dest)
        copied += 1

    if copied:
        logger.info("Seeded knowledge rules: %s file(s) into %s", copied, target_path)
    return copied > 0
