from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .verifier import PrecisionVerifier

_FILE_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GeomJsonStoreConfig:
    base_dir: Path = Path("data/dedup_geom")
    file_suffix: str = ".v2.json"

    @classmethod
    def from_env(cls) -> "GeomJsonStoreConfig":
        return cls(
            base_dir=Path(os.getenv("DEDUPCAD_GEOM_STORE_DIR", str(cls.base_dir))),
            file_suffix=os.getenv("DEDUPCAD_GEOM_STORE_SUFFIX", cls.file_suffix),
        )


class GeomJsonStore:
    """Filesystem-backed store for v2 geometry JSON keyed by `file_hash`."""

    def __init__(self, config: Optional[GeomJsonStoreConfig] = None) -> None:
        self.config = config or GeomJsonStoreConfig.from_env()
        self.config.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for_hash(self, file_hash: str) -> Path:
        if not _FILE_HASH_RE.match(file_hash):
            raise ValueError("Invalid file_hash; expected 64-char lowercase hex sha256")
        return self.config.base_dir / f"{file_hash}{self.config.file_suffix}"

    def exists(self, file_hash: str) -> bool:
        return self.path_for_hash(file_hash).exists()

    def load(self, file_hash: str) -> Optional[Dict[str, Any]]:
        path = self.path_for_hash(file_hash)
        if not path.exists():
            return None
        return PrecisionVerifier.load_json_bytes(path.read_bytes())

    def save(self, file_hash: str, geom_json: Dict[str, Any]) -> Path:
        path = self.path_for_hash(file_hash)
        content = PrecisionVerifier.canonical_json_bytes(geom_json)

        tmp_fd: Optional[int] = None
        tmp_path: Optional[str] = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), prefix=f".{file_hash}.", suffix=".tmp"
            )
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(content)
            Path(tmp_path).replace(path)
        finally:
            if tmp_path is not None and Path(tmp_path).exists():
                Path(tmp_path).unlink(missing_ok=True)
        return path

