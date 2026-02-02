"""
Artifact storage for experiment tracking.

Provides:
- Model artifact storage
- File copying and management
- Artifact metadata tracking
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ArtifactType(str, Enum):
    """Type of artifact."""
    MODEL = "model"
    CHECKPOINT = "checkpoint"
    DATA = "data"
    PLOT = "plot"
    CONFIG = "config"
    LOG = "log"
    OTHER = "other"


@dataclass
class Artifact:
    """An artifact stored in the experiment."""
    artifact_id: str
    name: str
    artifact_type: ArtifactType
    path: str
    size_bytes: int
    checksum: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "artifact_type": self.artifact_type.value,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """Create from dictionary."""
        return cls(
            artifact_id=data["artifact_id"],
            name=data["name"],
            artifact_type=ArtifactType(data["artifact_type"]),
            path=data["path"],
            size_bytes=data["size_bytes"],
            checksum=data.get("checksum"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


class ArtifactStore:
    """
    Storage for experiment artifacts.

    Supports:
    - File copying and storage
    - Checksum verification
    - Metadata tracking
    - Artifact retrieval
    """

    def __init__(self, base_path: Path):
        """
        Initialize artifact store.

        Args:
            base_path: Base directory for artifact storage
        """
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._artifacts: Dict[str, Artifact] = {}
        self._manifest_path = self._base_path / "artifacts.json"

        if self._manifest_path.exists():
            self._load_manifest()

    @property
    def artifact_count(self) -> int:
        """Get number of artifacts."""
        return len(self._artifacts)

    def _generate_id(self, name: str) -> str:
        """Generate artifact ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{name}_{timestamp}"

    def _compute_checksum(self, path: Path) -> str:
        """Compute MD5 checksum of file."""
        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def _infer_type(self, path: Path) -> ArtifactType:
        """Infer artifact type from file extension."""
        suffix = path.suffix.lower()
        type_map = {
            ".pth": ArtifactType.MODEL,
            ".pt": ArtifactType.MODEL,
            ".ckpt": ArtifactType.CHECKPOINT,
            ".h5": ArtifactType.MODEL,
            ".pkl": ArtifactType.MODEL,
            ".png": ArtifactType.PLOT,
            ".jpg": ArtifactType.PLOT,
            ".svg": ArtifactType.PLOT,
            ".json": ArtifactType.CONFIG,
            ".yaml": ArtifactType.CONFIG,
            ".yml": ArtifactType.CONFIG,
            ".csv": ArtifactType.DATA,
            ".parquet": ArtifactType.DATA,
            ".log": ArtifactType.LOG,
            ".txt": ArtifactType.LOG,
        }
        return type_map.get(suffix, ArtifactType.OTHER)

    def store(
        self,
        source_path: str,
        name: Optional[str] = None,
        artifact_type: Optional[ArtifactType] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compute_checksum: bool = True,
    ) -> Artifact:
        """
        Store an artifact.

        Args:
            source_path: Path to source file
            name: Artifact name (defaults to filename)
            artifact_type: Type of artifact (auto-detected if not provided)
            metadata: Additional metadata
            compute_checksum: Whether to compute checksum

        Returns:
            Artifact object
        """
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        name = name or src.name
        artifact_id = self._generate_id(name)
        artifact_type = artifact_type or self._infer_type(src)

        # Copy file to store
        dst = self._base_path / artifact_id / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        # Get file info
        size_bytes = dst.stat().st_size
        checksum = self._compute_checksum(dst) if compute_checksum else None

        artifact = Artifact(
            artifact_id=artifact_id,
            name=name,
            artifact_type=artifact_type,
            path=str(dst),
            size_bytes=size_bytes,
            checksum=checksum,
            metadata=metadata or {},
        )

        self._artifacts[artifact_id] = artifact
        self._save_manifest()

        logger.info(f"Stored artifact: {name} ({size_bytes} bytes)")
        return artifact

    def get(self, artifact_id: str) -> Optional[Artifact]:
        """
        Get artifact by ID.

        Args:
            artifact_id: Artifact ID

        Returns:
            Artifact or None
        """
        return self._artifacts.get(artifact_id)

    def get_by_name(self, name: str) -> List[Artifact]:
        """
        Get artifacts by name.

        Args:
            name: Artifact name

        Returns:
            List of matching artifacts
        """
        return [a for a in self._artifacts.values() if a.name == name]

    def get_by_type(self, artifact_type: ArtifactType) -> List[Artifact]:
        """
        Get artifacts by type.

        Args:
            artifact_type: Artifact type

        Returns:
            List of matching artifacts
        """
        return [a for a in self._artifacts.values() if a.artifact_type == artifact_type]

    def get_latest(self, name: str) -> Optional[Artifact]:
        """
        Get latest artifact with given name.

        Args:
            name: Artifact name

        Returns:
            Latest artifact or None
        """
        matching = self.get_by_name(name)
        if not matching:
            return None
        return max(matching, key=lambda a: a.created_at)

    def list_all(self) -> List[Artifact]:
        """
        List all artifacts.

        Returns:
            List of all artifacts
        """
        return list(self._artifacts.values())

    def delete(self, artifact_id: str) -> bool:
        """
        Delete an artifact.

        Args:
            artifact_id: Artifact ID

        Returns:
            True if deleted, False if not found
        """
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            return False

        # Delete file
        artifact_path = Path(artifact.path)
        if artifact_path.exists():
            artifact_path.unlink()
            # Remove parent directory if empty
            if artifact_path.parent.exists() and not any(artifact_path.parent.iterdir()):
                artifact_path.parent.rmdir()

        del self._artifacts[artifact_id]
        self._save_manifest()

        logger.info(f"Deleted artifact: {artifact_id}")
        return True

    def verify(self, artifact_id: str) -> bool:
        """
        Verify artifact integrity.

        Args:
            artifact_id: Artifact ID

        Returns:
            True if valid, False otherwise
        """
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            return False

        path = Path(artifact.path)
        if not path.exists():
            return False

        if artifact.checksum:
            current_checksum = self._compute_checksum(path)
            return current_checksum == artifact.checksum

        return True

    def export(self, artifact_id: str, destination: str) -> str:
        """
        Export artifact to destination.

        Args:
            artifact_id: Artifact ID
            destination: Destination path

        Returns:
            Exported file path
        """
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            raise ValueError(f"Artifact not found: {artifact_id}")

        src = Path(artifact.path)
        dst = Path(destination)

        if dst.is_dir():
            dst = dst / artifact.name

        shutil.copy2(src, dst)
        logger.info(f"Exported artifact to: {dst}")
        return str(dst)

    def _save_manifest(self) -> None:
        """Save artifact manifest."""
        data = {
            artifact_id: artifact.to_dict()
            for artifact_id, artifact in self._artifacts.items()
        }
        with open(self._manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_manifest(self) -> None:
        """Load artifact manifest."""
        with open(self._manifest_path) as f:
            data = json.load(f)

        self._artifacts = {
            artifact_id: Artifact.from_dict(artifact_data)
            for artifact_id, artifact_data in data.items()
        }

    def get_total_size(self) -> int:
        """Get total size of all artifacts in bytes."""
        return sum(a.size_bytes for a in self._artifacts.values())

    def __len__(self) -> int:
        return len(self._artifacts)

    def __repr__(self) -> str:
        return f"ArtifactStore(base_path={self._base_path}, count={len(self)})"
