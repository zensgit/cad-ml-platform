from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def faiss_recovery_state_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Ensure faiss recovery state does not persist across integration tests."""
    recovery_path = tmp_path / "faiss_recovery_state.json"
    monkeypatch.setenv("FAISS_RECOVERY_STATE_PATH", str(recovery_path))
    yield
