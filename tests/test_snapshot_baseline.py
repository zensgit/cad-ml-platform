import json
from pathlib import Path

from scripts import snapshot_baseline as sb


def test_snapshot_baseline_no_source_returns_nonzero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Ensure default source does not exist
    exit_code = sb.snapshot_baseline(force=False)
    assert exit_code == 1


def test_snapshot_baseline_creates_snapshot_and_latest_symlink(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Arrange: create a minimal baseline with sufficient samples
    source_dir = Path("reports/insights")
    source_dir.mkdir(parents=True, exist_ok=True)
    baseline = {
        "sample_count": 12,
        "updated_at": "2025-01-01T00:00:00Z",
        "metrics": {"combined": {"mean": 0.8, "stdev": 0.05, "history": [0.75, 0.8, 0.85]}},
    }
    with open(source_dir / "baseline.json", "w") as f:
        json.dump(baseline, f)

    # Act
    exit_code = sb.snapshot_baseline(force=False)

    # Assert
    assert exit_code == 0
    baselines_dir = Path("reports/baselines")
    snapshots = list(baselines_dir.glob("baseline_*.json"))
    assert snapshots, "Expected a quarterly snapshot file"

    latest = baselines_dir / "latest_snapshot.json"
    assert latest.exists()
    assert latest.is_symlink()
    # Symlink should point at one of the created snapshots by name
    assert latest.readlink().name == snapshots[0].name
