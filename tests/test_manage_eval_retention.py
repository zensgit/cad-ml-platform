from pathlib import Path
import json
from datetime import datetime, timedelta, timezone

from scripts import manage_eval_retention as mer


def _write_history(dirpath: Path, ts: datetime, kind: str = "combined", branch: str = "main") -> Path:
    fname = ts.strftime("%Y%m%d_%H%M%S") + f"_{kind}.json"
    path = dirpath / fname
    data = {
        "type": kind,
        "branch": branch,
        "commit": "deadbeef",
    }
    dirpath.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def test_retention_keeps_latest_per_period(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hist_dir = Path("reports/eval_history")

    # Create files across two months
    now = datetime.now(timezone.utc)
    last_month = now - timedelta(days=35)

    # Multiple files in last month and this month
    for d in [0, 1, 2, 10, 20, 30]:
        _write_history(hist_dir, last_month + timedelta(days=d))
    for d in [0, 2, 3, 5]:
        _write_history(hist_dir, now - timedelta(days=d))

    # Load EvaluationFile list
    files = [
        mer.EvaluationFile(p)
        for p in sorted(hist_dir.glob("*_combined.json"))
    ]

    keep, delete = mer.apply_retention_policy(files, dry_run=True, verbose=False)

    # Should keep at least one per relevant periods (daily/weekly/monthly/quarterly)
    assert keep, "Expected some files to be kept"
    assert len(keep) + len(delete) == len(files)

