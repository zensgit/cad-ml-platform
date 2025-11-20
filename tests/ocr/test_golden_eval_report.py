"""Smoke test: run golden evaluation script and check report is created.

This does not gate on thresholds here; CI can run the script directly.
"""

from pathlib import Path
import subprocess
import sys


def test_run_golden_evaluation_generates_report(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "tests" / "ocr" / "golden" / "run_golden_evaluation.py"
    assert script.exists()
    # Run the script in a subprocess to avoid event loop conflicts
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    # Script may exit with non-zero if thresholds not met; allow both for smoke
    assert proc.stdout.strip() != ""
    # Report should be generated regardless
    report = repo_root / "reports" / "ocr_evaluation.md"
    assert report.exists()
