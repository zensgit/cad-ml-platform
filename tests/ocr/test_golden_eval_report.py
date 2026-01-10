"""Smoke test: run golden evaluation script and check report is created.

This does not gate on thresholds here; CI can run the script directly.
"""

import os
import subprocess
import sys
from pathlib import Path


def test_run_golden_evaluation_generates_report(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "tests" / "ocr" / "golden" / "run_golden_evaluation.py"
    assert script.exists()

    report_path = tmp_path / "ocr_evaluation.md"
    calibration_path = tmp_path / "ocr_calibration.md"
    env = dict(os.environ)
    env["OCR_GOLDEN_EVALUATION_REPORT_PATH"] = str(report_path)
    env["OCR_GOLDEN_CALIBRATION_REPORT_PATH"] = str(calibration_path)

    # Run the script in a subprocess to avoid event loop conflicts
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env)
    # Script may exit with non-zero if thresholds not met; allow both for smoke
    assert proc.stdout.strip() != ""
    # Report should be generated regardless
    assert report_path.exists()
    assert calibration_path.exists()
