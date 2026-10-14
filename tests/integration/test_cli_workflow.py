"""Stage 10 -- runs this repo's real CLI script as a subprocess,
exactly as a researcher would from the command line.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_run_risk_analysis_cli_produces_a_real_report(tmp_path):
    output_dir = tmp_path / "reports"
    result = subprocess.run(
        [sys.executable, "scripts/run_risk_analysis.py", "--n-periods-months", "30",
         "--output-dir", str(output_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert (output_dir / "risk_report_benchmark_equal_weight.json").exists()
    assert (output_dir / "risk_report_benchmark_equal_weight.md").exists()
    assert "portfolio volatility" in result.stdout.lower()
