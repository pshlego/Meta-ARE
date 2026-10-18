"""prompts/app_cli_summary.md must match the app CLIs' click definitions (options included)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAIA2_CLI = ROOT.parent
SUMMARY = ROOT / "prompts" / "app_cli_summary.md"


def test_summary_is_generated_from_the_cli() -> None:
    env = {**os.environ, "PYTHONPATH": f"{GAIA2_CLI / 'cli'}{os.pathsep}{GAIA2_CLI / 'core'}"}
    result = subprocess.run(
        [sys.executable, "-m", "envharness_gaia2.app_cli_summary", "--check"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, "stale summary: rerun python -m envharness_gaia2.app_cli_summary\n" + result.stderr


def test_summary_lists_options() -> None:
    text = SUMMARY.read_text(encoding="utf-8")
    assert "  - `--start-datetime TEXT` — " in text
    assert "- `contacts get-contacts` — " in text
    contacts = text.split("- `contacts get-contacts` — ")[1].split("\n- ")[0]
    assert "--offset" in contacts and "--limit" not in contacts
    assert "oracle_function" not in text
