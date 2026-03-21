"""Smoke tests for the experiment CLI scripts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_single_game_and_evaluate_logs_scripts_run(tmp_path: Path) -> None:
    """The CLI scripts should run a baseline scenario and analyze its log."""

    repo_root = Path(__file__).resolve().parents[1]
    log_path = tmp_path / "single_game.jsonl"

    run_result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_single_game.py"),
            "--scenario",
            "s1_open_encounter",
            "--blue-agent",
            "rule",
            "--red-agent",
            "rule",
            "--output",
            str(log_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    run_payload = json.loads(run_result.stdout)
    assert log_path.exists()
    assert run_payload["log_path"] == str(log_path)
    assert run_payload["turns"] > 0

    eval_result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "evaluate_logs.py"),
            str(log_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    eval_payload = json.loads(eval_result.stdout)
    assert eval_payload["run_count"] == 1
    assert "mean_action_entropy" in eval_payload
