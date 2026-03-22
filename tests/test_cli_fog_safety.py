"""Tests for fog-of-war preset and warning behavior in CLI scripts."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

from wargame.experiments import MatchupSpec

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

run_single_game = importlib.import_module("run_single_game")
run_batch = importlib.import_module("run_batch")


def test_single_game_auto_baseline_preset_applies_safe_values() -> None:
    """Rule-vs-rule runs should auto-fill the safer baseline fog preset."""

    preset = run_single_game._select_single_game_fog_preset(
        blue_agent_spec="rule",
        red_agent_spec="random",
        user_preset="auto",
    )
    visibility, identification, messages = run_single_game._resolve_fog_settings(
        visibility_radius=3,
        identification_radius=1,
        explicit_visibility=False,
        explicit_identification=False,
        preset_name=preset,
        experiment_mode="baseline",
    )

    assert preset == "baseline"
    assert (visibility, identification) == (8, 3)
    assert any("applied fog preset 'baseline'" in message for message in messages)


def test_single_game_auto_llm_preset_applies_safe_values() -> None:
    """Runs with local_llm agents should auto-fill the established LLM preset."""

    preset = run_single_game._select_single_game_fog_preset(
        blue_agent_spec="local_llm:mlx-community/Qwen2.5-7B-Instruct-4bit",
        red_agent_spec="rule",
        user_preset="auto",
    )
    visibility, identification, messages = run_single_game._resolve_fog_settings(
        visibility_radius=3,
        identification_radius=1,
        explicit_visibility=False,
        explicit_identification=False,
        preset_name=preset,
        experiment_mode="llm",
    )

    assert preset == "llm"
    assert (visibility, identification) == (5, 2)
    assert any("applied fog preset 'llm'" in message for message in messages)


def test_batch_mixed_matchups_warn_and_choose_safe_baseline_preset() -> None:
    """Mixed batches should auto-select the safer baseline preset and warn."""

    preset, messages, experiment_mode = run_batch._select_batch_fog_preset(
        matchups=[
            MatchupSpec(
                name="qwen-vs-rule",
                blue_agent_name="local_llm:mlx-community/Qwen2.5-7B-Instruct-4bit",
                red_agent_name="rule",
            ),
            MatchupSpec(
                name="rule-vs-random",
                blue_agent_name="rule",
                red_agent_name="random",
            ),
        ],
        user_preset="auto",
    )

    assert preset == "baseline"
    assert experiment_mode == "baseline"
    assert any("mixed batch" in message for message in messages)


def test_single_game_cli_emits_warning_for_too_small_baseline_values(tmp_path: Path) -> None:
    """The CLI should warn on stderr when baseline runs use the known bad small fog values."""

    repo_root = Path(__file__).resolve().parents[1]
    log_path = tmp_path / "fog_warning.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_single_game.py"),
            "--scenario",
            "s1_open_encounter",
            "--blue-agent",
            "rule",
            "--red-agent",
            "rule",
            "--visibility-radius",
            "3",
            "--identification-radius",
            "1",
            "--output",
            str(log_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    payload = json.loads(result.stdout)
    assert payload["log_path"] == str(log_path)
    assert "baseline matchups below the recommended fog preset" in result.stderr
