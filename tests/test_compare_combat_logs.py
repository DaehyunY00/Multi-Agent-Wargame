"""Tests for deterministic-vs-stochastic combat log comparison utilities."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from wargame.analysis import aggregate_casualties_by_unit, combat_turn_count


def test_combat_metrics_use_logged_casualties_only(tmp_path: Path) -> None:
    """Combat detection should come only from positive casualties_by_unit values."""

    log_path = tmp_path / "combat.jsonl"
    _write_jsonl(
        log_path,
        [
            _record(
                turn=1,
                casualties_by_unit={},
                blue_strength=100,
                red_strength=100,
            ),
            _record(
                turn=2,
                casualties_by_unit={"blue-1": 3, "red-1": 7},
                blue_strength=97,
                red_strength=93,
            ),
            _record(
                turn=3,
                casualties_by_unit={"blue-1": 0, "red-1": 0},
                blue_strength=97,
                red_strength=93,
            ),
        ],
    )

    assert combat_turn_count(log_path) == 1
    assert aggregate_casualties_by_unit(log_path) == {"blue-1": 3, "red-1": 7}


def test_compare_combat_logs_script_outputs_expected_delta(tmp_path: Path) -> None:
    """The comparison CLI should summarize deterministic and stochastic logs."""

    repo_root = Path(__file__).resolve().parents[1]
    deterministic_path = tmp_path / "deterministic.jsonl"
    stochastic_path = tmp_path / "stochastic.jsonl"
    _write_jsonl(
        deterministic_path,
        [
            _record(
                turn=1,
                casualties_by_unit={"blue-1": 2, "red-1": 4},
                blue_strength=98,
                red_strength=96,
            ),
            _record(
                turn=2,
                casualties_by_unit={},
                blue_strength=98,
                red_strength=96,
            ),
        ],
    )
    _write_jsonl(
        stochastic_path,
        [
            _record(
                turn=1,
                casualties_by_unit={"blue-1": 3, "red-1": 5},
                blue_strength=97,
                red_strength=95,
            ),
            _record(
                turn=2,
                casualties_by_unit={"blue-1": 1},
                blue_strength=96,
                red_strength=95,
            ),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "compare_combat_logs.py"),
            str(deterministic_path),
            str(stochastic_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    payload = json.loads(result.stdout)

    assert payload["deterministic"]["combat_turn_count"] == 1
    assert payload["stochastic"]["combat_turn_count"] == 2
    assert payload["delta"]["combat_turn_count"] == 1
    assert payload["delta"]["total_casualties"] == 3
    assert payload["delta"]["blue_final_strength"] == -2
    assert payload["delta"]["red_final_strength"] == -1
    assert payload["delta"]["casualty_delta_by_unit"] == {"blue-1": 2, "red-1": 1}


def _record(
    *,
    turn: int,
    casualties_by_unit: dict[str, int],
    blue_strength: int,
    red_strength: int,
) -> dict[str, object]:
    """Build a minimal turn record for combat-log comparison tests."""

    return {
        "turn": turn,
        "actions": [],
        "combat": {
            "casualties_by_unit": casualties_by_unit,
            "summary": "combat summary",
        },
        "state": {
            "units": {
                "blue-1": {"faction": "blue", "strength": blue_strength},
                "red-1": {"faction": "red", "strength": red_strength},
            }
        },
        "metadata": {},
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Persist a list of JSON records as newline-delimited JSON."""

    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
