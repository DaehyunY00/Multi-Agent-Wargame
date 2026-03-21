"""Tests for log-driven experiment metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wargame.analysis import (
    action_entropy,
    escalation_sensitivity_index,
    win_rate,
)
from wargame.core.enums import Faction


def test_action_entropy_reads_from_jsonl_logs(tmp_path: Path) -> None:
    """Action entropy should reflect the action distribution stored in JSONL."""

    log_path = tmp_path / "entropy.jsonl"
    _write_jsonl(
        log_path,
        [
            _record(
                turn=1,
                actions=[
                    {"unit_id": "blue-1", "action_type": "hold"},
                    {"unit_id": "red-1", "action_type": "attack"},
                ],
            ),
            _record(
                turn=2,
                actions=[
                    {"unit_id": "blue-1", "action_type": "hold"},
                    {"unit_id": "red-1", "action_type": "attack"},
                ],
            ),
        ],
    )

    assert action_entropy(log_path) == pytest.approx(1.0)


def test_escalation_sensitivity_index_tracks_turn_to_turn_volatility(
    tmp_path: Path,
) -> None:
    """ESI should increase when turn aggression shifts sharply."""

    log_path = tmp_path / "esi.jsonl"
    _write_jsonl(
        log_path,
        [
            _record(
                turn=1,
                actions=[
                    {"unit_id": "blue-1", "action_type": "hold"},
                    {"unit_id": "red-1", "action_type": "hold"},
                ],
            ),
            _record(
                turn=2,
                actions=[
                    {"unit_id": "blue-1", "action_type": "attack"},
                    {"unit_id": "red-1", "action_type": "attack"},
                ],
            ),
            _record(
                turn=3,
                actions=[
                    {"unit_id": "blue-1", "action_type": "move"},
                    {"unit_id": "red-1", "action_type": "move"},
                ],
            ),
        ],
    )

    assert escalation_sensitivity_index(log_path) == pytest.approx(0.75)


def test_win_rate_uses_final_logged_force_totals(tmp_path: Path) -> None:
    """Win-rate calculation should be derived from stored final states."""

    blue_win_path = tmp_path / "blue_win.jsonl"
    red_win_path = tmp_path / "red_win.jsonl"
    draw_path = tmp_path / "draw.jsonl"

    _write_jsonl(
        blue_win_path,
        [_record(turn=1, actions=[], state=_state(blue_strength=60, red_strength=10))],
    )
    _write_jsonl(
        red_win_path,
        [_record(turn=1, actions=[], state=_state(blue_strength=5, red_strength=50))],
    )
    _write_jsonl(
        draw_path,
        [_record(turn=1, actions=[], state=_state(blue_strength=20, red_strength=20))],
    )

    runs = [blue_win_path, red_win_path, draw_path]
    assert win_rate(runs, faction=Faction.BLUE) == pytest.approx(1 / 3)
    assert win_rate(runs, faction=Faction.RED) == pytest.approx(1 / 3)


def _record(
    *,
    turn: int,
    actions: list[dict[str, object]],
    state: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a minimal turn record compatible with the analysis layer."""

    record: dict[str, object] = {
        "turn": turn,
        "actions": actions,
        "metadata": {
            "blue": {
                "used_fallback": False,
                "metadata": {"inference_time_s": 0.01},
            },
            "red": {
                "used_fallback": False,
                "metadata": {"inference_time_s": 0.01},
            },
        },
    }
    if state is not None:
        record["state"] = state
    return record


def _state(*, blue_strength: int, red_strength: int) -> dict[str, object]:
    """Build a minimal serialized state snapshot."""

    return {
        "units": {
            "blue-1": {"faction": "blue", "strength": blue_strength},
            "red-1": {"faction": "red", "strength": red_strength},
        }
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Persist a list of JSON records as newline-delimited JSON."""

    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
