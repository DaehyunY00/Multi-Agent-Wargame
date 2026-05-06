"""Tests for aggregate experiment visualization helpers and CLI."""

from __future__ import annotations

import builtins
import csv
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from visualize_logs import _collect_log_paths, main  # noqa: E402


def test_plots_module_keeps_matplotlib_lazy_on_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the plots module should not import matplotlib eagerly."""

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise AssertionError("matplotlib should not be imported at module import time")
        return original_import(name, globals, locals, fromlist, level)

    sys.modules.pop("wargame.analysis.plots", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("wargame.analysis.plots")

    assert hasattr(module, "build_aggregate_visualizations")


def test_collect_log_paths_accepts_files_and_directories(tmp_path: Path) -> None:
    """CLI path collection should merge directory and file inputs deterministically."""

    log_dir = tmp_path / "logs"
    nested_dir = log_dir / "nested"
    nested_dir.mkdir(parents=True)

    first = log_dir / "a.jsonl"
    second = nested_dir / "b.jsonl"
    _write_jsonl(first, [_turn_record(turn=1)])
    _write_jsonl(second, [_turn_record(turn=1)])

    collected = _collect_log_paths([str(log_dir), str(first)])

    assert collected == sorted([first.resolve(), second.resolve()])


def test_build_battlefield_replay_writes_html_manifest_and_svg_frames(tmp_path: Path) -> None:
    """Replay helper should write turn SVG frames plus a browser-viewable index."""

    from wargame.analysis import build_battlefield_replay

    records = _write_synthetic_runs(tmp_path / "runs")
    first_run_records = [
        json.loads(line)
        for line in records[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_dir = tmp_path / "replay"

    bundle = build_battlefield_replay(
        first_run_records,
        output_dir,
        run_label="s1-rule-vs-random-seed-0",
        title="Force Strength Over Turns - Open Encounter (s1-rule-vs-random-seed-0)",
    )

    expected = {
        output_dir / "index.html",
        output_dir / "manifest.json",
        output_dir / "turn_001.svg",
        output_dir / "turn_002.svg",
    }
    assert set(bundle.files) == expected

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 2
    assert manifest["run_label"] == "s1-rule-vs-random-seed-0"
    assert manifest["frames"][0]["turn"] == 1
    assert manifest["frames"][1]["moved_unit_ids"] == ["blue-1"]

    html = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "전장 replay" in html
    assert "turn_001.svg" in html

    svg = (output_dir / "turn_002.svg").read_text(encoding="utf-8")
    assert "Turn 2 | blue=80 | red=10" in svg
    assert "Open Encounter" in svg


def test_visualize_logs_cli_writes_expected_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The visualization CLI should emit summaries, comparison plots, and force curves."""

    pytest.importorskip("matplotlib", reason="plot generation requires matplotlib")
    pytest.importorskip("numpy", reason="plot generation requires numpy")

    log_dir = tmp_path / "runs"
    log_paths = _write_synthetic_runs(log_dir)
    output_dir = tmp_path / "visualizations"

    exit_code = main([str(log_dir), str(log_paths[0]), "--output-dir", str(output_dir)])

    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["output_dir"] == str(output_dir)
    assert payload["run_count"] == 3
    assert len(payload["written_files"]) == 28

    summary_json = output_dir / "aggregate_summary.json"
    summary_csv = output_dir / "aggregate_summary.csv"
    assert summary_json.exists()
    assert summary_csv.exists()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["input_log_count"] == 3
    assert summary["run_count"] == 3
    assert summary["scenario_ids"] == ["s1_open_encounter", "s2_river_crossing"]
    assert summary["agent_names"] == ["random", "rule"]
    assert summary["blue_win_rate"] == pytest.approx(2 / 3)
    assert summary["red_win_rate"] == pytest.approx(1 / 3)
    assert summary["mean_json_parsing_success_rate"] == pytest.approx(0.75)

    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["input_log_count"] == "3"
    assert rows[0]["run_count"] == "3"
    assert rows[0]["scenario_ids"] == "s1_open_encounter;s2_river_crossing"
    assert rows[0]["agent_names"] == "random;rule"

    expected_paths = {
        output_dir / "win_rate_by_scenario.png",
        output_dir / "win_rate_by_scenario.svg",
        output_dir / "action_entropy_by_agent.png",
        output_dir / "action_entropy_by_agent.svg",
        output_dir / "escalation_sensitivity_by_agent.png",
        output_dir / "escalation_sensitivity_by_agent.svg",
        output_dir / "parsing_success_by_agent.png",
        output_dir / "parsing_success_by_agent.svg",
        output_dir / "force_curves" / "s1-rule-vs-random-seed-0.png",
        output_dir / "force_curves" / "s1-rule-vs-random-seed-0.svg",
        output_dir / "force_curves" / "s1-random-vs-rule-seed-1.png",
        output_dir / "force_curves" / "s1-random-vs-rule-seed-1.svg",
        output_dir / "force_curves" / "s2-rule-vs-rule-seed-2.png",
        output_dir / "force_curves" / "s2-rule-vs-rule-seed-2.svg",
        output_dir / "battlefield_replays" / "s1-rule-vs-random-seed-0" / "index.html",
        output_dir / "battlefield_replays" / "s1-rule-vs-random-seed-0" / "manifest.json",
        output_dir / "battlefield_replays" / "s1-rule-vs-random-seed-0" / "turn_001.svg",
        output_dir / "battlefield_replays" / "s1-rule-vs-random-seed-0" / "turn_002.svg",
        output_dir / "battlefield_replays" / "s1-random-vs-rule-seed-1" / "index.html",
        output_dir / "battlefield_replays" / "s1-random-vs-rule-seed-1" / "manifest.json",
        output_dir / "battlefield_replays" / "s1-random-vs-rule-seed-1" / "turn_001.svg",
        output_dir / "battlefield_replays" / "s1-random-vs-rule-seed-1" / "turn_002.svg",
        output_dir / "battlefield_replays" / "s2-rule-vs-rule-seed-2" / "index.html",
        output_dir / "battlefield_replays" / "s2-rule-vs-rule-seed-2" / "manifest.json",
        output_dir / "battlefield_replays" / "s2-rule-vs-rule-seed-2" / "turn_001.svg",
        output_dir / "battlefield_replays" / "s2-rule-vs-rule-seed-2" / "turn_002.svg",
    }
    for path in expected_paths:
        assert path.exists(), f"missing artifact: {path}"


def _write_synthetic_runs(base_dir: Path) -> list[Path]:
    """Create a small multi-run fixture with distinct scenarios and agents."""

    base_dir.mkdir(parents=True, exist_ok=True)

    run1 = base_dir / "run1.jsonl"
    _write_jsonl(
        run1,
        [
            _turn_record(
                turn=1,
                run_id="s1-rule-vs-random-seed-0",
                scenario_id="s1_open_encounter",
                scenario_name="Open Encounter",
                blue_agent="rule",
                red_agent="random",
                blue_actions=[{"unit_id": "blue-1", "action_type": "hold"}],
                red_actions=[{"unit_id": "red-1", "action_type": "attack", "target_hex": {"q": 0, "r": 0}}],
                blue_parse_success=True,
                red_parse_success=False,
                blue_strength=90,
                red_strength=70,
                blue_position=(0, 0),
                red_position=(2, 0),
                combat_summary="Initial skirmish near the central approach.",
                casualties_by_unit={"red-1": 30},
            ),
            _turn_record(
                turn=2,
                run_id="s1-rule-vs-random-seed-0",
                scenario_id="s1_open_encounter",
                scenario_name="Open Encounter",
                blue_agent="rule",
                red_agent="random",
                blue_actions=[{"unit_id": "blue-1", "action_type": "move", "target_hex": {"q": 1, "r": 0}}],
                red_actions=[{"unit_id": "red-1", "action_type": "attack", "target_hex": {"q": 1, "r": 0}}],
                blue_parse_success=True,
                red_parse_success=True,
                blue_strength=80,
                red_strength=10,
                blue_position=(1, 0),
                red_position=(2, 0),
                combat_summary="Blue closes the gap while red force is attrited.",
                casualties_by_unit={"blue-1": 10, "red-1": 60},
            ),
        ],
    )

    run2 = base_dir / "run2.jsonl"
    _write_jsonl(
        run2,
        [
            _turn_record(
                turn=1,
                run_id="s1-random-vs-rule-seed-1",
                scenario_id="s1_open_encounter",
                scenario_name="Open Encounter",
                blue_agent="random",
                red_agent="rule",
                blue_actions=[{"unit_id": "blue-1", "action_type": "move", "target_hex": {"q": 1, "r": 1}}],
                red_actions=[{"unit_id": "red-1", "action_type": "hold"}],
                blue_parse_success=False,
                red_parse_success=True,
                blue_strength=65,
                red_strength=80,
                blue_position=(0, 1),
                red_position=(2, 1),
                combat_summary="Red holds a stronger line in open terrain.",
                casualties_by_unit={"blue-1": 35, "red-1": 20},
            ),
            _turn_record(
                turn=2,
                run_id="s1-random-vs-rule-seed-1",
                scenario_id="s1_open_encounter",
                scenario_name="Open Encounter",
                blue_agent="random",
                red_agent="rule",
                blue_actions=[{"unit_id": "blue-1", "action_type": "attack", "target_hex": {"q": 2, "r": 1}}],
                red_actions=[{"unit_id": "red-1", "action_type": "hold"}],
                blue_parse_success=False,
                red_parse_success=True,
                blue_strength=15,
                red_strength=60,
                blue_position=(1, 1),
                red_position=(2, 1),
                combat_summary="Blue pushes forward but remains heavily outmatched.",
                casualties_by_unit={"blue-1": 50, "red-1": 20},
            ),
        ],
    )

    run3 = base_dir / "run3.jsonl"
    _write_jsonl(
        run3,
        [
            _turn_record(
                turn=1,
                run_id="s2-rule-vs-rule-seed-2",
                scenario_id="s2_river_crossing",
                scenario_name="River Crossing",
                blue_agent="rule",
                red_agent="rule",
                blue_actions=[{"unit_id": "blue-1", "action_type": "attack", "target_hex": {"q": 6, "r": 4}}],
                red_actions=[{"unit_id": "red-1", "action_type": "move", "target_hex": {"q": 5, "r": 4}}],
                blue_parse_success=True,
                red_parse_success=True,
                blue_strength=95,
                red_strength=90,
                blue_position=(4, 4),
                red_position=(6, 4),
                combat_summary="Both forces probe the bridgehead with limited losses.",
                casualties_by_unit={"blue-1": 5, "red-1": 10},
            ),
            _turn_record(
                turn=2,
                run_id="s2-rule-vs-rule-seed-2",
                scenario_id="s2_river_crossing",
                scenario_name="River Crossing",
                blue_agent="rule",
                red_agent="rule",
                blue_actions=[{"unit_id": "blue-1", "action_type": "attack", "target_hex": {"q": 6, "r": 4}}],
                red_actions=[{"unit_id": "red-1", "action_type": "move", "target_hex": {"q": 5, "r": 4}}],
                blue_parse_success=True,
                red_parse_success=True,
                blue_strength=75,
                red_strength=40,
                blue_position=(5, 4),
                red_position=(6, 4),
                combat_summary="Blue secures the crossing with concentrated pressure.",
                casualties_by_unit={"blue-1": 20, "red-1": 50},
            ),
        ],
    )

    return [run1, run2, run3]


def _turn_record(
    *,
    turn: int,
    run_id: str = "run",
    scenario_id: str = "scenario",
    scenario_name: str = "Scenario",
    blue_agent: str = "rule",
    red_agent: str = "random",
    blue_actions: list[dict[str, object]] | None = None,
    red_actions: list[dict[str, object]] | None = None,
    blue_parse_success: bool = True,
    red_parse_success: bool = True,
    blue_strength: int = 100,
    red_strength: int = 100,
    blue_position: tuple[int, int] = (0, 0),
    red_position: tuple[int, int] = (1, 0),
    combat_summary: str = "No combat recorded.",
    casualties_by_unit: dict[str, int] | None = None,
) -> dict[str, object]:
    """Build one minimal JSONL turn record compatible with analysis helpers."""

    blue_action_list = blue_actions or [{"unit_id": "blue-1", "action_type": "hold"}]
    red_action_list = red_actions or [{"unit_id": "red-1", "action_type": "hold"}]
    combat_losses = casualties_by_unit or {}
    return {
        "turn": turn,
        "actions": [*blue_action_list, *red_action_list],
        "context": {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "seed": 0,
            "blue_agent": blue_agent,
            "red_agent": red_agent,
            "white_cell": None,
            "initial_force_totals": {"blue": 100, "red": 100},
            "metadata": {"seed_control": "manual"},
        },
        "metadata": {
            "blue": {
                "actions": blue_action_list,
                "used_fallback": False,
                "metadata": {
                    "decision_source": "local_llm",
                    "json_parse_success": blue_parse_success,
                    "inference_time_s": 0.01,
                },
            },
            "red": {
                "actions": red_action_list,
                "used_fallback": False,
                "metadata": {
                    "decision_source": "local_llm",
                    "json_parse_success": red_parse_success,
                    "inference_time_s": 0.01,
                },
            },
            "white_cell": None,
        },
        "combat": {
            "attacker_ids": ["blue-1"],
            "defender_ids": ["red-1"],
            "casualties_by_unit": combat_losses,
            "winner": "blue" if blue_strength > red_strength else "red" if red_strength > blue_strength else None,
            "summary": combat_summary,
        },
        "notes": [
            f"blue-1 at {blue_position} / red-1 at {red_position}",
            combat_summary,
        ],
        "state": {
            "metadata": {
                "scenario_name": scenario_name,
                "objective_hexes": {
                    "blue": [{"q": red_position[0], "r": red_position[1]}],
                    "red": [{"q": blue_position[0], "r": blue_position[1]}],
                },
            },
            "terrain_by_hex": {
                f"{blue_position[0]},{blue_position[1]}": "open",
                f"{red_position[0]},{red_position[1]}": "urban" if scenario_id == "s2_river_crossing" else "forest",
            },
            "units": {
                "blue-1": {
                    "faction": "blue",
                    "strength": blue_strength,
                    "position": {"q": blue_position[0], "r": blue_position[1]},
                    "posture": "attack",
                    "status": "ready",
                },
                "red-1": {
                    "faction": "red",
                    "strength": red_strength,
                    "position": {"q": red_position[0], "r": red_position[1]},
                    "posture": "defend",
                    "status": "ready",
                },
            }
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Persist newline-delimited JSON records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
