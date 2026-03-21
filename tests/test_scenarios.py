"""Tests for YAML scenario loading and state conversion."""

from __future__ import annotations

from pathlib import Path

from wargame.core import Faction, Position
from wargame.scenarios import build_grid, load_scenario, scenario_to_game_state


def test_preset_scenario_loads_into_game_state() -> None:
    """Preset YAML scenarios should load into executable game state objects."""

    scenario_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "wargame"
        / "scenarios"
        / "presets"
        / "s1_open_encounter.yaml"
    )
    scenario = load_scenario(scenario_path)
    grid = build_grid(scenario)
    state = scenario_to_game_state(scenario)

    assert scenario.scenario_id == "s1_open_encounter"
    assert grid.width == 20
    assert grid.height == 15
    assert len(scenario.forces) == 6
    assert len(state.units) == 6
    assert len(state.terrain_by_hex) >= 3
    assert state.metadata["scenario_name"] == "Open Encounter"
    assert "Secure the urban crossroads" in state.metadata["mission"][Faction.BLUE.value][0]
    assert Position(10, 7) in scenario.map.terrain
