"""Tests for deterministic observed-state rendering."""

from wargame.agents import StateRenderer
from wargame.core import Faction, Position, TerrainType, Unit
from wargame.engine import FactionViewState, ObservedUnit, TurnMetadata
from wargame.core.enums import VisibilityLevel


def test_state_to_text_renders_stable_tactical_report_shape() -> None:
    """Observed state should render into deterministic, structured text."""

    view = FactionViewState(
        faction=Faction.BLUE,
        turn_metadata=TurnMetadata(turn=2, max_turns=10),
        friendly_units={
            "blue-2": Unit(
                unit_id="blue-2",
                faction=Faction.BLUE,
                position=Position(1, 0),
                strength=90,
            ),
            "blue-1": Unit(
                unit_id="blue-1",
                faction=Faction.BLUE,
                position=Position(0, 0),
                strength=100,
            ),
        },
        enemy_observations={
            "red-1": ObservedUnit(
                unit_id="red-1",
                faction=Faction.RED,
                visibility=VisibilityLevel.DETECTED,
                last_known_position=Position(2, 1),
                estimated_strength=80,
                strength_range=(64, 96),
                observed_turn=2,
            )
        },
        terrain_by_hex={
            Position(1, 0): TerrainType.FOREST,
            Position(0, 0): TerrainType.OPEN,
        },
        metadata={"visible_enemy_count": 1, "scenario_id": "demo"},
    )
    renderer = StateRenderer()

    first = renderer.render(view)
    second = renderer.render(view)

    assert first == second
    assert "TACTICAL REPORT" in first
    assert "Turn: 2/10" in first
    assert "Faction: blue" in first
    assert "Friendly Units:" in first
    assert "- blue-1: position=(0,0), strength=100, posture=defend, status=ready" in first
    assert "Enemy Contacts:" in first
    assert "red-1: visibility=detected, position=(2,1), estimated_strength=80, range=64-96, observed_turn=2" in first
    assert "Terrain References:" in first
    assert "- (0,0): open" in first
    assert "- scenario_id=demo" in first
