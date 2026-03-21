"""Tests for canonical state management and faction-specific fog of war."""

from wargame.combat import LanchesterResolver
from wargame.core import Faction, GameState, Position, TerrainType, Unit
from wargame.engine import FogOfWarFilter, SimulationEngine, StateManager


def _build_state() -> GameState:
    return GameState(
        turn=0,
        max_turns=2,
        units={
            "blue-1": Unit(
                unit_id="blue-1",
                faction=Faction.BLUE,
                position=Position(0, 0),
                strength=100,
            ),
            "red-near": Unit(
                unit_id="red-near",
                faction=Faction.RED,
                position=Position(1, 0),
                strength=90,
            ),
            "red-detected": Unit(
                unit_id="red-detected",
                faction=Faction.RED,
                position=Position(2, 0),
                strength=80,
            ),
            "red-hidden": Unit(
                unit_id="red-hidden",
                faction=Faction.RED,
                position=Position(6, 0),
                strength=120,
            ),
        },
        terrain_by_hex={
            Position(0, 0): TerrainType.OPEN,
            Position(1, 0): TerrainType.FOREST,
            Position(2, 0): TerrainType.URBAN,
            Position(6, 0): TerrainType.MOUNTAIN,
        },
    )


def _build_engine() -> SimulationEngine:
    fog = FogOfWarFilter(visibility_radius=3, identification_radius=1, uncertainty_ratio=0.25)
    manager = StateManager(initial_state=_build_state(), fog_of_war=fog)
    return SimulationEngine(
        state_manager=manager,
        combat_resolver=LanchesterResolver(),
        fog_of_war=fog,
    )


def test_get_state_returns_visible_and_hidden_enemy_information() -> None:
    """Only enemies within visibility range should appear in the faction view."""

    engine = _build_engine()

    blue_view = engine.get_state(Faction.BLUE)

    assert "blue-1" in blue_view.friendly_units
    assert "red-near" in blue_view.enemy_observations
    assert "red-hidden" not in blue_view.enemy_observations
    assert blue_view.enemy_observations["red-near"].faction == Faction.RED


def test_detected_enemy_supports_uncertain_observations() -> None:
    """Detected but not identified units should carry uncertainty metadata."""

    engine = _build_engine()

    contact = engine.get_state(Faction.BLUE).enemy_observations["red-detected"]

    assert contact.visibility.value == "detected"
    assert contact.estimated_strength == 80
    assert contact.strength_range == (60, 100)
    assert contact.posture is None
    assert contact.status is None


def test_state_snapshots_are_stable_and_separate_from_internal_state() -> None:
    """Mutating returned snapshots should not mutate the canonical full state."""

    manager = StateManager(initial_state=_build_state(), fog_of_war=FogOfWarFilter())

    snapshot = manager.snapshot()
    snapshot.units["blue-1"].strength = 1

    assert manager.current_state().units["blue-1"].strength == 100


def test_turn_counter_behavior_updates_metadata() -> None:
    """Turn progression should update both state metadata and engine log."""

    engine = _build_engine()

    engine.advance_turn()
    result = engine.advance_turn()
    blue_view = engine.get_state(Faction.BLUE)

    assert result.turn == 2
    assert blue_view.turn_metadata.turn == 2
    assert blue_view.turn_metadata.max_turns == 2
    assert engine.is_terminal() is True
