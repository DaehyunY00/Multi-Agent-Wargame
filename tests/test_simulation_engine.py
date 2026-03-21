"""Tests for localized combat semantics in the simulation engine."""

from __future__ import annotations

from wargame.combat import LanchesterConfig, LanchesterResolver
from wargame.core import (
    ActionCommand,
    ActionType,
    Faction,
    GameState,
    HexGrid,
    Position,
    Posture,
    TerrainType,
    Unit,
)
from wargame.engine import FogOfWarFilter, SimulationEngine, StateManager


def test_attack_does_not_move_and_only_engaged_units_take_losses() -> None:
    """Attack actions should preserve position and combat should stay localized."""

    engine = _build_engine(
        GameState(
            turn=0,
            max_turns=3,
            units={
                "blue-1": Unit("blue-1", Faction.BLUE, Position(0, 0), 100),
                "blue-2": Unit("blue-2", Faction.BLUE, Position(0, 4), 100),
                "red-1": Unit("red-1", Faction.RED, Position(1, 0), 100),
                "red-2": Unit("red-2", Faction.RED, Position(4, 4), 100),
            },
            terrain_by_hex={
                Position(0, 0): TerrainType.OPEN,
                Position(1, 0): TerrainType.OPEN,
                Position(0, 4): TerrainType.OPEN,
                Position(4, 4): TerrainType.OPEN,
            },
        )
    )

    engine.execute_actions(
        [
            ActionCommand("blue-1", ActionType.ATTACK, target_hex=Position(1, 0), posture=Posture.ATTACK),
            ActionCommand("blue-2", ActionType.HOLD, posture=Posture.DEFEND),
            ActionCommand("red-1", ActionType.HOLD, posture=Posture.DEFEND),
            ActionCommand("red-2", ActionType.HOLD, posture=Posture.DEFEND),
        ]
    )

    state_after_orders = engine.state_manager.current_state()
    assert state_after_orders.units["blue-1"].position == Position(0, 0)

    combat = engine.resolve_combat()
    state_after_combat = engine.state_manager.current_state()

    assert combat.combat is not None
    assert combat.metadata["engagement_count"] == 1
    assert state_after_combat.units["blue-1"].strength < 100
    assert state_after_combat.units["red-1"].strength < 100
    assert state_after_combat.units["blue-2"].strength == 100
    assert state_after_combat.units["red-2"].strength == 100


def test_stochastic_combat_is_reproducible_when_engine_seed_is_fixed() -> None:
    """Engine-level combat seeding should reproduce stochastic engagement outcomes."""

    initial_state = GameState(
        turn=0,
        max_turns=3,
        units={
            "blue-1": Unit("blue-1", Faction.BLUE, Position(0, 0), 100),
            "red-1": Unit("red-1", Faction.RED, Position(1, 0), 100),
        },
        terrain_by_hex={
            Position(0, 0): TerrainType.OPEN,
            Position(1, 0): TerrainType.OPEN,
        },
    )
    first = _build_engine(initial_state, stochastic=True, combat_seed=21)
    second = _build_engine(initial_state, stochastic=True, combat_seed=21)

    actions = [
        ActionCommand("blue-1", ActionType.HOLD, posture=Posture.DEFEND),
        ActionCommand("red-1", ActionType.HOLD, posture=Posture.DEFEND),
    ]
    first.execute_actions(actions)
    second.execute_actions(actions)

    first_result = first.resolve_combat()
    second_result = second.resolve_combat()

    assert first_result.combat == second_result.combat
    assert first_result.metadata == second_result.metadata


def _build_engine(
    initial_state: GameState,
    *,
    stochastic: bool = False,
    combat_seed: int | None = None,
) -> SimulationEngine:
    """Construct a simulation engine for focused combat tests."""

    fog = FogOfWarFilter(visibility_radius=3, identification_radius=1)
    return SimulationEngine(
        state_manager=StateManager(initial_state=initial_state, fog_of_war=fog),
        combat_resolver=LanchesterResolver(
            LanchesterConfig(stochastic=stochastic, noise_std=0.2)
        ),
        fog_of_war=fog,
        grid=HexGrid(width=6, height=6),
        combat_seed=combat_seed,
    )
