"""Tests for the dedicated flanking scripted agent."""

from wargame.agents import FlankingManeuverAgent
from wargame.core import Faction, HexGrid, Position, TerrainType, Unit
from wargame.core.enums import ActionType, Posture, VisibilityLevel
from wargame.engine import FactionViewState, ObservedUnit, TurnMetadata


def _build_state(
    *,
    friendly_position: Position,
    enemy_position: Position | None,
) -> FactionViewState:
    enemy_observations: dict[str, ObservedUnit] = {}
    if enemy_position is not None:
        enemy_observations["red-1"] = ObservedUnit(
            unit_id="red-1",
            faction=Faction.RED,
            visibility=VisibilityLevel.IDENTIFIED,
            last_known_position=enemy_position,
            estimated_strength=90,
            strength_range=(90, 90),
            observed_turn=1,
            posture=Posture.DEFEND,
        )

    return FactionViewState(
        faction=Faction.BLUE,
        turn_metadata=TurnMetadata(turn=1, max_turns=5),
        friendly_units={
            "blue-1": Unit(
                unit_id="blue-1",
                faction=Faction.BLUE,
                position=friendly_position,
                strength=100,
            )
        },
        enemy_observations=enemy_observations,
        terrain_by_hex={
            Position(q, r): TerrainType.OPEN
            for q in range(6)
            for r in range(6)
        },
        metadata={},
    )


def test_flanking_agent_holds_when_no_enemy_is_visible() -> None:
    """The agent should default to a defensive hold without enemy contact."""

    agent = FlankingManeuverAgent(
        grid=HexGrid(width=6, height=6),
        unit_id="blue-1",
    )

    decision = agent.decide(
        _build_state(
            friendly_position=Position(1, 1),
            enemy_position=None,
        )
    )

    assert decision.action_type is ActionType.HOLD
    assert decision.posture is Posture.DEFEND
    assert decision.target_hex is None


def test_flanking_agent_attacks_when_enemy_is_adjacent() -> None:
    """The agent should attack instead of maneuvering when already adjacent."""

    agent = FlankingManeuverAgent(
        grid=HexGrid(width=6, height=6),
        unit_id="blue-1",
    )

    decision = agent.decide(
        _build_state(
            friendly_position=Position(2, 2),
            enemy_position=Position(3, 2),
        )
    )

    assert decision.action_type is ActionType.ATTACK
    assert decision.posture is Posture.ATTACK
    assert decision.target_hex == Position(3, 2)


def test_flanking_agent_moves_toward_a_flank_hex() -> None:
    """The maneuver should bias toward a side approach instead of a direct frontal step."""

    agent = FlankingManeuverAgent(
        grid=HexGrid(width=6, height=6),
        unit_id="blue-1",
    )

    decision = agent.decide(
        _build_state(
            friendly_position=Position(2, 1),
            enemy_position=Position(4, 1),
        )
    )

    assert decision.action_type is ActionType.MOVE
    assert decision.posture is Posture.MANEUVER
    assert decision.target_hex == Position(2, 2)
