"""Tests for scripted, rule-based, and random baseline agents."""

from wargame.agents import RandomAgent, RuleBasedAgent, ScriptAgent, ScriptBehavior, StateRenderer
from wargame.core import Faction, HexGrid, Position, TerrainType, Unit
from wargame.core.enums import VisibilityLevel
from wargame.engine import FactionViewState, ObservedUnit, TurnMetadata
from wargame.agents.parser import ActionParser


def _render_report() -> str:
    renderer = StateRenderer()
    return renderer.render(
        FactionViewState(
            faction=Faction.BLUE,
            turn_metadata=TurnMetadata(turn=1, max_turns=5),
            friendly_units={
                "blue-1": Unit(
                    unit_id="blue-1",
                    faction=Faction.BLUE,
                    position=Position(0, 0),
                    strength=120,
                ),
                "blue-2": Unit(
                    unit_id="blue-2",
                    faction=Faction.BLUE,
                    position=Position(1, 0),
                    strength=80,
                ),
            },
            enemy_observations={
                "red-1": ObservedUnit(
                    unit_id="red-1",
                    faction=Faction.RED,
                    visibility=VisibilityLevel.IDENTIFIED,
                    last_known_position=Position(2, 0),
                    estimated_strength=90,
                    strength_range=(90, 90),
                    observed_turn=1,
                )
            },
            terrain_by_hex={
                Position(0, 0): TerrainType.OPEN,
                Position(1, 0): TerrainType.FOREST,
                Position(2, 0): TerrainType.OPEN,
            },
            metadata={"scenario_id": "baseline-test"},
        )
    )


def _validate_agent_output(agent) -> None:
    report = _render_report()
    parser = ActionParser(grid=HexGrid(width=5, height=5))

    decision = agent.decide(report, valid_unit_ids={"blue-1", "blue-2"})

    assert len(decision.actions) == 2
    parser.validate_actions(decision.actions, valid_unit_ids={"blue-1", "blue-2"})


def test_script_agent_presets_produce_valid_actions() -> None:
    """Each scripted preset should emit valid engine commands."""

    for behavior in (
        ScriptBehavior.FRONTAL_ASSAULT,
        ScriptBehavior.FLANK_MANEUVER,
        ScriptBehavior.DELAY_DEFENSE,
    ):
        agent = ScriptAgent(
            grid=HexGrid(width=5, height=5),
            faction=Faction.BLUE,
            behavior=behavior,
        )
        _validate_agent_output(agent)


def test_rule_based_agent_is_deterministic_for_fixed_seed() -> None:
    """Rule-based logic should be reproducible under the same fixed seed."""

    first = RuleBasedAgent(
        grid=HexGrid(width=5, height=5),
        faction=Faction.BLUE,
        seed=7,
    )
    second = RuleBasedAgent(
        grid=HexGrid(width=5, height=5),
        faction=Faction.BLUE,
        seed=7,
    )
    report = _render_report()

    first_decision = first.decide(report, valid_unit_ids={"blue-1", "blue-2"})
    second_decision = second.decide(report, valid_unit_ids={"blue-1", "blue-2"})

    assert first_decision.actions == second_decision.actions
    ActionParser(grid=HexGrid(width=5, height=5)).validate_actions(
        first_decision.actions,
        valid_unit_ids={"blue-1", "blue-2"},
    )


def test_random_agent_samples_only_valid_actions() -> None:
    """Random baseline should sample only from parser-valid action space."""

    agent = RandomAgent(
        grid=HexGrid(width=5, height=5),
        faction=Faction.BLUE,
        seed=11,
    )

    _validate_agent_output(agent)
