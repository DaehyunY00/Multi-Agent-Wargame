"""Integration tests for the main tactical turn loop."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass, field

from wargame.agents import (
    ActionParser,
    AgentDecision,
    BaseAgent,
    HeuristicWhiteCellAgent,
    StateRenderer,
)
from wargame.combat import LanchesterResolver
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
from wargame.logging import JsonlLogger
from wargame.orchestrator import TurnLoop


@dataclass(slots=True)
class FakeAgent(BaseAgent):
    """Deterministic fake agent used for turn-loop integration tests."""

    planned_actions: list[ActionCommand] = field(default_factory=list)
    reasoning_text: str = "test reasoning"
    doctrine_text: str = "test doctrine"

    def decide(
        self,
        state_text: str,
        *,
        valid_unit_ids: Collection[str] = (),
    ) -> AgentDecision:
        return AgentDecision(
            faction=self.faction,
            reasoning=self.reasoning_text,
            doctrine_reference=self.doctrine_text,
            actions=list(self.planned_actions),
            metadata={"saw_state_text": bool(state_text), "valid_unit_ids": sorted(valid_unit_ids)},
        )


def test_turn_loop_runs_end_to_end_and_writes_jsonl(tmp_path) -> None:
    """A full game should run with fake agents, combat, fallback, and JSONL logs."""

    initial_state = GameState(
        turn=0,
        max_turns=1,
        units={
            "blue-1": Unit(
                unit_id="blue-1",
                faction=Faction.BLUE,
                position=Position(0, 0),
                strength=100,
            ),
            "red-1": Unit(
                unit_id="red-1",
                faction=Faction.RED,
                position=Position(1, 0),
                strength=100,
            ),
        },
        terrain_by_hex={
            Position(0, 0): TerrainType.OPEN,
            Position(1, 0): TerrainType.OPEN,
        },
    )
    fog = FogOfWarFilter(visibility_radius=3, identification_radius=1)
    engine = SimulationEngine(
        state_manager=StateManager(initial_state=initial_state, fog_of_war=fog),
        combat_resolver=LanchesterResolver(),
        fog_of_war=fog,
    )
    blue_agent = FakeAgent(
        name="blue_test_agent",
        faction=Faction.BLUE,
        planned_actions=[
            ActionCommand(
                unit_id="blue-1",
                action_type=ActionType.HOLD,
                posture=Posture.DEFEND,
            )
        ],
    )
    red_agent = FakeAgent(
        name="red_test_agent",
        faction=Faction.RED,
        planned_actions=[
            ActionCommand(
                unit_id="red-1",
                action_type=ActionType.MOVE,
                target_hex=Position(99, 99),
                posture=Posture.MANEUVER,
            )
        ],
    )
    logger = JsonlLogger(path=tmp_path / "turns.jsonl")
    loop = TurnLoop(
        engine=engine,
        blue_agent=blue_agent,
        red_agent=red_agent,
        renderer=StateRenderer(),
        parser=ActionParser(grid=HexGrid(width=5, height=5)),
        logger=logger,
    )

    results = loop.run_until_terminal()

    assert len(results) == 1
    assert results[0].combat is not None
    assert results[0].turn == 1
    assert results[0].metadata["blue"]["used_fallback"] is False
    assert results[0].metadata["red"]["used_fallback"] is True
    assert engine.is_terminal() is True
    assert engine.get_log()[0].turn == 1

    current_state = engine.state_manager.current_state()
    assert current_state.turn == 1
    assert current_state.units["blue-1"].strength < 100
    assert current_state.units["red-1"].strength < 100
    assert current_state.units["red-1"].position == Position(1, 0)

    lines = (tmp_path / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["turn"] == 1
    assert record["metadata"]["red"]["used_fallback"] is True
    assert record["metadata"]["red"]["metadata"]["error_type"] == "ActionParseError"
    assert record["metadata"]["red"]["metadata"]["error_stage"] == "action_validation"
    assert record["state"]["turn"] == 1
    assert "blue-1" in record["state"]["units"]


def test_turn_loop_records_white_cell_scores(tmp_path) -> None:
    """White-cell evaluation should attach doctrine and rationality scores to turn logs."""

    initial_state = GameState(
        turn=0,
        max_turns=1,
        units={
            "blue-1": Unit(
                unit_id="blue-1",
                faction=Faction.BLUE,
                position=Position(0, 0),
                strength=100,
            ),
            "red-1": Unit(
                unit_id="red-1",
                faction=Faction.RED,
                position=Position(1, 0),
                strength=90,
            ),
        },
        terrain_by_hex={
            Position(0, 0): TerrainType.OPEN,
            Position(1, 0): TerrainType.FOREST,
        },
    )
    fog = FogOfWarFilter(visibility_radius=3, identification_radius=1)
    engine = SimulationEngine(
        state_manager=StateManager(initial_state=initial_state, fog_of_war=fog),
        combat_resolver=LanchesterResolver(),
        fog_of_war=fog,
        grid=HexGrid(width=5, height=5),
    )
    loop = TurnLoop(
        engine=engine,
        blue_agent=FakeAgent(
            name="blue_attack",
            faction=Faction.BLUE,
            planned_actions=[
                ActionCommand(
                    unit_id="blue-1",
                    action_type=ActionType.ATTACK,
                    target_hex=Position(1, 0),
                    posture=Posture.ATTACK,
                )
            ],
        ),
        red_agent=FakeAgent(
            name="red_hold",
            faction=Faction.RED,
            planned_actions=[
                ActionCommand(
                    unit_id="red-1",
                    action_type=ActionType.HOLD,
                    posture=Posture.DEFEND,
                )
            ],
        ),
        white_cell=HeuristicWhiteCellAgent(),
        renderer=StateRenderer(),
        parser=ActionParser(grid=HexGrid(width=5, height=5)),
        logger=JsonlLogger(path=tmp_path / "white_cell.jsonl"),
    )

    result = loop.run_until_terminal()[0]

    white_cell_metadata = result.metadata["white_cell"]["metadata"]
    assert white_cell_metadata["scores"]["doctrine_compliance"] > 0.0
    assert white_cell_metadata["scores"]["tactical_rationality"] >= 1.0
