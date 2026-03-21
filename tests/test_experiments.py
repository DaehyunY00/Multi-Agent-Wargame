"""Tests for experiment runner behavior and logging context."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass, field

from wargame.agents import AgentDecision, BaseAgent, ActionParser, StateRenderer
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
from wargame.experiments import ExperimentRunner
from wargame.logging import JsonlLogger
from wargame.orchestrator import TurnLoop
from wargame.scenarios.schema import ScenarioSpec


@dataclass(slots=True)
class StaticAgent(BaseAgent):
    """Deterministic agent for experiment-runner tests."""

    planned_actions: list[ActionCommand] = field(default_factory=list)

    def decide(
        self,
        state_text: str,
        *,
        valid_unit_ids: Collection[str] = (),
    ) -> AgentDecision:
        del state_text, valid_unit_ids
        return AgentDecision(
            faction=self.faction,
            reasoning="static",
            doctrine_reference="test/static",
            actions=list(self.planned_actions),
        )


def test_experiment_runner_applies_seed_hook_and_logs_context(tmp_path) -> None:
    """ExperimentRunner should expose seed handling explicitly in log context."""

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
    loop = TurnLoop(
        engine=engine,
        blue_agent=StaticAgent(
            name="blue_static",
            faction=Faction.BLUE,
            planned_actions=[
                ActionCommand(
                    unit_id="blue-1",
                    action_type=ActionType.HOLD,
                    posture=Posture.DEFEND,
                )
            ],
        ),
        red_agent=StaticAgent(
            name="red_static",
            faction=Faction.RED,
            planned_actions=[
                ActionCommand(
                    unit_id="red-1",
                    action_type=ActionType.HOLD,
                    posture=Posture.DEFEND,
                )
            ],
        ),
        renderer=StateRenderer(),
        parser=ActionParser(grid=HexGrid(width=5, height=5)),
        logger=JsonlLogger(path=tmp_path / "run.jsonl"),
    )
    scenario = ScenarioSpec(scenario_id="s_test", name="Test Scenario")
    captured: list[int | None] = []

    runner = ExperimentRunner(
        scenario=scenario,
        turn_loop=loop,
        seed=7,
        prepare_run=lambda turn_loop, seed: captured.append(seed),
    )

    run = runner.run()

    assert run.context.seed == 7
    assert run.context.metadata["seed_control"] == "hook"
    assert captured == [7]

    record = json.loads((tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["context"]["seed"] == 7
    assert record["context"]["metadata"]["seed_control"] == "hook"
    assert record["context"]["run_id"] == run.context.run_id


def test_experiment_runner_auto_seeding_logs_seed_bundle(tmp_path) -> None:
    """Default experiment execution should apply and log deterministic component seeds."""

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
        grid=HexGrid(width=5, height=5),
    )
    loop = TurnLoop(
        engine=engine,
        blue_agent=StaticAgent(
            name="blue_static",
            faction=Faction.BLUE,
            planned_actions=[
                ActionCommand(
                    unit_id="blue-1",
                    action_type=ActionType.HOLD,
                    posture=Posture.DEFEND,
                )
            ],
        ),
        red_agent=StaticAgent(
            name="red_static",
            faction=Faction.RED,
            planned_actions=[
                ActionCommand(
                    unit_id="red-1",
                    action_type=ActionType.HOLD,
                    posture=Posture.DEFEND,
                )
            ],
        ),
        renderer=StateRenderer(),
        parser=ActionParser(grid=HexGrid(width=5, height=5)),
        logger=JsonlLogger(path=tmp_path / "auto_seed.jsonl"),
    )
    scenario = ScenarioSpec(scenario_id="s_test", name="Test Scenario")

    run = ExperimentRunner(
        scenario=scenario,
        turn_loop=loop,
        seed=3,
    ).run()

    assert run.context.metadata["seed_control"] == "automatic"
    assert run.context.metadata["seed_bundle"]["engine_seed"] == 31
