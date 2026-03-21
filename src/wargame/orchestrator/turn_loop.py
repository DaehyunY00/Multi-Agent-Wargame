"""Turn-loop orchestration for agents, engine, combat, and logging."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Any

from wargame.agents.base import AgentDecision, BaseAgent, StructuredStateAgent
from wargame.agents.parser import ActionParser
from wargame.agents.state_to_text import StateRenderer
from wargame.core.models import TurnResult
from wargame.engine.fog_of_war import FactionViewState
from wargame.engine.simulation import SimulationEngine
from wargame.logging.jsonl_logger import JsonlLogger


@dataclass(slots=True)
class TurnLoop:
    """Coordinate the per-turn flow between engine, agents, and adjudication."""

    engine: SimulationEngine
    blue_agent: BaseAgent
    red_agent: BaseAgent
    renderer: StateRenderer
    parser: ActionParser
    white_cell: BaseAgent | None = None
    logger: JsonlLogger | None = None
    turn_history: list[TurnResult] = field(default_factory=list)

    def run_turn(self) -> TurnResult:
        """Run one full blue/red decision cycle."""

        if self.engine.is_terminal():
            raise RuntimeError("Cannot run a turn after the game is terminal.")

        starting_turn = self.engine.state_manager.turn_metadata().turn

        blue_view = self.engine.get_state(self.blue_agent.faction)
        blue_report = self.renderer.render(blue_view)
        blue_decision = self._decide_for_faction(
            agent=self.blue_agent,
            observed_state=blue_view,
            rendered_state=blue_report,
            valid_unit_ids=blue_view.friendly_units.keys(),
        )
        blue_execution = self.engine.execute_actions(blue_decision.actions)

        red_view = self.engine.get_state(self.red_agent.faction)
        red_report = self.renderer.render(red_view)
        red_decision = self._decide_for_faction(
            agent=self.red_agent,
            observed_state=red_view,
            rendered_state=red_report,
            valid_unit_ids=red_view.friendly_units.keys(),
        )
        red_execution = self.engine.execute_actions(red_decision.actions)

        combat_result = self.engine.resolve_combat()
        advanced_turn = self.engine.advance_turn()

        white_cell_decision = self._evaluate_white_cell(
            blue_view=blue_view,
            red_view=red_view,
            blue_decision=blue_decision,
            red_decision=red_decision,
            combat_result=combat_result,
        )

        final_state = self.engine.state_manager.current_state()
        completed_turn = TurnResult(
            turn=advanced_turn.turn,
            actions=[*blue_decision.actions, *red_decision.actions],
            combat=combat_result.combat,
            notes=[
                *blue_execution.notes,
                *red_execution.notes,
                *combat_result.notes,
                *advanced_turn.notes,
            ],
            metadata={
                "starting_turn": starting_turn,
                "blue": self._decision_metadata(blue_decision),
                "red": self._decision_metadata(red_decision),
                "white_cell": self._decision_metadata(white_cell_decision)
                if white_cell_decision is not None
                else None,
                "terminal": self.engine.is_terminal(),
            },
        )

        self.engine.record_turn(completed_turn)
        self.turn_history.append(completed_turn)
        if self.logger is not None:
            self.logger.log_turn(completed_turn, state=final_state)
            self.logger.flush()
        return completed_turn

    def run_until_terminal(self) -> list[TurnResult]:
        """Run turns until the scenario reaches a terminal condition."""

        results: list[TurnResult] = []
        while not self.engine.is_terminal():
            results.append(self.run_turn())
        return results

    def _decide_for_faction(
        self,
        *,
        agent: BaseAgent,
        observed_state: FactionViewState,
        rendered_state: str,
        valid_unit_ids: Collection[str],
    ) -> AgentDecision:
        """Collect a safe, validated decision for one faction."""

        unit_ids = set(valid_unit_ids)
        started_at = perf_counter()
        try:
            if isinstance(agent, StructuredStateAgent):
                decision = agent.decide_view(observed_state, valid_unit_ids=unit_ids)
            else:
                decision = agent.decide(rendered_state, valid_unit_ids=unit_ids)
        except Exception as exc:
            return self._fallback_decision(
                agent=agent,
                unit_ids=unit_ids,
                error=exc,
                error_stage="agent_decision",
                started_at=started_at,
            )

        decision.metadata = {
            **decision.metadata,
            "source_agent": agent.name,
            "agent_class": type(agent).__name__,
            "inference_time_s": decision.metadata.get(
                "inference_time_s",
                perf_counter() - started_at,
            ),
        }

        try:
            self.parser.validate_actions(decision.actions, valid_unit_ids=unit_ids)
            return decision
        except Exception as exc:
            return self._fallback_decision(
                agent=agent,
                unit_ids=unit_ids,
                error=exc,
                error_stage="action_validation",
                started_at=started_at,
            )

    def _fallback_decision(
        self,
        *,
        agent: BaseAgent,
        unit_ids: set[str],
        error: Exception,
        error_stage: str,
        started_at: float,
    ) -> AgentDecision:
        """Build a logged fallback decision after a decision or validation failure."""

        fallback = self.parser.build_fallback_plan(unit_ids=unit_ids, error=error)
        return AgentDecision(
            faction=agent.faction,
            reasoning=fallback.reasoning,
            doctrine_reference=fallback.doctrine_reference,
            actions=fallback.actions,
            used_fallback=True,
            metadata={
                "errors": list(fallback.errors),
                "source_agent": agent.name,
                "agent_class": type(agent).__name__,
                "error_type": type(error).__name__,
                "error_stage": error_stage,
                "inference_time_s": perf_counter() - started_at,
            },
        )

    def _evaluate_white_cell(
        self,
        *,
        blue_view: FactionViewState,
        red_view: FactionViewState,
        blue_decision: AgentDecision,
        red_decision: AgentDecision,
        combat_result: TurnResult,
    ) -> AgentDecision | None:
        """Run an optional white-cell evaluation against a turn summary."""

        if self.white_cell is None:
            return None

        white_report = json.dumps(
            self._build_white_cell_payload(
                blue_view=blue_view,
                red_view=red_view,
                blue_decision=blue_decision,
                red_decision=red_decision,
                combat_result=combat_result,
            ),
            sort_keys=True,
        )
        try:
            return self.white_cell.decide(white_report, valid_unit_ids=())
        except Exception as exc:
            return AgentDecision(
                faction=self.white_cell.faction,
                reasoning="White-cell evaluation failed.",
                doctrine_reference="white_cell/error",
                used_fallback=True,
                metadata={"errors": [str(exc)]},
            )

    @staticmethod
    def _decision_metadata(decision: AgentDecision) -> dict[str, object]:
        """Summarize an agent decision into JSON-friendly metadata."""

        return {
            "faction": decision.faction.value if decision.faction is not None else None,
            "reasoning": decision.reasoning,
            "doctrine_reference": decision.doctrine_reference,
            "actions": [TurnLoop._serialize_action(action) for action in decision.actions],
            "used_fallback": decision.used_fallback,
            "metadata": dict(decision.metadata),
        }

    @staticmethod
    def _build_white_cell_payload(
        *,
        blue_view: FactionViewState,
        red_view: FactionViewState,
        blue_decision: AgentDecision,
        red_decision: AgentDecision,
        combat_result: TurnResult,
    ) -> dict[str, Any]:
        """Build a structured, machine-readable payload for white-cell scoring."""

        return {
            "turn": combat_result.turn,
            "blue_view": TurnLoop._serialize_view(blue_view),
            "red_view": TurnLoop._serialize_view(red_view),
            "blue_decision": TurnLoop._decision_metadata(blue_decision),
            "red_decision": TurnLoop._decision_metadata(red_decision),
            "combat": TurnLoop._serialize_combat(combat_result),
        }

    @staticmethod
    def _serialize_view(view: FactionViewState) -> dict[str, Any]:
        """Convert a faction view into a JSON-friendly summary."""

        return {
            "faction": view.faction.value,
            "turn_metadata": {
                "turn": view.turn_metadata.turn,
                "max_turns": view.turn_metadata.max_turns,
                "phase": view.turn_metadata.phase,
            },
            "friendly_units": {
                unit_id: TurnLoop._serialize_unit(unit)
                for unit_id, unit in sorted(view.friendly_units.items())
            },
            "enemy_observations": {
                unit_id: {
                    "unit_id": observation.unit_id,
                    "faction": observation.faction.value,
                    "visibility": observation.visibility.value,
                    "last_known_position": TurnLoop._serialize_position(observation.last_known_position),
                    "estimated_strength": observation.estimated_strength,
                    "strength_range": list(observation.strength_range)
                    if observation.strength_range is not None
                    else None,
                    "observed_turn": observation.observed_turn,
                    "posture": observation.posture.value if observation.posture is not None else None,
                    "status": observation.status.value if observation.status is not None else None,
                }
                for unit_id, observation in sorted(view.enemy_observations.items())
            },
            "terrain_by_hex": {
                f"{position.q},{position.r}": getattr(terrain, "value", terrain)
                for position, terrain in sorted(view.terrain_by_hex.items(), key=lambda item: (item[0].q, item[0].r))
            },
            "metadata": TurnLoop._serialize_value(view.metadata),
        }

    @staticmethod
    def _serialize_unit(unit: Any) -> dict[str, Any]:
        """Serialize a unit-like object for turn logging."""

        return {
            "unit_id": unit.unit_id,
            "faction": unit.faction.value,
            "position": TurnLoop._serialize_position(unit.position),
            "strength": unit.strength,
            "combat_power": unit.combat_power,
            "supply": unit.supply,
            "morale": unit.morale,
            "posture": unit.posture.value,
            "status": unit.status.value,
        }

    @staticmethod
    def _serialize_action(action: Any) -> dict[str, Any]:
        """Serialize an action command for white-cell and log metadata."""

        return {
            "unit_id": action.unit_id,
            "action_type": action.action_type.value,
            "target_hex": TurnLoop._serialize_position(action.target_hex),
            "posture": action.posture.value if action.posture is not None else None,
            "metadata": TurnLoop._serialize_value(action.metadata),
        }

    @staticmethod
    def _serialize_combat(result: TurnResult) -> dict[str, Any]:
        """Serialize the combat portion of a turn result."""

        combat = result.combat
        if combat is None:
            return {}
        return {
            "attacker_ids": list(combat.attacker_ids),
            "defender_ids": list(combat.defender_ids),
            "casualties_by_unit": dict(combat.casualties_by_unit),
            "winner": combat.winner.value if combat.winner is not None else None,
            "summary": combat.summary,
            "metadata": TurnLoop._serialize_value(result.metadata),
        }

    @staticmethod
    def _serialize_position(position: Any) -> dict[str, int] | None:
        """Serialize a hex position when present."""

        if position is None:
            return None
        return {"q": int(position.q), "r": int(position.r)}

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize nested values into JSON-friendly structures."""

        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {
                str(key): TurnLoop._serialize_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [TurnLoop._serialize_value(item) for item in value]
        if hasattr(value, "q") and hasattr(value, "r"):
            return TurnLoop._serialize_position(value)
        return value
