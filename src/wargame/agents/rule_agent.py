"""Deterministic rule-based baseline agent."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from wargame.core.enums import ActionType, Faction, Posture
from wargame.core.hexgrid import HexGrid
from wargame.core.models import ActionCommand
from wargame.engine.fog_of_war import FactionViewState

from .base import AgentDecision, BaseAgent
from ._baseline_utils import (
    forward_step,
    hex_distance,
    nearest_enemy_with_position,
    parse_tactical_report,
    tactical_report_from_view,
    step_away,
    step_toward,
)


@dataclass(slots=True, kw_only=True)
class RuleBasedAgent(BaseAgent):
    """Rule-based baseline agent with interpretable if-then logic.

    Rules are evaluated in order for each friendly unit:
    1. No visible enemy -> recon forward.
    2. Outmatched and enemy is close -> withdraw.
    3. Strong enough and adjacent -> attack.
    4. Strong enough and enemy is distant -> maneuver toward enemy.
    5. Otherwise hold and defend.
    """

    grid: HexGrid
    seed: int | str | None = None
    name: str = "rule_agent"
    faction: Faction | None = None

    def decide(
        self,
        state_text: str,
        *,
        valid_unit_ids: Collection[str] = (),
    ) -> AgentDecision:
        report = parse_tactical_report(state_text)
        actions = [
            self._action_for_unit(report, unit_id)
            for unit_id in sorted(valid_unit_ids)
            if unit_id in report.friendly_units
        ]
        return AgentDecision(
            faction=self.faction,
            reasoning="Applied deterministic rule-based baseline logic.",
            doctrine_reference="baseline/rule/interpretable_if_then",
            actions=actions,
            metadata={"rule_set": "v1"},
        )

    def decide_view(
        self,
        state_view: FactionViewState,
        *,
        valid_unit_ids: Collection[str] = (),
    ) -> AgentDecision:
        """Produce actions directly from structured observed state."""

        report = tactical_report_from_view(state_view)
        actions = [
            self._action_for_unit(report, unit_id)
            for unit_id in sorted(valid_unit_ids)
            if unit_id in report.friendly_units
        ]
        return AgentDecision(
            faction=self.faction,
            reasoning="Applied deterministic rule-based baseline logic.",
            doctrine_reference="baseline/rule/interpretable_if_then",
            actions=actions,
            metadata={"rule_set": "v1", "decision_source": "structured_state"},
        )

    def reset_seed(self, seed: int | str | None) -> None:
        """Reset the deterministic tiebreak seed used by this agent."""

        self.seed = seed

    def _action_for_unit(self, report, unit_id: str) -> ActionCommand:
        unit = report.friendly_units[unit_id]
        enemy = nearest_enemy_with_position(report, from_position=unit.position)
        if enemy is None or enemy.position is None:
            target = forward_step(
                self.grid,
                start=unit.position,
                faction=self.faction,
                seed=self.seed,
                context=f"recon:{unit_id}:{report.turn}",
            )
            if target is not None:
                return ActionCommand(
                    unit_id=unit_id,
                    action_type=ActionType.RECON,
                    target_hex=target,
                    posture=Posture.MANEUVER,
                )
            return _hold_action(unit_id)

        distance = hex_distance(unit.position, enemy.position)
        enemy_strength = enemy.estimated_strength

        if unit.strength < enemy_strength and distance <= 2:
            target = step_away(
                self.grid,
                start=unit.position,
                threat=enemy.position,
                seed=self.seed,
                context=f"withdraw:{unit_id}:{report.turn}",
            )
            if target is not None:
                return ActionCommand(
                    unit_id=unit_id,
                    action_type=ActionType.WITHDRAW,
                    target_hex=target,
                    posture=Posture.WITHDRAW,
                )
            return _hold_action(unit_id)

        if unit.strength >= enemy_strength and distance <= 1:
            return ActionCommand(
                unit_id=unit_id,
                action_type=ActionType.ATTACK,
                target_hex=enemy.position,
                posture=Posture.ATTACK,
            )

        if unit.strength >= enemy_strength:
            target = step_toward(
                self.grid,
                start=unit.position,
                target=enemy.position,
                seed=self.seed,
                context=f"maneuver:{unit_id}:{report.turn}",
            )
            if target is not None:
                return ActionCommand(
                    unit_id=unit_id,
                    action_type=ActionType.MOVE,
                    target_hex=target,
                    posture=Posture.MANEUVER,
                )

        return _hold_action(unit_id)


RuleAgent = RuleBasedAgent


def _hold_action(unit_id: str) -> ActionCommand:
    """Build a conservative hold action."""

    return ActionCommand(
        unit_id=unit_id,
        action_type=ActionType.HOLD,
        posture=Posture.DEFEND,
    )
