"""Transparent scripted baseline agent with preset behavior profiles."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum

from wargame.core.enums import ActionType, Faction, Posture
from wargame.core.hexgrid import HexGrid
from wargame.core.models import ActionCommand
from wargame.engine.fog_of_war import FactionViewState

from .base import AgentDecision, BaseAgent
from ._baseline_utils import (
    flank_objective,
    forward_step,
    nearest_enemy_with_position,
    parse_tactical_report,
    tactical_report_from_view,
    step_away,
    step_toward,
)


class ScriptBehavior(StrEnum):
    """Preset scripted behaviors used as experiment baselines."""

    FRONTAL_ASSAULT = "frontal_assault"
    FLANK_MANEUVER = "flank_maneuver"
    DELAY_DEFENSE = "delay_defense"


@dataclass(slots=True, kw_only=True)
class ScriptAgent(BaseAgent):
    """Scripted baseline agent with transparent behavior presets.

    Supported presets:
    - `frontal_assault`: attack visible enemies directly, otherwise advance.
    - `flank_maneuver`: move toward a simple flank objective near the enemy.
    - `delay_defense`: withdraw from close threats, otherwise hold.
    """

    grid: HexGrid
    behavior: ScriptBehavior = ScriptBehavior.FRONTAL_ASSAULT
    seed: int | str | None = None
    name: str = "script_agent"
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
            reasoning=f"Script behavior '{self.behavior.value}' selected.",
            doctrine_reference=f"baseline/script/{self.behavior.value}",
            actions=actions,
            metadata={"behavior": self.behavior.value},
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
            reasoning=f"Script behavior '{self.behavior.value}' selected.",
            doctrine_reference=f"baseline/script/{self.behavior.value}",
            actions=actions,
            metadata={"behavior": self.behavior.value, "decision_source": "structured_state"},
        )

    def reset_seed(self, seed: int | str | None) -> None:
        """Reset the deterministic tiebreak seed used by this agent."""

        self.seed = seed

    def _action_for_unit(self, report, unit_id: str) -> ActionCommand:
        unit = report.friendly_units[unit_id]
        enemy = nearest_enemy_with_position(report, from_position=unit.position)

        if self.behavior is ScriptBehavior.FRONTAL_ASSAULT:
            if enemy is not None and enemy.position is not None:
                return ActionCommand(
                    unit_id=unit_id,
                    action_type=ActionType.ATTACK,
                    target_hex=enemy.position,
                    posture=Posture.ATTACK,
                )
            target = forward_step(
                self.grid,
                start=unit.position,
                faction=self.faction,
                seed=self.seed,
                context=f"{self.behavior.value}:{unit_id}:{report.turn}",
            )
            if target is not None:
                return ActionCommand(
                    unit_id=unit_id,
                    action_type=ActionType.MOVE,
                    target_hex=target,
                    posture=Posture.MANEUVER,
                )
            return _hold_action(unit_id)

        if self.behavior is ScriptBehavior.FLANK_MANEUVER:
            if enemy is not None and enemy.position is not None:
                objective = flank_objective(enemy.position, faction=self.faction)
                target = (
                    step_toward(
                        self.grid,
                        start=unit.position,
                        target=objective if self.grid.is_within_bounds(objective) else enemy.position,
                        seed=self.seed,
                        context=f"{self.behavior.value}:{unit_id}:{report.turn}",
                    )
                    or step_toward(
                        self.grid,
                        start=unit.position,
                        target=enemy.position,
                        seed=self.seed,
                        context=f"{self.behavior.value}:fallback:{unit_id}:{report.turn}",
                    )
                )
                if target is not None:
                    return ActionCommand(
                        unit_id=unit_id,
                        action_type=ActionType.MOVE,
                        target_hex=target,
                        posture=Posture.MANEUVER,
                    )
            return _hold_action(unit_id)

        if enemy is not None and enemy.position is not None:
            target = step_away(
                self.grid,
                start=unit.position,
                threat=enemy.position,
                seed=self.seed,
                context=f"{self.behavior.value}:{unit_id}:{report.turn}",
            )
            if target is not None:
                return ActionCommand(
                    unit_id=unit_id,
                    action_type=ActionType.WITHDRAW,
                    target_hex=target,
                    posture=Posture.WITHDRAW,
                )
        return _hold_action(unit_id)


def _hold_action(unit_id: str) -> ActionCommand:
    """Build a conservative hold action."""

    return ActionCommand(
        unit_id=unit_id,
        action_type=ActionType.HOLD,
        posture=Posture.DEFEND,
    )
