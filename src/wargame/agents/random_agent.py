"""Random baseline agent with valid action sampling only."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from random import Random

from wargame.core.enums import ActionType, Faction, Posture
from wargame.core.hexgrid import HexGrid
from wargame.core.models import ActionCommand
from wargame.engine.fog_of_war import FactionViewState

from .base import AgentDecision, BaseAgent
from ._baseline_utils import (
    nearest_enemy_with_position,
    parse_tactical_report,
    tactical_report_from_view,
)


@dataclass(slots=True, kw_only=True)
class RandomAgent(BaseAgent):
    """Random baseline agent that samples only from valid actions."""

    grid: HexGrid
    seed: int | str | None = None
    name: str = "random_agent"
    faction: Faction | None = None
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)

    def decide(
        self,
        state_text: str,
        *,
        valid_unit_ids: Collection[str] = (),
    ) -> AgentDecision:
        report = parse_tactical_report(state_text)
        actions = [
            self._sample_action(report, unit_id)
            for unit_id in sorted(valid_unit_ids)
            if unit_id in report.friendly_units
        ]
        return AgentDecision(
            faction=self.faction,
            reasoning="Sampled uniformly from valid baseline actions.",
            doctrine_reference="baseline/random/uniform_valid_actions",
            actions=actions,
            metadata={"seed": self.seed},
        )

    def decide_view(
        self,
        state_view: FactionViewState,
        *,
        valid_unit_ids: Collection[str] = (),
    ) -> AgentDecision:
        """Sample valid actions directly from structured observed state."""

        report = tactical_report_from_view(state_view)
        actions = [
            self._sample_action(report, unit_id)
            for unit_id in sorted(valid_unit_ids)
            if unit_id in report.friendly_units
        ]
        return AgentDecision(
            faction=self.faction,
            reasoning="Sampled uniformly from valid baseline actions.",
            doctrine_reference="baseline/random/uniform_valid_actions",
            actions=actions,
            metadata={"seed": self.seed, "decision_source": "structured_state"},
        )

    def reset_seed(self, seed: int | str | None) -> None:
        """Reset the internal random sampler for reproducible experiments."""

        self.seed = seed
        self._rng = Random(seed)

    def _sample_action(self, report, unit_id: str) -> ActionCommand:
        unit = report.friendly_units[unit_id]
        candidates = [
            ActionCommand(
                unit_id=unit_id,
                action_type=ActionType.HOLD,
                posture=Posture.DEFEND,
            )
        ]
        for neighbor in sorted(self.grid.neighbors(unit.position), key=lambda item: (item.q, item.r)):
            candidates.append(
                ActionCommand(
                    unit_id=unit_id,
                    action_type=ActionType.MOVE,
                    target_hex=neighbor,
                    posture=Posture.MANEUVER,
                )
            )
            candidates.append(
                ActionCommand(
                    unit_id=unit_id,
                    action_type=ActionType.RECON,
                    target_hex=neighbor,
                    posture=Posture.MANEUVER,
                )
            )
            candidates.append(
                ActionCommand(
                    unit_id=unit_id,
                    action_type=ActionType.WITHDRAW,
                    target_hex=neighbor,
                    posture=Posture.WITHDRAW,
                )
            )

        enemy = nearest_enemy_with_position(report, from_position=unit.position)
        if enemy is not None and enemy.position is not None:
            candidates.append(
                ActionCommand(
                    unit_id=unit_id,
                    action_type=ActionType.ATTACK,
                    target_hex=enemy.position,
                    posture=Posture.ATTACK,
                )
            )
            candidates.append(
                ActionCommand(
                    unit_id=unit_id,
                    action_type=ActionType.SUPPORT_BY_FIRE,
                    target_hex=enemy.position,
                    posture=Posture.SUPPORT,
                )
            )

        return self._rng.choice(candidates)
