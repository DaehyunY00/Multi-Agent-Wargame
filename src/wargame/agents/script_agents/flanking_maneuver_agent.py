"""Scripted agent that approaches visible enemies from the flank."""

from __future__ import annotations

from dataclasses import dataclass

from wargame.core.enums import ActionType, Posture
from wargame.core.hexgrid import AXIAL_DIRECTIONS, HexGrid
from wargame.core.models import ActionCommand, Position, Unit
from wargame.engine.fog_of_war import FactionViewState, ObservedUnit

_SQRT_3_OVER_2 = 0.8660254037844386


@dataclass(slots=True, kw_only=True)
class FlankingManeuverAgent:
    """Single-unit scripted agent that prefers side approaches over frontal ones.

    The agent chooses one friendly unit from the provided
    :class:`~wargame.engine.fog_of_war.FactionViewState` and returns a single
    engine-facing :class:`~wargame.core.models.ActionCommand`.

    Decision policy:
    1. Hold defensively when no enemy positions are visible.
    2. Attack the nearest visible enemy when already adjacent.
    3. Otherwise move one hex toward an enemy flank objective chosen from the
       enemy's neighboring hexes whose direction is most perpendicular to the
       current line of approach. If no flank step is available, fall back to a
       direct advance toward that enemy.
    """

    grid: HexGrid
    unit_id: str | None = None
    name: str = "flanking_maneuver_agent"
    prefer_left_flank: bool = True

    def decide(self, state: FactionViewState) -> ActionCommand:
        """Return the next action for the controlled unit."""

        unit_id, unit = self._select_unit(state)
        enemy = self._nearest_visible_enemy(unit.position, state)
        if enemy is None or enemy.last_known_position is None:
            return self._hold_action(unit_id)

        enemy_position = enemy.last_known_position
        if self.grid.distance(unit.position, enemy_position) <= 1:
            return ActionCommand(
                unit_id=unit_id,
                action_type=ActionType.ATTACK,
                target_hex=enemy_position,
                posture=Posture.ATTACK,
                metadata={"script": "flanking_maneuver", "target": enemy.unit_id},
            )

        blocked_positions = {
            friendly.position
            for friendly_id, friendly in state.friendly_units.items()
            if friendly_id != unit_id
        }
        flank_step = self._select_flanking_step(
            start=unit.position,
            enemy_position=enemy_position,
            blocked_positions=blocked_positions,
        )
        if flank_step is None:
            flank_step = self._best_step_toward(
                start=unit.position,
                target=enemy_position,
                blocked_positions=blocked_positions | {enemy_position},
            )
        if flank_step is None:
            return self._hold_action(unit_id)

        return ActionCommand(
            unit_id=unit_id,
            action_type=ActionType.MOVE,
            target_hex=flank_step,
            posture=Posture.MANEUVER,
            metadata={"script": "flanking_maneuver", "target": enemy.unit_id},
        )

    def _select_unit(self, state: FactionViewState) -> tuple[str, Unit]:
        """Return the friendly unit controlled by this agent."""

        if self.unit_id is not None:
            unit = state.friendly_units.get(self.unit_id)
            if unit is None:
                raise ValueError(
                    f"Friendly unit {self.unit_id!r} is not present in the state view."
                )
            return self.unit_id, unit

        if not state.friendly_units:
            raise ValueError("Cannot decide without at least one friendly unit.")

        unit_id = sorted(state.friendly_units)[0]
        return unit_id, state.friendly_units[unit_id]

    def _nearest_visible_enemy(
        self,
        position: Position,
        state: FactionViewState,
    ) -> ObservedUnit | None:
        """Return the closest visible enemy with a known position."""

        visible_enemies = [
            enemy
            for enemy in state.enemy_observations.values()
            if enemy.last_known_position is not None
        ]
        if not visible_enemies:
            return None

        return min(
            visible_enemies,
            key=lambda enemy: (
                self.grid.distance(position, enemy.last_known_position),
                enemy.unit_id,
            ),
        )

    def _select_flanking_step(
        self,
        *,
        start: Position,
        enemy_position: Position,
        blocked_positions: set[Position],
    ) -> Position | None:
        """Choose the first feasible step toward the best flank objective."""

        for flank_objective in self._ranked_flank_objectives(
            start=start,
            enemy_position=enemy_position,
        ):
            step = self._best_step_toward(
                start=start,
                target=flank_objective,
                blocked_positions=blocked_positions | {enemy_position},
            )
            if step is not None:
                return step
        return None

    def _ranked_flank_objectives(
        self,
        *,
        start: Position,
        enemy_position: Position,
    ) -> list[Position]:
        """Rank enemy-adjacent flank objectives by perpendicularity and side."""

        approach_vector = _axial_to_cartesian(
            Position(enemy_position.q - start.q, enemy_position.r - start.r)
        )
        desired_side_sign = 1 if self.prefer_left_flank else -1
        objectives: list[tuple[float, int, int, int, Position]] = []

        for delta in AXIAL_DIRECTIONS:
            objective = Position(enemy_position.q + delta.q, enemy_position.r + delta.r)
            if not self.grid.is_within_bounds(objective):
                continue
            if objective == start:
                continue

            flank_vector = _axial_to_cartesian(delta)
            perpendicularity = abs(_dot(approach_vector, flank_vector))
            side_rank = (
                0
                if _cross(approach_vector, flank_vector) * desired_side_sign > 0
                else 1
            )
            objectives.append(
                (
                    perpendicularity,
                    side_rank,
                    self.grid.distance(start, objective),
                    objective.q,
                    objective.r,
                    objective,
                )
            )

        objectives.sort()
        return [objective for *_, objective in objectives]

    def _best_step_toward(
        self,
        *,
        start: Position,
        target: Position,
        blocked_positions: set[Position],
    ) -> Position | None:
        """Choose a deterministic neighboring step that closes on the target."""

        current_distance = self.grid.distance(start, target)
        candidates = [
            neighbor
            for neighbor in self.grid.neighbors(start)
            if neighbor not in blocked_positions
            and self.grid.distance(neighbor, target) < current_distance
        ]
        if not candidates:
            return None

        candidates.sort(
            key=lambda neighbor: (
                self.grid.distance(neighbor, target),
                neighbor.q,
                neighbor.r,
            )
        )
        return candidates[0]

    @staticmethod
    def _hold_action(unit_id: str) -> ActionCommand:
        """Build a conservative hold action."""

        return ActionCommand(
            unit_id=unit_id,
            action_type=ActionType.HOLD,
            posture=Posture.DEFEND,
            metadata={"script": "flanking_maneuver"},
        )


def _axial_to_cartesian(delta: Position) -> tuple[float, float]:
    """Project an axial offset into a 2D plane for direction ranking."""

    return (delta.q + 0.5 * delta.r, _SQRT_3_OVER_2 * delta.r)


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    """Return the 2D dot product."""

    return (left[0] * right[0]) + (left[1] * right[1])


def _cross(left: tuple[float, float], right: tuple[float, float]) -> float:
    """Return the signed 2D cross product."""

    return (left[0] * right[1]) - (left[1] * right[0])
