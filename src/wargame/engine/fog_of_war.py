"""Fog-of-war filtering and faction-specific state export helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from wargame.core.enums import Faction, Posture, TerrainType, UnitStatus, VisibilityLevel
from wargame.core.models import GameState, Position, Unit


@dataclass(frozen=True, slots=True)
class TurnMetadata:
    """Turn-level metadata shared by full-state and observed-state exports."""

    turn: int
    max_turns: int
    phase: str = "command"


@dataclass(frozen=True, slots=True)
class ObservedUnit:
    """Faction-specific observation about an enemy unit."""

    unit_id: str
    faction: Faction
    visibility: VisibilityLevel
    last_known_position: Position | None
    estimated_strength: int | None
    strength_range: tuple[int, int] | None
    observed_turn: int
    posture: Posture | None = None
    status: UnitStatus | None = None


@dataclass(frozen=True, slots=True)
class FactionViewState:
    """Observed state exported to a single faction."""

    faction: Faction
    turn_metadata: TurnMetadata
    friendly_units: dict[str, Unit]
    enemy_observations: dict[str, ObservedUnit]
    terrain_by_hex: dict[Position, TerrainType]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class FogOfWarFilter:
    """Visibility policy for converting full state into faction views."""

    visibility_radius: int = 3
    identification_radius: int = 1
    uncertainty_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.visibility_radius < 0:
            raise ValueError("visibility_radius must be non-negative.")
        if self.identification_radius < 0:
            raise ValueError("identification_radius must be non-negative.")
        if self.identification_radius > self.visibility_radius:
            raise ValueError("identification_radius cannot exceed visibility_radius.")
        if self.uncertainty_ratio < 0:
            raise ValueError("uncertainty_ratio must be non-negative.")

    def filter_state(self, state: GameState, faction: Faction) -> FactionViewState:
        """Project the canonical full state into one faction's observed view."""

        friendly_units = {
            unit_id: deepcopy(unit)
            for unit_id, unit in state.units.items()
            if unit.faction == faction
        }
        enemy_observations = {
            unit_id: observation
            for unit_id, observation in (
                self._observe_enemy(unit=unit, state=state, faction=faction)
                for unit in state.units.values()
                if unit.faction != faction
            )
            if observation is not None
        }

        metadata = dict(state.metadata)
        metadata["visible_enemy_count"] = len(enemy_observations)

        return FactionViewState(
            faction=faction,
            turn_metadata=TurnMetadata(turn=state.turn, max_turns=state.max_turns),
            friendly_units=friendly_units,
            enemy_observations=enemy_observations,
            terrain_by_hex=deepcopy(state.terrain_by_hex),
            metadata=metadata,
        )

    def _observe_enemy(
        self,
        *,
        unit: Unit,
        state: GameState,
        faction: Faction,
    ) -> tuple[str, ObservedUnit | None]:
        """Return an observed contact if the enemy unit is within sensor range."""

        nearest_distance = self._nearest_friendly_distance(
            enemy_position=unit.position,
            state=state,
            faction=faction,
        )
        if nearest_distance is None or nearest_distance > self.visibility_radius:
            return unit.unit_id, None

        if nearest_distance <= self.identification_radius:
            return unit.unit_id, ObservedUnit(
                unit_id=unit.unit_id,
                faction=unit.faction,
                visibility=VisibilityLevel.IDENTIFIED,
                last_known_position=unit.position,
                estimated_strength=unit.strength,
                strength_range=(unit.strength, unit.strength),
                observed_turn=state.turn,
                posture=unit.posture,
                status=unit.status,
            )

        uncertainty = max(1, round(unit.strength * self.uncertainty_ratio))
        return unit.unit_id, ObservedUnit(
            unit_id=unit.unit_id,
            faction=unit.faction,
            visibility=VisibilityLevel.DETECTED,
            last_known_position=unit.position,
            estimated_strength=unit.strength,
            strength_range=(max(0, unit.strength - uncertainty), unit.strength + uncertainty),
            observed_turn=state.turn,
        )

    @staticmethod
    def _nearest_friendly_distance(
        *,
        enemy_position: Position,
        state: GameState,
        faction: Faction,
    ) -> int | None:
        """Return the shortest distance from any friendly unit to the enemy."""

        distances = [
            _hex_distance(friendly.position, enemy_position)
            for friendly in state.units.values()
            if friendly.faction == faction
        ]
        return min(distances) if distances else None


def _hex_distance(start: Position, end: Position) -> int:
    """Compute axial hex distance without requiring a grid instance."""

    dq = end.q - start.q
    dr = end.r - start.r
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2
