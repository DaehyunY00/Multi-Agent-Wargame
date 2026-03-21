"""Terrain definitions and configurable modifier lookup helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import TerrainType


@dataclass(frozen=True, slots=True)
class TerrainModifier:
    """Movement, defense, and observation effects for a terrain type."""

    terrain: TerrainType
    movement_cost: float
    defense_modifier: float
    observation_modifier: float = 0.0


DEFAULT_TERRAIN_MODIFIERS: dict[TerrainType, TerrainModifier] = {
    TerrainType.OPEN: TerrainModifier(
        terrain=TerrainType.OPEN,
        movement_cost=1.0,
        defense_modifier=1.0,
        observation_modifier=0.0,
    ),
    TerrainType.MOUNTAIN: TerrainModifier(
        terrain=TerrainType.MOUNTAIN,
        movement_cost=2.0,
        defense_modifier=1.5,
        observation_modifier=-0.2,
    ),
    TerrainType.URBAN: TerrainModifier(
        terrain=TerrainType.URBAN,
        movement_cost=1.5,
        defense_modifier=1.4,
        observation_modifier=-0.2,
    ),
    TerrainType.FOREST: TerrainModifier(
        terrain=TerrainType.FOREST,
        movement_cost=1.5,
        defense_modifier=1.25,
        observation_modifier=-0.3,
    ),
    TerrainType.RIVER: TerrainModifier(
        terrain=TerrainType.RIVER,
        movement_cost=2.5,
        defense_modifier=0.9,
        observation_modifier=0.1,
    ),
}


@dataclass(frozen=True, slots=True)
class TerrainLibrary:
    """Container for configurable terrain modifier lookups."""

    modifiers: dict[TerrainType, TerrainModifier] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "modifiers", dict(self.modifiers))

    def get_modifier(self, terrain: TerrainType) -> TerrainModifier:
        """Look up the modifier bundle for a terrain type."""

        try:
            return self.modifiers[terrain]
        except KeyError as exc:
            raise KeyError(f"No terrain modifier configured for {terrain.value!r}.") from exc

    @classmethod
    def default(cls) -> "TerrainLibrary":
        """Build the explicit default terrain table for the core engine."""

        return cls(modifiers=DEFAULT_TERRAIN_MODIFIERS)
