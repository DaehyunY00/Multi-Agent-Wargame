"""Core domain types and helpers for the tactical wargame."""

from .enums import (
    ActionType,
    Faction,
    Posture,
    TerrainType,
    UnitStatus,
    VisibilityLevel,
)
from .hexgrid import HexGrid
from .models import (
    ActionCommand,
    CombatResult,
    Force,
    GameState,
    HexCoord,
    Observation,
    Position,
    TurnResult,
    Unit,
    UnitState,
)
from .terrain import DEFAULT_TERRAIN_MODIFIERS, TerrainLibrary, TerrainModifier

__all__ = [
    "ActionCommand",
    "ActionType",
    "CombatResult",
    "Faction",
    "Force",
    "GameState",
    "HexCoord",
    "HexGrid",
    "Observation",
    "Posture",
    "Position",
    "DEFAULT_TERRAIN_MODIFIERS",
    "TerrainLibrary",
    "TerrainModifier",
    "TerrainType",
    "TurnResult",
    "Unit",
    "UnitState",
    "UnitStatus",
    "VisibilityLevel",
]
