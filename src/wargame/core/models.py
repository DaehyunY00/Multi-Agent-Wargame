"""Shared dataclasses for tactical state, map positions, and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import (
    ActionType,
    Faction,
    Posture,
    TerrainType,
    UnitStatus,
    VisibilityLevel,
)


@dataclass(frozen=True, slots=True)
class Position:
    """Immutable axial hex coordinate."""

    q: int
    r: int


HexCoord = Position


@dataclass(slots=True)
class Unit:
    """Mutable unit snapshot tracked by the tactical engine."""

    unit_id: str
    faction: Faction
    position: Position
    strength: int
    combat_power: float = 1.0
    supply: float = 1.0
    morale: float = 1.0
    posture: Posture = Posture.DEFEND
    status: UnitStatus = UnitStatus.READY


UnitState = Unit


@dataclass(frozen=True, slots=True)
class Force:
    """Immutable grouping of units under a single command element."""

    force_id: str
    faction: Faction
    unit_ids: tuple[str, ...]
    label: str = ""


@dataclass(frozen=True, slots=True)
class Observation:
    """Immutable faction-specific observation about a unit."""

    observer_faction: Faction
    unit_id: str
    last_known_position: Position | None = None
    estimated_strength: int | None = None
    visibility: VisibilityLevel = VisibilityLevel.HIDDEN
    notes: str = ""


@dataclass(slots=True)
class ActionCommand:
    """Structured command emitted by an agent for engine execution."""

    unit_id: str
    action_type: ActionType
    target_hex: Position | None = None
    posture: Posture | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    terrain_bonus: float = 0.0


@dataclass(slots=True)
class CombatResult:
    """Outcome container returned by the combat resolver."""

    attacker_ids: list[str] = field(default_factory=list)
    defender_ids: list[str] = field(default_factory=list)
    casualties_by_unit: dict[str, int] = field(default_factory=dict)
    winner: Faction | None = None
    summary: str = ""


@dataclass(slots=True)
class TurnResult:
    """Per-turn result bundle for logging and orchestration."""

    turn: int
    actions: list[ActionCommand] = field(default_factory=list)
    combat: CombatResult | None = None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GameState:
    """Top-level tactical state container shared by engine layers."""

    turn: int = 0
    max_turns: int = 20
    units: dict[str, Unit] = field(default_factory=dict)
    forces: dict[str, Force] = field(default_factory=dict)
    terrain_by_hex: dict[Position, TerrainType] = field(default_factory=dict)
    observations: dict[Faction, list[Observation]] = field(default_factory=dict)
    victory_points: dict[Faction, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
