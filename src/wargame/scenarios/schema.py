"""Scenario schema placeholders aligned with the research plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wargame.core.enums import Faction, Posture, TerrainType, UnitStatus
from wargame.core.models import HexCoord


@dataclass(slots=True)
class MapSpec:
    """Static scenario map definition."""

    width: int = 20
    height: int = 15
    terrain: dict[HexCoord, TerrainType] = field(default_factory=dict)


@dataclass(slots=True)
class ScenarioUnitSpec:
    """Initial unit placement and status for a scenario."""

    unit_id: str
    faction: Faction
    position: HexCoord
    strength: int
    combat_power: float = 1.0
    supply: float = 1.0
    morale: float = 1.0
    posture: Posture = Posture.DEFEND
    status: UnitStatus = UnitStatus.READY


@dataclass(slots=True)
class ObjectiveSpec:
    """Scenario objective metadata for a faction."""

    faction: Faction
    description: str
    target_hex: HexCoord | None = None
    priority: int = 1


@dataclass(slots=True)
class ScenarioSpec:
    """Top-level scenario definition container."""

    scenario_id: str
    name: str
    description: str = ""
    map: MapSpec = field(default_factory=MapSpec)
    forces: list[ScenarioUnitSpec] = field(default_factory=list)
    objectives: list[ObjectiveSpec] = field(default_factory=list)
    max_turns: int = 20
    metadata: dict[str, Any] = field(default_factory=dict)
