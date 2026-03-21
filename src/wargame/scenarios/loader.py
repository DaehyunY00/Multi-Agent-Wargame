"""YAML-backed scenario loading and GameState conversion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from wargame.core.enums import Faction, Posture, TerrainType, UnitStatus
from wargame.core.hexgrid import HexGrid
from wargame.core.models import GameState, Position, Unit

from .schema import MapSpec, ObjectiveSpec, ScenarioSpec, ScenarioUnitSpec


def load_scenario(path: str | Path) -> ScenarioSpec:
    """Load and validate a scenario definition from disk."""

    scenario_path = Path(path)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    raw_data = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_data, dict):
        raise ValueError("Scenario YAML must define a top-level mapping.")

    map_payload = raw_data.get("map", {})
    if not isinstance(map_payload, dict):
        raise ValueError("'map' must be a mapping.")

    terrain_entries = raw_data.get("terrain", map_payload.get("terrain", []))
    scenario = ScenarioSpec(
        scenario_id=_require_str(raw_data, "id"),
        name=_require_str(raw_data, "name"),
        description=str(raw_data.get("description", "")),
        map=MapSpec(
            width=_require_int(map_payload, "width", default=20),
            height=_require_int(map_payload, "height", default=15),
            terrain=_parse_terrain_entries(terrain_entries),
        ),
        forces=_parse_force_entries(raw_data.get("forces", [])),
        objectives=_parse_objective_entries(raw_data.get("objectives", [])),
        max_turns=_require_int(raw_data, "max_turns", default=20),
        metadata=dict(raw_data.get("metadata", {})) if isinstance(raw_data.get("metadata"), dict) else {},
    )
    return scenario


def build_grid(scenario: ScenarioSpec) -> HexGrid:
    """Build the tactical grid for a loaded scenario."""

    return HexGrid(width=scenario.map.width, height=scenario.map.height)


def scenario_to_game_state(scenario: ScenarioSpec) -> GameState:
    """Convert a scenario definition into an initial engine GameState."""

    mission = {
        faction.value: [
            objective.description
            for objective in scenario.objectives
            if objective.faction is faction
        ]
        for faction in (Faction.BLUE, Faction.RED)
    }
    objective_hexes = {
        faction.value: [
            {"q": objective.target_hex.q, "r": objective.target_hex.r}
            for objective in scenario.objectives
            if objective.faction is faction and objective.target_hex is not None
        ]
        for faction in (Faction.BLUE, Faction.RED)
    }

    return GameState(
        turn=0,
        max_turns=scenario.max_turns,
        units={
            unit_spec.unit_id: Unit(
                unit_id=unit_spec.unit_id,
                faction=unit_spec.faction,
                position=unit_spec.position,
                strength=unit_spec.strength,
                combat_power=unit_spec.combat_power,
                supply=unit_spec.supply,
                morale=unit_spec.morale,
                posture=unit_spec.posture,
                status=unit_spec.status,
            )
            for unit_spec in scenario.forces
        },
        terrain_by_hex=dict(scenario.map.terrain),
        metadata={
            **scenario.metadata,
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "scenario_description": scenario.description,
            "mission": mission,
            "objective_hexes": objective_hexes,
        },
    )


def _parse_terrain_entries(raw_entries: Any) -> dict[Position, TerrainType]:
    """Parse terrain cells from the YAML list or mapping representation."""

    if raw_entries is None:
        return {}
    if isinstance(raw_entries, dict):
        return {
            _parse_position_key(key): _parse_terrain_type(value)
            for key, value in raw_entries.items()
        }
    if not isinstance(raw_entries, list):
        raise ValueError("'terrain' must be a list or mapping.")

    terrain: dict[Position, TerrainType] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise ValueError("Each terrain entry must be a mapping.")
        position = _parse_position(entry.get("position", entry))
        terrain_type = _parse_terrain_type(entry.get("type"))
        terrain[position] = terrain_type
    return terrain


def _parse_force_entries(raw_entries: Any) -> list[ScenarioUnitSpec]:
    """Parse initial unit placements from YAML."""

    if raw_entries is None:
        return []
    if not isinstance(raw_entries, list):
        raise ValueError("'forces' must be a list.")

    units: list[ScenarioUnitSpec] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise ValueError("Each force entry must be a mapping.")
        units.append(
            ScenarioUnitSpec(
                unit_id=_require_str(entry, "unit_id"),
                faction=Faction(_require_str(entry, "faction")),
                position=_parse_position(entry.get("position")),
                strength=_require_int(entry, "strength"),
                combat_power=float(entry.get("combat_power", 1.0)),
                supply=float(entry.get("supply", 1.0)),
                morale=float(entry.get("morale", 1.0)),
                posture=Posture(str(entry.get("posture", Posture.DEFEND.value))),
                status=UnitStatus(str(entry.get("status", UnitStatus.READY.value))),
            )
        )
    return units


def _parse_objective_entries(raw_entries: Any) -> list[ObjectiveSpec]:
    """Parse objective definitions from YAML."""

    if raw_entries is None:
        return []
    if not isinstance(raw_entries, list):
        raise ValueError("'objectives' must be a list.")

    objectives: list[ObjectiveSpec] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise ValueError("Each objective entry must be a mapping.")
        target = entry.get("target_hex")
        objectives.append(
            ObjectiveSpec(
                faction=Faction(_require_str(entry, "faction")),
                description=_require_str(entry, "description"),
                target_hex=_parse_position(target) if target is not None else None,
                priority=_require_int(entry, "priority", default=1),
            )
        )
    return objectives


def _require_str(payload: dict[str, Any], key: str, default: str | None = None) -> str:
    """Require a string field from the YAML payload."""

    value = payload.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key!r} must be a string.")
    return value


def _require_int(payload: dict[str, Any], key: str, default: int | None = None) -> int:
    """Require an integer field from the YAML payload."""

    value = payload.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"{key!r} must be an integer.")
    return value


def _parse_position(raw_value: Any) -> Position:
    """Parse a position mapping into an axial hex coordinate."""

    if not isinstance(raw_value, dict):
        raise ValueError("Positions must be mappings with 'q' and 'r'.")
    q = raw_value.get("q")
    r = raw_value.get("r")
    if not isinstance(q, int) or not isinstance(r, int):
        raise ValueError("Position q and r must be integers.")
    return Position(q=q, r=r)


def _parse_position_key(raw_value: Any) -> Position:
    """Parse one terrain dictionary key into a position."""

    if isinstance(raw_value, str):
        q_text, r_text = raw_value.split(",", maxsplit=1)
        return Position(q=int(q_text), r=int(r_text))
    raise ValueError("Terrain mapping keys must use 'q,r' string form.")


def _parse_terrain_type(raw_value: Any) -> TerrainType:
    """Parse one terrain enum value."""

    if not isinstance(raw_value, str):
        raise ValueError("Terrain type must be a string.")
    return TerrainType(raw_value)
