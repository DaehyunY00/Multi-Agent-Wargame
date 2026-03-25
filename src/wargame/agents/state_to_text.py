"""Deterministic rendering of observed state into tactical text reports."""

from __future__ import annotations

from dataclasses import dataclass

from wargame.core.models import Position, Unit
from wargame.engine.fog_of_war import FactionViewState, ObservedUnit


@dataclass(slots=True)
class StateRenderer:
    """Render observed state into a concise, deterministic tactical report."""

    include_metadata: bool = True
    max_terrain_entries: int = 8

    def render(self, state: FactionViewState) -> str:
        """Render a faction-specific observed state into stable plain text."""

        lines = [
            "TACTICAL REPORT",
            f"Turn: {state.turn_metadata.turn}/{state.turn_metadata.max_turns}",
            f"Faction: {state.faction.value}",
        ]

        mission_lines = self._render_mission(state)
        if mission_lines:
            lines.extend([
                "",
                "Mission:",
                *mission_lines,
            ])

        lines.extend([
            "",
            "Friendly Units:",
            *self._render_friendly_units(state.friendly_units),
            "",
            "IMPORTANT — You command ONLY the Friendly Units listed above.",
            f"Valid unit_ids for your orders: {', '.join(sorted(state.friendly_units.keys()))}",
            "Do NOT issue orders to Enemy Contact unit IDs. They are for observation only.",
            "",
            "Enemy Contacts:",
            *self._render_enemy_contacts(state.enemy_observations),
            "",
            "Terrain References:",
            *self._render_terrain(state),
        ])

        if self.include_metadata:
            lines.extend(["", "Metadata:", *self._render_metadata(state.metadata)])

        return "\n".join(lines)

    @staticmethod
    def _render_friendly_units(units: dict[str, Unit]) -> list[str]:
        if not units:
            return ["- none"]

        return [
            (
                f"- {unit.unit_id}: position={_format_position(unit.position)}, "
                f"strength={unit.strength}, posture={unit.posture.value}, "
                f"status={unit.status.value}"
            )
            for unit in sorted(units.values(), key=lambda item: item.unit_id)
        ]

    @staticmethod
    def _render_enemy_contacts(observations: dict[str, ObservedUnit]) -> list[str]:
        if not observations:
            return ["- none"]

        rendered: list[str] = []
        for observation in sorted(observations.values(), key=lambda item: item.unit_id):
            location = (
                _format_position(observation.last_known_position)
                if observation.last_known_position is not None
                else "unknown"
            )
            strength_text = (
                f"strength={observation.estimated_strength}"
                if observation.strength_range is None
                or (
                    observation.strength_range[0] == observation.strength_range[1]
                    and observation.estimated_strength is not None
                )
                else (
                    f"estimated_strength={observation.estimated_strength}, "
                    f"range={observation.strength_range[0]}-{observation.strength_range[1]}"
                )
            )
            details = [
                f"- {observation.unit_id}: visibility={observation.visibility.value}",
                f"position={location}",
                strength_text,
                f"observed_turn={observation.observed_turn}",
            ]
            if observation.posture is not None:
                details.append(f"posture={observation.posture.value}")
            if observation.status is not None:
                details.append(f"status={observation.status.value}")
            rendered.append(", ".join(details))
        return rendered

    @staticmethod
    def _render_mission(state: FactionViewState) -> list[str]:
        mission = state.metadata.get("mission")
        if isinstance(mission, dict):
            faction_mission = mission.get(state.faction.value)
            if isinstance(faction_mission, list):
                rendered = [f"- {item}" for item in faction_mission if isinstance(item, str)]
                if rendered:
                    return rendered

        scenario_name = state.metadata.get("scenario_name")
        if isinstance(scenario_name, str) and scenario_name:
            return [f"- Scenario: {scenario_name}"]
        return []

    def _render_terrain(self, state: FactionViewState) -> list[str]:
        terrain_by_hex = state.terrain_by_hex
        if not terrain_by_hex:
            return ["- none"]

        reference_positions = [
            unit.position
            for unit in state.friendly_units.values()
        ]
        reference_positions.extend(
            observation.last_known_position
            for observation in state.enemy_observations.values()
            if observation.last_known_position is not None
        )
        reference_positions.extend(_objective_positions(state.metadata, state.faction))

        entries = sorted(
            terrain_by_hex.items(),
            key=lambda item: (
                _distance_to_references(item[0], reference_positions),
                item[0].q,
                item[0].r,
            ),
        )
        if self.max_terrain_entries >= 0:
            entries = entries[: self.max_terrain_entries]

        return [
            f"- {_format_position(position)}: {getattr(terrain, 'value', terrain)}"
            for position, terrain in entries
        ]

    @staticmethod
    def _render_metadata(metadata: dict[str, object]) -> list[str]:
        visible_keys = [
            key
            for key in sorted(metadata)
            if key not in {"mission", "objective_hexes"}
        ]
        if not visible_keys:
            return ["- none"]

        return [f"- {key}={metadata[key]}" for key in visible_keys]


def _format_position(position: Position) -> str:
    """Render a position in a compact machine-friendly form."""

    return f"({position.q},{position.r})"


def _distance_to_references(position: Position, references: list[Position]) -> int:
    """Return the nearest hex distance from one terrain cell to reference points."""

    if not references:
        return 0
    return min(_hex_distance(position, reference) for reference in references)


def _hex_distance(start: Position, end: Position) -> int:
    """Compute axial hex distance for deterministic terrain prioritization."""

    dq = end.q - start.q
    dr = end.r - start.r
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _objective_positions(metadata: dict[str, object], faction) -> list[Position]:
    """Recover objective hexes for terrain prioritization."""

    raw_objectives = metadata.get("objective_hexes")
    if not isinstance(raw_objectives, dict):
        return []

    faction_entries = raw_objectives.get(getattr(faction, "value", faction))
    if not isinstance(faction_entries, list):
        return []

    positions: list[Position] = []
    for item in faction_entries:
        if not isinstance(item, dict):
            continue
        q = item.get("q")
        r = item.get("r")
        if isinstance(q, int) and isinstance(r, int):
            positions.append(Position(q, r))
    return positions
