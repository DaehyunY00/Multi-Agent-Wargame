"""Shared helpers for transparent baseline tactical agents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from random import Random

from wargame.core.enums import Faction, Posture, UnitStatus, VisibilityLevel
from wargame.core.hexgrid import HexGrid
from wargame.core.models import Position
from wargame.engine.fog_of_war import FactionViewState, ObservedUnit

_FRIENDLY_RE = re.compile(
    r"^- (?P<unit_id>[^:]+): position=\((?P<q>-?\d+),(?P<r>-?\d+)\), "
    r"strength=(?P<strength>\d+), posture=(?P<posture>[a-z_]+), status=(?P<status>[a-z_]+)$"
)
_ENEMY_RE = re.compile(
    r"^- (?P<unit_id>[^:]+): visibility=(?P<visibility>[a-z_]+), "
    r"position=(?P<position>unknown|\((-?\d+),(-?\d+)\)), "
    r"(?P<strength_key>strength|estimated_strength)=(?P<strength>\d+)"
    r"(?:, range=(?P<range_min>\d+)-(?P<range_max>\d+))?, "
    r"observed_turn=(?P<observed_turn>\d+)"
    r"(?:, posture=(?P<posture>[a-z_]+))?"
    r"(?:, status=(?P<status>[a-z_]+))?$"
)


@dataclass(frozen=True, slots=True)
class FriendlyUnitReport:
    """Parsed friendly unit entry from a tactical report."""

    unit_id: str
    position: Position
    strength: int
    posture: Posture
    status: UnitStatus


@dataclass(frozen=True, slots=True)
class EnemyContactReport:
    """Parsed enemy contact entry from a tactical report."""

    unit_id: str
    visibility: VisibilityLevel
    position: Position | None
    estimated_strength: int
    strength_range: tuple[int, int] | None
    observed_turn: int
    posture: Posture | None = None
    status: UnitStatus | None = None


@dataclass(frozen=True, slots=True)
class TacticalReport:
    """Structured view reconstructed from a deterministic text report."""

    faction: Faction
    turn: int
    max_turns: int
    friendly_units: dict[str, FriendlyUnitReport] = field(default_factory=dict)
    enemy_contacts: dict[str, EnemyContactReport] = field(default_factory=dict)


def parse_tactical_report(state_text: str) -> TacticalReport:
    """Parse the deterministic tactical report emitted by `StateRenderer`."""

    faction: Faction | None = None
    turn = 0
    max_turns = 0
    section: str | None = None
    friendly_units: dict[str, FriendlyUnitReport] = {}
    enemy_contacts: dict[str, EnemyContactReport] = {}

    for raw_line in state_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Turn:"):
            turn_text = line.removeprefix("Turn:").strip()
            current_turn, max_turn = turn_text.split("/", maxsplit=1)
            turn = int(current_turn)
            max_turns = int(max_turn)
            continue
        if line.startswith("Faction:"):
            faction = Faction(line.removeprefix("Faction:").strip())
            continue
        if line == "Friendly Units:":
            section = "friendly"
            continue
        if line == "Enemy Contacts:":
            section = "enemy"
            continue
        if line in {"Terrain References:", "Metadata:"}:
            section = None
            continue
        if line == "- none":
            continue

        if section == "friendly":
            friendly = _parse_friendly_line(line)
            friendly_units[friendly.unit_id] = friendly
        elif section == "enemy":
            enemy = _parse_enemy_line(line)
            enemy_contacts[enemy.unit_id] = enemy

    if faction is None:
        raise ValueError("Tactical report is missing faction information.")

    return TacticalReport(
        faction=faction,
        turn=turn,
        max_turns=max_turns,
        friendly_units=friendly_units,
        enemy_contacts=enemy_contacts,
    )


def tactical_report_from_view(state_view: FactionViewState) -> TacticalReport:
    """Build the baseline-agent report directly from structured observed state."""

    return TacticalReport(
        faction=state_view.faction,
        turn=state_view.turn_metadata.turn,
        max_turns=state_view.turn_metadata.max_turns,
        friendly_units={
            unit_id: FriendlyUnitReport(
                unit_id=unit.unit_id,
                position=unit.position,
                strength=unit.strength,
                posture=unit.posture,
                status=unit.status,
            )
            for unit_id, unit in state_view.friendly_units.items()
        },
        enemy_contacts={
            unit_id: _enemy_contact_from_observation(observation)
            for unit_id, observation in state_view.enemy_observations.items()
        },
    )


def nearest_enemy_with_position(
    report: TacticalReport,
    *,
    from_position: Position,
) -> EnemyContactReport | None:
    """Return the nearest enemy contact with a known position."""

    contacts = [
        contact
        for contact in report.enemy_contacts.values()
        if contact.position is not None
    ]
    if not contacts:
        return None

    return min(
        contacts,
        key=lambda contact: (
            hex_distance(from_position, contact.position),
            contact.unit_id,
        ),
    )


def step_toward(
    grid: HexGrid,
    *,
    start: Position,
    target: Position,
    seed: int | str | None = None,
    context: str,
) -> Position | None:
    """Choose one valid neighboring step that decreases distance to the target."""

    current_distance = hex_distance(start, target)
    candidates = [
        neighbor
        for neighbor in sorted(grid.neighbors(start), key=lambda item: (item.q, item.r))
        if hex_distance(neighbor, target) < current_distance
    ]
    return _choose_position(candidates, seed=seed, context=context)


def step_away(
    grid: HexGrid,
    *,
    start: Position,
    threat: Position,
    seed: int | str | None = None,
    context: str,
) -> Position | None:
    """Choose one valid neighboring step that increases distance from a threat."""

    current_distance = hex_distance(start, threat)
    candidates = [
        neighbor
        for neighbor in sorted(grid.neighbors(start), key=lambda item: (item.q, item.r))
        if hex_distance(neighbor, threat) > current_distance
    ]
    return _choose_position(candidates, seed=seed, context=context)


def forward_step(
    grid: HexGrid,
    *,
    start: Position,
    faction: Faction | None,
    seed: int | str | None = None,
    context: str,
) -> Position | None:
    """Choose a simple forward movement step when no enemy is visible."""

    if faction is Faction.RED:
        desired_q = start.q - 1
    else:
        desired_q = start.q + 1

    candidates = [
        neighbor
        for neighbor in sorted(grid.neighbors(start), key=lambda item: (item.q, item.r))
        if (neighbor.q - start.q) == (desired_q - start.q)
    ]
    if not candidates:
        candidates = sorted(grid.neighbors(start), key=lambda item: (item.q, item.r))
    return _choose_position(candidates, seed=seed, context=context)


def flank_objective(enemy_position: Position, *, faction: Faction | None) -> Position:
    """Return a simple flank-oriented objective adjacent to an enemy position."""

    offset = 1 if faction is not Faction.RED else -1
    return Position(enemy_position.q, enemy_position.r + offset)


def hex_distance(start: Position, end: Position) -> int:
    """Compute axial hex distance."""

    dq = end.q - start.q
    dr = end.r - start.r
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _choose_position(
    candidates: list[Position],
    *,
    seed: int | str | None,
    context: str,
) -> Position | None:
    """Choose one candidate deterministically or with deterministic seeded tiebreaking."""

    if not candidates:
        return None
    if seed is None or len(candidates) == 1:
        return candidates[0]
    return Random(f"{seed}:{context}").choice(candidates)


def _parse_friendly_line(line: str) -> FriendlyUnitReport:
    """Parse one friendly unit line from the renderer output."""

    match = _FRIENDLY_RE.match(line)
    if match is None:
        raise ValueError(f"Unrecognized friendly unit line: {line!r}")
    return FriendlyUnitReport(
        unit_id=match.group("unit_id"),
        position=Position(int(match.group("q")), int(match.group("r"))),
        strength=int(match.group("strength")),
        posture=Posture(match.group("posture")),
        status=UnitStatus(match.group("status")),
    )


def _parse_enemy_line(line: str) -> EnemyContactReport:
    """Parse one enemy contact line from the renderer output."""

    match = _ENEMY_RE.match(line)
    if match is None:
        raise ValueError(f"Unrecognized enemy contact line: {line!r}")

    position_text = match.group("position")
    position = None if position_text == "unknown" else _parse_position_text(position_text)
    range_min = match.group("range_min")
    range_max = match.group("range_max")

    return EnemyContactReport(
        unit_id=match.group("unit_id"),
        visibility=VisibilityLevel(match.group("visibility")),
        position=position,
        estimated_strength=int(match.group("strength")),
        strength_range=(
            (int(range_min), int(range_max))
            if range_min is not None and range_max is not None
            else None
        ),
        observed_turn=int(match.group("observed_turn")),
        posture=Posture(match.group("posture")) if match.group("posture") is not None else None,
        status=UnitStatus(match.group("status")) if match.group("status") is not None else None,
    )


def _parse_position_text(value: str) -> Position:
    """Parse `(q,r)` text into a position."""

    q_text, r_text = value.strip("()").split(",", maxsplit=1)
    return Position(int(q_text), int(r_text))


def _enemy_contact_from_observation(observation: ObservedUnit) -> EnemyContactReport:
    """Normalize an observed enemy unit into the baseline report shape."""

    return EnemyContactReport(
        unit_id=observation.unit_id,
        visibility=observation.visibility,
        position=observation.last_known_position,
        estimated_strength=observation.estimated_strength or 0,
        strength_range=observation.strength_range,
        observed_turn=observation.observed_turn,
        posture=observation.posture,
        status=observation.status,
    )
