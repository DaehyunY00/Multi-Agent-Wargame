"""Hex-grid helpers for the tactical map model."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Position

AXIAL_DIRECTIONS: tuple[Position, ...] = (
    Position(1, 0),
    Position(1, -1),
    Position(0, -1),
    Position(-1, 0),
    Position(-1, 1),
    Position(0, 1),
)


@dataclass(frozen=True, slots=True)
class HexGrid:
    """Rectangular axial hex-grid with bounded neighbor and distance helpers."""

    width: int
    height: int
    directions: tuple[Position, ...] = field(default=AXIAL_DIRECTIONS)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("HexGrid width and height must be positive integers.")

    def is_within_bounds(self, coord: Position) -> bool:
        """Return whether the given coordinate is valid for this grid."""

        return 0 <= coord.q < self.width and 0 <= coord.r < self.height

    def neighbors(self, coord: Position) -> list[Position]:
        """Return bounded adjacent coordinates for the given hex."""

        return [
            Position(coord.q + delta.q, coord.r + delta.r)
            for delta in self.directions
            if self.is_within_bounds(Position(coord.q + delta.q, coord.r + delta.r))
        ]

    def distance(self, start: Position, end: Position) -> int:
        """Return the axial hex distance between two coordinates."""

        dq = end.q - start.q
        dr = end.r - start.r
        return (abs(dq) + abs(dr) + abs(dq + dr)) // 2
