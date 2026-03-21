"""Tests for the core axial hex-grid implementation."""

from wargame.core import HexGrid, Position


def test_center_hex_has_six_neighbors() -> None:
    """Interior hexes should expose all six adjacent axial neighbors."""

    grid = HexGrid(width=5, height=5)

    neighbors = grid.neighbors(Position(2, 2))

    assert len(neighbors) == 6
    assert Position(3, 2) in neighbors
    assert Position(2, 3) in neighbors


def test_corner_hex_has_two_neighbors() -> None:
    """Corner hexes should only return in-bounds neighbors."""

    grid = HexGrid(width=5, height=5)

    assert grid.neighbors(Position(0, 0)) == [Position(1, 0), Position(0, 1)]


def test_hex_distance_is_symmetric() -> None:
    """Distance should be identical regardless of traversal direction."""

    grid = HexGrid(width=10, height=10)
    start = Position(1, 2)
    end = Position(6, 4)

    assert grid.distance(start, end) == grid.distance(end, start)
    assert grid.distance(start, start) == 0
