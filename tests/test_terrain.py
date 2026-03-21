"""Tests for configurable terrain modifier lookup."""

from wargame.core import TerrainLibrary, TerrainType


def test_default_terrain_lookup_returns_explicit_values() -> None:
    """Default terrain modifiers should expose the configured lookup table."""

    library = TerrainLibrary.default()

    open_modifier = library.get_modifier(TerrainType.OPEN)
    mountain_modifier = library.get_modifier(TerrainType.MOUNTAIN)

    assert open_modifier.movement_cost == 1.0
    assert open_modifier.defense_modifier == 1.0
    assert mountain_modifier.movement_cost == 2.0
    assert mountain_modifier.defense_modifier == 1.5
