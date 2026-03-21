"""Tests for core tactical dataclass construction."""

from wargame.core import (
    Faction,
    Force,
    GameState,
    Observation,
    Position,
    TerrainType,
    Unit,
)


def test_core_dataclasses_construct_with_expected_values() -> None:
    """Core domain objects should be easy to instantiate and compose."""

    position = Position(q=3, r=4)
    unit = Unit(unit_id="blue-1", faction=Faction.BLUE, position=position, strength=120)
    force = Force(force_id="blue-main", faction=Faction.BLUE, unit_ids=("blue-1",))
    observation = Observation(
        observer_faction=Faction.BLUE,
        unit_id="red-1",
        last_known_position=Position(q=5, r=5),
    )
    state = GameState(
        turn=1,
        max_turns=15,
        units={"blue-1": unit},
        forces={"blue-main": force},
        terrain_by_hex={position: TerrainType.OPEN},
        observations={Faction.BLUE: [observation]},
    )

    assert unit.position == position
    assert force.unit_ids == ("blue-1",)
    assert observation.last_known_position == Position(5, 5)
    assert state.units["blue-1"].strength == 120
    assert state.forces["blue-main"].faction == Faction.BLUE
    assert state.terrain_by_hex[position] == TerrainType.OPEN
