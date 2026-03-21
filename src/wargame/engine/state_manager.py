"""Canonical full-state management with faction-specific view export."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from wargame.core.enums import Faction
from wargame.core.models import GameState

from .fog_of_war import FactionViewState, FogOfWarFilter, TurnMetadata


@dataclass(slots=True)
class StateManager:
    """Own the canonical full state and export stable snapshots."""

    initial_state: GameState
    fog_of_war: FogOfWarFilter = field(default_factory=FogOfWarFilter)
    _state: GameState = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._state = deepcopy(self.initial_state)

    def current_state(self) -> GameState:
        """Return a stable snapshot of the canonical full state."""

        return deepcopy(self._state)

    def snapshot(self) -> GameState:
        """Return a safe snapshot of the current state."""

        return self.current_state()

    def update(self, next_state: GameState) -> None:
        """Replace the managed state after a turn update."""

        self._state = deepcopy(next_state)

    def visible_state(self, faction: Faction) -> FactionViewState:
        """Return the faction-specific visible state after fog-of-war filtering."""

        return self.fog_of_war.filter_state(self._state, faction)

    def turn_metadata(self) -> TurnMetadata:
        """Return machine-friendly turn progression metadata."""

        return TurnMetadata(turn=self._state.turn, max_turns=self._state.max_turns)

    def advance_turn(self) -> TurnMetadata:
        """Increment the internal turn counter and return updated metadata."""

        self._state.turn += 1
        return self.turn_metadata()
