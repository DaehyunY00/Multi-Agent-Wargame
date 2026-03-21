"""Top-level simulation engine interface for turn-based wargaming."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import floor
from random import Random

from wargame.combat.lanchester import CombatResolver, LanchesterOutcome
from wargame.core import HexGrid, TerrainLibrary
from wargame.core.enums import ActionType, Faction, Posture, UnitStatus
from wargame.core.models import ActionCommand, CombatResult, GameState, Position, TurnResult, Unit

from .fog_of_war import FactionViewState, FogOfWarFilter
from .state_manager import StateManager

_CONTACT_RANGE = 1
_ATTACK_RANGE = 1
_SUPPORT_RANGE = 2

_OFFENSE_ACTION_FACTOR: dict[ActionType, float] = {
    ActionType.HOLD: 0.9,
    ActionType.MOVE: 0.8,
    ActionType.ATTACK: 1.25,
    ActionType.SUPPORT_BY_FIRE: 1.1,
    ActionType.RECON: 0.7,
    ActionType.WITHDRAW: 0.5,
}
_POSTURE_OFFENSE_FACTOR: dict[Posture, float] = {
    Posture.ATTACK: 1.15,
    Posture.DEFEND: 1.0,
    Posture.SUPPORT: 1.05,
    Posture.MANEUVER: 0.95,
    Posture.WITHDRAW: 0.75,
}
_POSTURE_DEFENSE_FACTOR: dict[Posture, float] = {
    Posture.ATTACK: 0.9,
    Posture.DEFEND: 1.15,
    Posture.SUPPORT: 1.0,
    Posture.MANEUVER: 0.95,
    Posture.WITHDRAW: 0.85,
}
_ACTION_EXPOSURE_FACTOR: dict[ActionType, float] = {
    ActionType.HOLD: 0.9,
    ActionType.MOVE: 1.0,
    ActionType.ATTACK: 1.2,
    ActionType.SUPPORT_BY_FIRE: 1.0,
    ActionType.RECON: 1.05,
    ActionType.WITHDRAW: 1.1,
}


@dataclass(frozen=True, slots=True)
class Engagement:
    """One localized combat cluster resolved independently this turn."""

    blue_unit_ids: tuple[str, ...]
    red_unit_ids: tuple[str, ...]


@dataclass(slots=True)
class SimulationEngine:
    """Simulation engine coordinating state updates and localized combat."""

    state_manager: StateManager
    combat_resolver: CombatResolver
    fog_of_war: FogOfWarFilter
    terrain_library: TerrainLibrary = field(default_factory=TerrainLibrary.default)
    grid: HexGrid | None = None
    combat_seed: int | None = None
    turn_log: list[TurnResult] = field(default_factory=list)
    _combat_rng: Random | None = field(init=False, default=None, repr=False)
    _pending_actions: dict[str, ActionCommand] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.state_manager.fog_of_war = self.fog_of_war
        self.set_combat_seed(self.combat_seed)

    def set_combat_seed(self, seed: int | None) -> None:
        """Configure the combat RNG for reproducible stochastic resolution."""

        self.combat_seed = seed
        self._combat_rng = Random(seed) if seed is not None else None

    def get_state(self, faction: Faction) -> FactionViewState:
        """Return the state visible to the requested faction."""

        return self.state_manager.visible_state(faction)

    def execute_actions(self, actions: list[ActionCommand]) -> TurnResult:
        """Apply validated actions to the underlying game state.

        Movement actions advance at most one hex per turn when a grid is
        configured. Fire actions preserve the current position and only affect
        later combat resolution.
        """

        state = self.state_manager.current_state()
        notes: list[str] = []

        for action in actions:
            unit = state.units.get(action.unit_id)
            if unit is None:
                notes.append(f"Skipped unknown unit {action.unit_id}.")
                continue
            if unit.status is UnitStatus.DESTROYED or unit.strength <= 0:
                notes.append(f"Skipped destroyed unit {action.unit_id}.")
                continue

            self._pending_actions[action.unit_id] = deepcopy(action)

            if action.posture is not None:
                unit.posture = action.posture

            if action.action_type in {ActionType.MOVE, ActionType.RECON, ActionType.WITHDRAW}:
                destination = self._resolve_movement_destination(state=state, unit=unit, action=action)
                if destination != unit.position:
                    unit.position = destination
                    notes.append(
                        f"{unit.unit_id} advanced to ({destination.q},{destination.r}) via {action.action_type.value}."
                    )
            elif action.action_type in {ActionType.ATTACK, ActionType.SUPPORT_BY_FIRE}:
                notes.append(
                    f"{unit.unit_id} committed to {action.action_type.value} toward "
                    f"{_format_position(action.target_hex)}."
                )

        self.state_manager.update(state)
        return TurnResult(
            turn=state.turn,
            actions=deepcopy(actions),
            notes=notes,
            metadata={"phase": "action_execution"},
        )

    def resolve_combat(self) -> TurnResult:
        """Resolve localized combat outcomes for the current turn."""

        state = self.state_manager.current_state()
        engagements = self._build_engagements(state)

        if not engagements:
            self._pending_actions.clear()
            combat = CombatResult(
                summary="No localized engagements met contact or fire criteria this turn.",
            )
            return TurnResult(
                turn=state.turn,
                combat=combat,
                notes=["Combat skipped."],
                metadata={"phase": "combat", "engagement_count": 0, "engagements": []},
            )

        all_casualties: dict[str, int] = {}
        engagement_summaries: list[dict[str, object]] = []
        blue_remaining_total = 0.0
        red_remaining_total = 0.0
        blue_start_total = 0.0
        red_start_total = 0.0

        for engagement in engagements:
            blue_units = [state.units[unit_id] for unit_id in engagement.blue_unit_ids]
            red_units = [state.units[unit_id] for unit_id in engagement.red_unit_ids]
            blue_strength = sum(self._effective_offense(unit) for unit in blue_units)
            red_strength = sum(self._effective_offense(unit) for unit in red_units)
            blue_start_total += blue_strength
            red_start_total += red_strength

            blue_defense_modifier = self._defense_modifier_for_units(state, blue_units)
            red_defense_modifier = self._defense_modifier_for_units(state, red_units)
            outcome = self.combat_resolver.resolve(
                blue_strength,
                red_strength,
                blue_defense_modifier=blue_defense_modifier,
                red_defense_modifier=red_defense_modifier,
                rng=self._combat_rng,
            )

            blue_losses = _allocate_losses(
                int(round(outcome.blue_loss)),
                blue_units,
                exposure_weights={unit.unit_id: self._casualty_exposure(unit) for unit in blue_units},
            )
            red_losses = _allocate_losses(
                int(round(outcome.red_loss)),
                red_units,
                exposure_weights={unit.unit_id: self._casualty_exposure(unit) for unit in red_units},
            )

            for unit_id, loss in {**blue_losses, **red_losses}.items():
                all_casualties[unit_id] = all_casualties.get(unit_id, 0) + loss
                unit = state.units[unit_id]
                unit.strength = max(0, unit.strength - loss)
                if unit.strength == 0:
                    unit.status = UnitStatus.DESTROYED

            blue_remaining_total += outcome.blue_remaining
            red_remaining_total += outcome.red_remaining
            engagement_summaries.append(
                {
                    "blue_unit_ids": list(engagement.blue_unit_ids),
                    "red_unit_ids": list(engagement.red_unit_ids),
                    "blue_strength": round(blue_strength, 4),
                    "red_strength": round(red_strength, 4),
                    "blue_losses": sum(blue_losses.values()),
                    "red_losses": sum(red_losses.values()),
                    "outcome": _serialize_outcome(outcome),
                }
            )

        self._pending_actions.clear()
        self.state_manager.update(state)

        combat = CombatResult(
            attacker_ids=sorted({unit_id for engagement in engagements for unit_id in engagement.blue_unit_ids}),
            defender_ids=sorted({unit_id for engagement in engagements for unit_id in engagement.red_unit_ids}),
            casualties_by_unit=all_casualties,
            winner=_determine_winner_from_totals(blue_remaining_total, red_remaining_total),
            summary=(
                f"Resolved {len(engagements)} localized engagement(s); "
                f"Blue losses={sum(loss for unit_id, loss in all_casualties.items() if state.units[unit_id].faction is Faction.BLUE)}, "
                f"Red losses={sum(loss for unit_id, loss in all_casualties.items() if state.units[unit_id].faction is Faction.RED)}."
            ),
        )
        return TurnResult(
            turn=state.turn,
            combat=combat,
            notes=["Combat resolved."],
            metadata={
                "phase": "combat",
                "engagement_count": len(engagements),
                "engagements": engagement_summaries,
                "aggregate_outcome": {
                    "blue_start": round(blue_start_total, 4),
                    "red_start": round(red_start_total, 4),
                    "blue_remaining": round(blue_remaining_total, 4),
                    "red_remaining": round(red_remaining_total, 4),
                },
            },
        )

    def advance_turn(self) -> TurnResult:
        """Advance the simulation to the next turn."""

        self._pending_actions.clear()
        metadata = self.state_manager.advance_turn()
        return TurnResult(
            turn=metadata.turn,
            notes=["Turn advanced."],
            metadata={"phase": "advance_turn"},
        )

    def is_terminal(self) -> bool:
        """Return whether the current scenario has reached a terminal state."""

        state = self.state_manager.current_state()
        blue_strength = sum(unit.strength for unit in state.units.values() if unit.faction == Faction.BLUE)
        red_strength = sum(unit.strength for unit in state.units.values() if unit.faction == Faction.RED)
        return state.turn >= state.max_turns or blue_strength <= 0 or red_strength <= 0

    def get_log(self) -> list[TurnResult]:
        """Return the accumulated turn log."""

        return deepcopy(self.turn_log)

    def record_turn(self, result: TurnResult) -> None:
        """Store a completed turn result in the engine log."""

        self.turn_log.append(deepcopy(result))

    def _resolve_movement_destination(
        self,
        *,
        state: GameState,
        unit: Unit,
        action: ActionCommand,
    ) -> Position:
        """Move directly or one step depending on grid availability."""

        if action.target_hex is None:
            return unit.position
        if self.grid is None:
            return action.target_hex

        neighbors = sorted(self.grid.neighbors(unit.position), key=lambda item: (item.q, item.r))
        if action.target_hex in neighbors:
            return action.target_hex
        if not neighbors:
            return unit.position

        if action.action_type is ActionType.WITHDRAW:
            ranked = sorted(
                neighbors,
                key=lambda neighbor: (
                    -_hex_distance(neighbor, action.target_hex),
                    self._movement_cost(state, neighbor),
                    neighbor.q,
                    neighbor.r,
                ),
            )
            return ranked[0]

        ranked = sorted(
            neighbors,
            key=lambda neighbor: (
                _hex_distance(neighbor, action.target_hex),
                self._movement_cost(state, neighbor),
                neighbor.q,
                neighbor.r,
            ),
        )
        best = ranked[0]
        if _hex_distance(best, action.target_hex) >= _hex_distance(unit.position, action.target_hex):
            return unit.position
        return best

    def _movement_cost(self, state: GameState, end: Position) -> float:
        """Return the movement cost of entering a destination hex."""

        terrain = state.terrain_by_hex.get(end)
        if terrain is None:
            return 1.0
        return self.terrain_library.get_modifier(terrain).movement_cost

    def _build_engagements(self, state: GameState) -> list[Engagement]:
        """Build localized engagement components from contact and fire intent."""

        living_units = {
            unit.unit_id: unit
            for unit in state.units.values()
            if unit.strength > 0 and unit.status is not UnitStatus.DESTROYED
        }
        pair_graph: dict[str, set[str]] = {unit_id: set() for unit_id in living_units}
        blue_units = [unit for unit in living_units.values() if unit.faction is Faction.BLUE]
        red_units = [unit for unit in living_units.values() if unit.faction is Faction.RED]

        for blue_unit in blue_units:
            for red_unit in red_units:
                if _hex_distance(blue_unit.position, red_unit.position) <= _CONTACT_RANGE:
                    pair_graph[blue_unit.unit_id].add(red_unit.unit_id)
                    pair_graph[red_unit.unit_id].add(blue_unit.unit_id)

        for unit_id, action in self._pending_actions.items():
            unit = living_units.get(unit_id)
            if unit is None or action.target_hex is None:
                continue
            fire_range = _fire_range_for_action(action.action_type)
            if fire_range <= 0:
                continue
            for enemy in living_units.values():
                if enemy.faction is unit.faction:
                    continue
                if _hex_distance(unit.position, enemy.position) <= fire_range and _hex_distance(
                    action.target_hex,
                    enemy.position,
                ) <= 1:
                    pair_graph[unit.unit_id].add(enemy.unit_id)
                    pair_graph[enemy.unit_id].add(unit.unit_id)

        engagements: list[Engagement] = []
        visited: set[str] = set()
        for unit_id in sorted(pair_graph):
            if unit_id in visited or not pair_graph[unit_id]:
                continue
            stack = [unit_id]
            component: set[str] = set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                stack.extend(sorted(pair_graph[current] - visited))

            blue_component = tuple(sorted(unit for unit in component if living_units[unit].faction is Faction.BLUE))
            red_component = tuple(sorted(unit for unit in component if living_units[unit].faction is Faction.RED))
            if blue_component and red_component:
                engagements.append(Engagement(blue_component, red_component))
        return engagements

    def _effective_offense(self, unit: Unit) -> float:
        """Compute one unit's offensive contribution for this combat step."""

        action = self._pending_actions.get(unit.unit_id)
        action_type = action.action_type if action is not None else ActionType.HOLD
        posture = unit.posture
        return (
            unit.strength
            * max(unit.combat_power, 0.0)
            * max(unit.supply, 0.0)
            * max(unit.morale, 0.0)
            * _OFFENSE_ACTION_FACTOR[action_type]
            * _POSTURE_OFFENSE_FACTOR[posture]
        )

    def _casualty_exposure(self, unit: Unit) -> float:
        """Compute how much of a unit's strength is exposed to incoming losses."""

        action = self._pending_actions.get(unit.unit_id)
        action_type = action.action_type if action is not None else ActionType.HOLD
        return max(unit.strength * _ACTION_EXPOSURE_FACTOR[action_type], 1.0)

    def _defense_modifier_for_units(self, state: GameState, units: list[Unit]) -> float:
        """Compute a strength-weighted terrain and posture defense modifier."""

        total_strength = sum(unit.strength for unit in units)
        if total_strength <= 0:
            return 1.0

        weighted_modifier = 0.0
        for unit in units:
            terrain = state.terrain_by_hex.get(unit.position)
            terrain_modifier = 1.0
            if terrain is not None:
                terrain_modifier = self.terrain_library.get_modifier(terrain).defense_modifier
            weighted_modifier += (
                terrain_modifier
                * _POSTURE_DEFENSE_FACTOR[unit.posture]
                * unit.strength
            )
        return weighted_modifier / total_strength


def _fire_range_for_action(action_type: ActionType) -> int:
    """Return the effective combat range for a fire-oriented action."""

    if action_type is ActionType.ATTACK:
        return _ATTACK_RANGE
    if action_type is ActionType.SUPPORT_BY_FIRE:
        return _SUPPORT_RANGE
    return 0


def _living_units(state: GameState, faction: Faction) -> list[Unit]:
    """Return combat-capable units for the requested faction."""

    return [
        unit
        for unit in state.units.values()
        if unit.faction == faction and unit.strength > 0 and unit.status is not UnitStatus.DESTROYED
    ]


def _allocate_losses(
    total_loss: int,
    units: list[Unit],
    *,
    exposure_weights: dict[str, float] | None = None,
) -> dict[str, int]:
    """Distribute integer losses across units using optional exposure weights."""

    if total_loss <= 0 or not units:
        return {unit.unit_id: 0 for unit in units}

    weights = exposure_weights or {unit.unit_id: float(unit.strength) for unit in units}
    total_weight = sum(max(weights.get(unit.unit_id, 0.0), 0.0) for unit in units)
    if total_weight <= 0:
        return {unit.unit_id: 0 for unit in units}

    raw_shares = {
        unit.unit_id: (total_loss * max(weights.get(unit.unit_id, 0.0), 0.0)) / total_weight
        for unit in units
    }
    losses = {
        unit_id: min(
            next(unit.strength for unit in units if unit.unit_id == unit_id),
            floor(share),
        )
        for unit_id, share in raw_shares.items()
    }
    remainder = total_loss - sum(losses.values())
    ranked_units = sorted(
        units,
        key=lambda unit: (raw_shares[unit.unit_id] - floor(raw_shares[unit.unit_id]), unit.unit_id),
        reverse=True,
    )
    for unit in ranked_units:
        if remainder <= 0:
            break
        if losses[unit.unit_id] < unit.strength:
            losses[unit.unit_id] += 1
            remainder -= 1

    return losses


def _determine_winner_from_totals(
    blue_remaining_total: float,
    red_remaining_total: float,
) -> Faction | None:
    """Return the side with more remaining localized combat power."""

    if blue_remaining_total > red_remaining_total:
        return Faction.BLUE
    if red_remaining_total > blue_remaining_total:
        return Faction.RED
    return None


def _serialize_outcome(outcome: LanchesterOutcome) -> dict[str, float | bool]:
    """Convert a combat outcome into a JSON-friendly metadata payload."""

    return {
        "blue_start": outcome.blue_start,
        "red_start": outcome.red_start,
        "blue_loss": outcome.blue_loss,
        "red_loss": outcome.red_loss,
        "blue_remaining": outcome.blue_remaining,
        "red_remaining": outcome.red_remaining,
        "blue_defense_modifier": outcome.blue_defense_modifier,
        "red_defense_modifier": outcome.red_defense_modifier,
        "stochastic": outcome.stochastic,
        "blue_noise": outcome.blue_noise,
        "red_noise": outcome.red_noise,
    }


def _hex_distance(start: Position, end: Position) -> int:
    """Compute axial hex distance without requiring a grid instance."""

    dq = end.q - start.q
    dr = end.r - start.r
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _format_position(position: Position | None) -> str:
    """Render a position for execution notes."""

    if position is None:
        return "unknown"
    return f"({position.q},{position.r})"
