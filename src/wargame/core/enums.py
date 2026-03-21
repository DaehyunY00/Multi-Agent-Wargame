"""Shared enumerations used across the tactical wargame skeleton."""

from enum import StrEnum


class Faction(StrEnum):
    """Faction identifiers used throughout the simulation."""

    BLUE = "blue"
    RED = "red"
    WHITE = "white"


class TerrainType(StrEnum):
    """Supported terrain categories from the research plan."""

    OPEN = "open"
    MOUNTAIN = "mountain"
    URBAN = "urban"
    FOREST = "forest"
    RIVER = "river"


class ActionType(StrEnum):
    """High-level action types available to agents."""

    HOLD = "hold"
    MOVE = "move"
    ATTACK = "attack"
    SUPPORT_BY_FIRE = "support_by_fire"
    RECON = "recon"
    WITHDRAW = "withdraw"


class Posture(StrEnum):
    """Combat posture or intent carried with an action."""

    ATTACK = "attack"
    DEFEND = "defend"
    SUPPORT = "support"
    MANEUVER = "maneuver"
    WITHDRAW = "withdraw"


class VisibilityLevel(StrEnum):
    """Visibility confidence levels for fog-of-war filtered observations."""

    HIDDEN = "hidden"
    DETECTED = "detected"
    IDENTIFIED = "identified"


class UnitStatus(StrEnum):
    """Coarse unit readiness states for tactical entities."""

    READY = "ready"
    SUPPRESSED = "suppressed"
    DISRUPTED = "disrupted"
    DESTROYED = "destroyed"
