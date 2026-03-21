"""Combat resolution helpers for the tactical wargame."""

from .lanchester import (
    CombatResolver,
    LanchesterConfig,
    LanchesterOutcome,
    LanchesterResolver,
    resolve_lanchester,
    resolve_lanchester_deterministic,
    resolve_lanchester_stochastic,
)

__all__ = [
    "CombatResolver",
    "LanchesterConfig",
    "LanchesterOutcome",
    "LanchesterResolver",
    "resolve_lanchester",
    "resolve_lanchester_deterministic",
    "resolve_lanchester_stochastic",
]
