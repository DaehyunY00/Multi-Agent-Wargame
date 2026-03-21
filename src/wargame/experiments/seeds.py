"""Seed management helpers for repeatable experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """Derived per-component seeds used within one experiment run."""

    base_seed: int | None = None
    engine_seed: int | None = None
    blue_agent_seed: int | None = None
    red_agent_seed: int | None = None
    white_cell_seed: int | None = None

    def as_dict(self) -> dict[str, int | None]:
        """Return a JSON-friendly representation of the seed bundle."""

        return {
            "base_seed": self.base_seed,
            "engine_seed": self.engine_seed,
            "blue_agent_seed": self.blue_agent_seed,
            "red_agent_seed": self.red_agent_seed,
            "white_cell_seed": self.white_cell_seed,
        }


def get_seed_sequence(count: int | None = None) -> tuple[int, ...]:
    """Return a stable tuple of experiment seeds."""

    if count is None:
        return DEFAULT_SEEDS
    if count < 0:
        raise ValueError("count must be non-negative.")
    return tuple(range(count))


def derive_seed_bundle(base_seed: int | None) -> SeedBundle:
    """Derive deterministic per-component seeds from one experiment seed."""

    if base_seed is None:
        return SeedBundle()
    return SeedBundle(
        base_seed=base_seed,
        engine_seed=base_seed * 10 + 1,
        blue_agent_seed=base_seed * 10 + 2,
        red_agent_seed=base_seed * 10 + 3,
        white_cell_seed=base_seed * 10 + 4,
    )


def apply_seed_bundle(turn_loop: Any, seed_bundle: SeedBundle) -> None:
    """Apply derived seeds to the engine and any seed-aware agents."""

    engine = getattr(turn_loop, "engine", None)
    if engine is not None and hasattr(engine, "set_combat_seed"):
        engine.set_combat_seed(seed_bundle.engine_seed)

    _seed_component(getattr(turn_loop, "blue_agent", None), seed_bundle.blue_agent_seed)
    _seed_component(getattr(turn_loop, "red_agent", None), seed_bundle.red_agent_seed)
    _seed_component(getattr(turn_loop, "white_cell", None), seed_bundle.white_cell_seed)


def _seed_component(component: Any, seed: int | None) -> None:
    """Seed one experiment component when it exposes a reset hook or seed field."""

    if component is None or seed is None:
        return
    if hasattr(component, "reset_seed"):
        component.reset_seed(seed)
        return
    if hasattr(component, "seed"):
        setattr(component, "seed", seed)
