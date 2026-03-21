"""Deterministic and stochastic Lanchester-style combat resolution.

This module intentionally stays engine-agnostic. It resolves a single combat
step from raw strength values and explicit defense modifiers, making it easy to
test in isolation and reuse from the engine later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isclose
from random import Random


@dataclass(frozen=True, slots=True)
class LanchesterConfig:
    """Parameters for a single Lanchester-style combat step.

    `blue_attrition_rate` is the per-turn firepower coefficient that blue
    applies against red. `red_attrition_rate` is the corresponding coefficient
    that red applies against blue.
    """

    blue_attrition_rate: float = 0.05
    red_attrition_rate: float = 0.05
    time_step: float = 1.0
    noise_std: float = 0.1
    stochastic: bool = False


@dataclass(frozen=True, slots=True)
class LanchesterOutcome:
    """Structured output from one combat step."""

    blue_start: float
    red_start: float
    blue_loss: float
    red_loss: float
    blue_remaining: float
    red_remaining: float
    blue_defense_modifier: float
    red_defense_modifier: float
    stochastic: bool
    blue_noise: float = 0.0
    red_noise: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("blue_start", self.blue_start),
            ("red_start", self.red_start),
            ("blue_loss", self.blue_loss),
            ("red_loss", self.red_loss),
            ("blue_remaining", self.blue_remaining),
            ("red_remaining", self.red_remaining),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")

        if self.blue_remaining > self.blue_start:
            raise ValueError("blue_remaining cannot exceed blue_start.")
        if self.red_remaining > self.red_start:
            raise ValueError("red_remaining cannot exceed red_start.")

        if not isclose(
            self.blue_start - self.blue_loss,
            self.blue_remaining,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Blue losses and remaining strength are inconsistent.")
        if not isclose(
            self.red_start - self.red_loss,
            self.red_remaining,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Red losses and remaining strength are inconsistent.")


class CombatResolver(ABC):
    """Abstract interface for engine-facing combat resolvers."""

    @abstractmethod
    def resolve(
        self,
        blue_strength: float,
        red_strength: float,
        *,
        blue_defense_modifier: float = 1.0,
        red_defense_modifier: float = 1.0,
        stochastic: bool | None = None,
        seed: int | None = None,
        rng: Random | None = None,
    ) -> LanchesterOutcome:
        """Resolve one combat step and return the structured outcome."""


def resolve_lanchester(
    blue_strength: float,
    red_strength: float,
    *,
    config: LanchesterConfig | None = None,
    blue_defense_modifier: float = 1.0,
    red_defense_modifier: float = 1.0,
    stochastic: bool | None = None,
    seed: int | None = None,
    rng: Random | None = None,
) -> LanchesterOutcome:
    """Resolve one combat step using a discrete Lanchester-style update.

    The update uses a square-law-inspired one-turn approximation:

    - blue_loss ~= red_attrition_rate * red_strength * time_step / blue_defense_modifier
    - red_loss  ~= blue_attrition_rate * blue_strength * time_step / red_defense_modifier

    Terrain is modeled as an explicit defensive modifier on the force occupying
    that terrain. A higher defense modifier reduces incoming losses.
    """

    config = config or LanchesterConfig()
    stochastic = config.stochastic if stochastic is None else stochastic

    _validate_inputs(
        blue_strength=blue_strength,
        red_strength=red_strength,
        blue_defense_modifier=blue_defense_modifier,
        red_defense_modifier=red_defense_modifier,
        config=config,
        seed=seed,
        rng=rng,
    )

    base_blue_loss = (
        config.red_attrition_rate * red_strength * config.time_step / blue_defense_modifier
    )
    base_red_loss = (
        config.blue_attrition_rate * blue_strength * config.time_step / red_defense_modifier
    )

    blue_noise = 0.0
    red_noise = 0.0

    if stochastic:
        local_rng = _coerce_rng(seed=seed, rng=rng)
        blue_noise = local_rng.gauss(0.0, config.noise_std * base_blue_loss)
        red_noise = local_rng.gauss(0.0, config.noise_std * base_red_loss)

    blue_loss = _clamp_loss(base_blue_loss + blue_noise, blue_strength)
    red_loss = _clamp_loss(base_red_loss + red_noise, red_strength)

    return LanchesterOutcome(
        blue_start=blue_strength,
        red_start=red_strength,
        blue_loss=blue_loss,
        red_loss=red_loss,
        blue_remaining=blue_strength - blue_loss,
        red_remaining=red_strength - red_loss,
        blue_defense_modifier=blue_defense_modifier,
        red_defense_modifier=red_defense_modifier,
        stochastic=stochastic,
        blue_noise=blue_noise if stochastic else 0.0,
        red_noise=red_noise if stochastic else 0.0,
    )


def resolve_lanchester_deterministic(
    blue_strength: float,
    red_strength: float,
    *,
    config: LanchesterConfig | None = None,
    blue_defense_modifier: float = 1.0,
    red_defense_modifier: float = 1.0,
) -> LanchesterOutcome:
    """Resolve one deterministic combat step."""

    return resolve_lanchester(
        blue_strength,
        red_strength,
        config=config,
        blue_defense_modifier=blue_defense_modifier,
        red_defense_modifier=red_defense_modifier,
        stochastic=False,
    )


def resolve_lanchester_stochastic(
    blue_strength: float,
    red_strength: float,
    *,
    config: LanchesterConfig | None = None,
    blue_defense_modifier: float = 1.0,
    red_defense_modifier: float = 1.0,
    seed: int | None = None,
    rng: Random | None = None,
) -> LanchesterOutcome:
    """Resolve one stochastic combat step."""

    return resolve_lanchester(
        blue_strength,
        red_strength,
        config=config,
        blue_defense_modifier=blue_defense_modifier,
        red_defense_modifier=red_defense_modifier,
        stochastic=True,
        seed=seed,
        rng=rng,
    )


@dataclass(frozen=True, slots=True)
class LanchesterResolver(CombatResolver):
    """Small wrapper that adapts the pure function for future engine wiring."""

    config: LanchesterConfig = LanchesterConfig()

    def resolve(
        self,
        blue_strength: float,
        red_strength: float,
        *,
        blue_defense_modifier: float = 1.0,
        red_defense_modifier: float = 1.0,
        stochastic: bool | None = None,
        seed: int | None = None,
        rng: Random | None = None,
    ) -> LanchesterOutcome:
        return resolve_lanchester(
            blue_strength,
            red_strength,
            config=self.config,
            blue_defense_modifier=blue_defense_modifier,
            red_defense_modifier=red_defense_modifier,
            stochastic=stochastic,
            seed=seed,
            rng=rng,
        )


def _validate_inputs(
    *,
    blue_strength: float,
    red_strength: float,
    blue_defense_modifier: float,
    red_defense_modifier: float,
    config: LanchesterConfig,
    seed: int | None,
    rng: Random | None,
) -> None:
    for name, value in (
        ("blue_strength", blue_strength),
        ("red_strength", red_strength),
        ("blue_defense_modifier", blue_defense_modifier),
        ("red_defense_modifier", red_defense_modifier),
        ("blue_attrition_rate", config.blue_attrition_rate),
        ("red_attrition_rate", config.red_attrition_rate),
        ("time_step", config.time_step),
        ("noise_std", config.noise_std),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")

    if blue_defense_modifier == 0:
        raise ValueError("blue_defense_modifier must be greater than zero.")
    if red_defense_modifier == 0:
        raise ValueError("red_defense_modifier must be greater than zero.")
    if seed is not None and rng is not None:
        raise ValueError("Provide either seed or rng, not both.")


def _coerce_rng(*, seed: int | None, rng: Random | None) -> Random:
    """Build or forward a random generator for stochastic combat."""

    if rng is not None:
        return rng
    return Random(seed)


def _clamp_loss(raw_loss: float, available_strength: float) -> float:
    """Clamp losses into the valid range for a single combat step."""

    return min(max(raw_loss, 0.0), available_strength)
