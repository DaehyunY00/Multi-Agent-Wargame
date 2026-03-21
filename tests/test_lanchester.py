"""Tests for deterministic and stochastic Lanchester combat resolution."""

from math import isclose

from wargame.combat import (
    LanchesterConfig,
    resolve_lanchester_deterministic,
    resolve_lanchester_stochastic,
)


def test_symmetric_case_produces_symmetric_losses() -> None:
    """Equal forces with equal parameters should degrade symmetrically."""

    outcome = resolve_lanchester_deterministic(
        100.0,
        100.0,
        config=LanchesterConfig(
            blue_attrition_rate=0.05,
            red_attrition_rate=0.05,
            time_step=1.0,
        ),
    )

    assert isclose(outcome.blue_loss, outcome.red_loss)
    assert isclose(outcome.blue_remaining, outcome.red_remaining)


def test_terrain_modifier_reduces_incoming_losses() -> None:
    """A stronger defense modifier should preserve more remaining strength."""

    baseline = resolve_lanchester_deterministic(
        100.0,
        100.0,
        config=LanchesterConfig(blue_attrition_rate=0.05, red_attrition_rate=0.05),
        blue_defense_modifier=1.0,
    )
    protected = resolve_lanchester_deterministic(
        100.0,
        100.0,
        config=LanchesterConfig(blue_attrition_rate=0.05, red_attrition_rate=0.05),
        blue_defense_modifier=1.5,
    )

    assert protected.blue_loss < baseline.blue_loss
    assert isclose(protected.red_loss, baseline.red_loss)


def test_deterministic_resolution_is_reproducible() -> None:
    """Deterministic mode should return the same structured output every time."""

    config = LanchesterConfig(
        blue_attrition_rate=0.06,
        red_attrition_rate=0.04,
        time_step=1.0,
    )

    first = resolve_lanchester_deterministic(120.0, 90.0, config=config)
    second = resolve_lanchester_deterministic(120.0, 90.0, config=config)

    assert first == second


def test_stochastic_resolution_is_reproducible_with_fixed_seed() -> None:
    """Stochastic mode should be reproducible when the seed is fixed."""

    config = LanchesterConfig(
        blue_attrition_rate=0.05,
        red_attrition_rate=0.05,
        noise_std=0.2,
    )

    first = resolve_lanchester_stochastic(100.0, 100.0, config=config, seed=7)
    second = resolve_lanchester_stochastic(100.0, 100.0, config=config, seed=7)

    assert first == second


def test_losses_are_clamped_to_available_strength() -> None:
    """A single combat step should never drive remaining strength below zero."""

    outcome = resolve_lanchester_deterministic(
        5.0,
        200.0,
        config=LanchesterConfig(red_attrition_rate=10.0),
    )

    assert outcome.blue_remaining == 0.0
    assert outcome.blue_loss == 5.0
