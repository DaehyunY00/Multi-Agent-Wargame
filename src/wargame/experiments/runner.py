"""Single-experiment execution helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wargame.core.enums import Faction
from wargame.core.models import TurnResult
from wargame.logging.jsonl_logger import JsonlLogger
from wargame.orchestrator.turn_loop import TurnLoop
from wargame.scenarios.schema import ScenarioSpec

from .seeds import apply_seed_bundle, derive_seed_bundle


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    """Stable metadata attached to every turn record of one run."""

    run_id: str
    scenario_id: str
    scenario_name: str
    seed: int | None
    blue_agent: str
    red_agent: str
    white_cell: str | None
    initial_force_totals: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """Executed experiment bundle with context, turns, and log location."""

    context: ExperimentContext
    turns: list[TurnResult]
    log_path: Path | None = None
    terminal: bool = False


@dataclass(slots=True)
class ExperimentRunner:
    """Execute one scenario/agent configuration and persist run metadata."""

    scenario: ScenarioSpec
    turn_loop: TurnLoop
    logger: JsonlLogger | None = None
    seed: int | None = None
    run_id: str | None = None
    blue_agent_name: str | None = None
    red_agent_name: str | None = None
    extra_context: dict[str, Any] = field(default_factory=dict)
    reset_log: bool = True
    prepare_run: Callable[[TurnLoop, int | None], None] | None = None

    def __post_init__(self) -> None:
        if self.logger is None and self.turn_loop.logger is not None:
            self.logger = self.turn_loop.logger
        elif self.logger is not None:
            self.turn_loop.logger = self.logger

    def run(self) -> ExperimentRun:
        """Execute one experiment configuration from start to finish."""

        if self.prepare_run is not None:
            self.prepare_run(self.turn_loop, self.seed)
        else:
            apply_seed_bundle(self.turn_loop, derive_seed_bundle(self.seed))
        context = self.build_context()
        if self.logger is not None:
            if self.reset_log:
                self.logger.clear()
            self.logger.set_context(
                run_id=context.run_id,
                scenario_id=context.scenario_id,
                scenario_name=context.scenario_name,
                seed=context.seed,
                blue_agent=context.blue_agent,
                red_agent=context.red_agent,
                white_cell=context.white_cell,
                initial_force_totals=context.initial_force_totals,
                metadata=context.metadata,
            )

        turns = self.turn_loop.run_until_terminal()
        return ExperimentRun(
            context=context,
            turns=turns,
            log_path=self.logger.path if self.logger is not None else None,
            terminal=self.turn_loop.engine.is_terminal(),
        )

    def build_context(self) -> ExperimentContext:
        """Build a JSON-friendly metadata bundle for the current run."""

        initial_state = self.turn_loop.engine.state_manager.initial_state
        blue_agent = self.blue_agent_name or self.turn_loop.blue_agent.name
        red_agent = self.red_agent_name or self.turn_loop.red_agent.name
        run_id = self.run_id or _default_run_id(
            scenario_id=self.scenario.scenario_id,
            blue_agent=blue_agent,
            red_agent=red_agent,
            seed=self.seed,
        )
        return ExperimentContext(
            run_id=run_id,
            scenario_id=self.scenario.scenario_id,
            scenario_name=self.scenario.name,
            seed=self.seed,
            blue_agent=blue_agent,
            red_agent=red_agent,
            white_cell=self.turn_loop.white_cell.name if self.turn_loop.white_cell else None,
            initial_force_totals={
                faction.value: _initial_force_total(initial_state.units, faction)
                for faction in (Faction.BLUE, Faction.RED)
            },
            metadata={
                **self.extra_context,
                "seed_control": "hook" if self.prepare_run is not None else "automatic",
                "seed_bundle": derive_seed_bundle(self.seed).as_dict(),
            },
        )


def _default_run_id(
    *,
    scenario_id: str,
    blue_agent: str,
    red_agent: str,
    seed: int | None,
) -> str:
    """Generate a stable, filename-friendly run identifier."""

    seed_label = "na" if seed is None else str(seed)
    return "-".join(
        _slugify(component)
        for component in (
            scenario_id,
            f"{blue_agent}-vs-{red_agent}",
            f"seed-{seed_label}",
        )
    )


def _initial_force_total(units: dict[str, Any], faction: Faction) -> int:
    """Sum the initial force strength for one faction."""

    return sum(
        int(unit.strength)
        for unit in units.values()
        if getattr(unit, "faction", None) == faction
    )


def _slugify(value: str) -> str:
    """Normalize a human-readable label into a compact token."""

    normalized = [
        character.lower()
        if character.isalnum()
        else "-"
        for character in value.strip()
    ]
    slug = "".join(normalized)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "run"
