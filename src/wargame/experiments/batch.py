"""Batch experiment execution helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wargame.scenarios.schema import ScenarioSpec

from .runner import ExperimentRun, ExperimentRunner


@dataclass(frozen=True, slots=True)
class MatchupSpec:
    """Describe one blue-vs-red agent pairing in a batch."""

    name: str
    blue_agent_name: str
    red_agent_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExperimentCondition:
    """One point in a scenario x matchup x seed experiment matrix."""

    scenario: ScenarioSpec
    matchup: MatchupSpec
    seed: int
    run_id: str
    log_path: Path | None = None


RunnerFactory = Callable[[ExperimentCondition], ExperimentRunner]


@dataclass(slots=True)
class BatchRunner:
    """Coordinate a batch of experiment runners or create them from a matrix."""

    runners: list[ExperimentRunner] = field(default_factory=list)

    @classmethod
    def from_matrix(
        cls,
        *,
        scenarios: Sequence[ScenarioSpec],
        matchups: Sequence[MatchupSpec],
        seeds: Sequence[int],
        runner_factory: RunnerFactory,
        output_dir: Path | None = None,
    ) -> BatchRunner:
        """Expand a scenario x matchup x seed matrix into executable runners."""

        runners: list[ExperimentRunner] = []
        for scenario in scenarios:
            for matchup in matchups:
                for seed in seeds:
                    run_id = _build_run_id(
                        scenario_id=scenario.scenario_id,
                        matchup_name=matchup.name,
                        seed=seed,
                    )
                    log_path = None
                    if output_dir is not None:
                        log_path = output_dir / scenario.scenario_id / matchup.name / f"seed_{seed}.jsonl"
                    condition = ExperimentCondition(
                        scenario=scenario,
                        matchup=matchup,
                        seed=seed,
                        run_id=run_id,
                        log_path=log_path,
                    )
                    runners.append(runner_factory(condition))
        return cls(runners=runners)

    def run(self) -> dict[str, ExperimentRun]:
        """Execute the configured batch and key results by run_id."""

        results: dict[str, ExperimentRun] = {}
        for runner in self.runners:
            run = runner.run()
            results[run.context.run_id] = run
        return results


def _build_run_id(*, scenario_id: str, matchup_name: str, seed: int) -> str:
    """Create a stable run identifier for one matrix condition."""

    return f"{_slugify(scenario_id)}-{_slugify(matchup_name)}-seed-{seed}"


def _slugify(value: str) -> str:
    """Normalize a label into a compact path-safe token."""

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
