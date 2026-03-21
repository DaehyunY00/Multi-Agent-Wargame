"""CLI entrypoint for running batched wargame experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wargame.analysis import summarize_runs  # noqa: E402
from wargame.experiments import BatchRunner, ExperimentRunner, MatchupSpec, get_seed_sequence  # noqa: E402
from wargame.logging import JsonlLogger  # noqa: E402
from wargame.scenarios import load_scenario  # noqa: E402

from run_single_game import (  # noqa: E402
    _build_agent,
    _build_white_cell,
    _resolve_scenario_path,
)
from wargame.agents import ActionParser, StateRenderer  # noqa: E402
from wargame.combat import LanchesterConfig, LanchesterResolver  # noqa: E402
from wargame.engine import FogOfWarFilter, SimulationEngine, StateManager  # noqa: E402
from wargame.orchestrator import TurnLoop  # noqa: E402
from wargame.scenarios import build_grid, scenario_to_game_state  # noqa: E402
from wargame.core import Faction  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    """Run a scenario-agent-seed matrix and print summary statistics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", required=True, help="Scenario id or YAML path. Repeatable.")
    parser.add_argument("--matchup", action="append", required=True, help="Matchup in blue_spec,red_spec form. Repeatable.")
    parser.add_argument("--seed-count", type=int, default=3, help="Number of sequential seeds to execute.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/batch"))
    parser.add_argument("--white-cell", default="heuristic")
    parser.add_argument("--stochastic-combat", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--visibility-radius", type=int, default=3)
    parser.add_argument("--identification-radius", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--context-window", type=int, default=2048)
    parser.add_argument("--backend", choices=["auto", "mlx", "vllm"], default="auto")
    args = parser.parse_args(list(argv) if argv is not None else None)

    scenarios = [load_scenario(_resolve_scenario_path(item)) for item in args.scenario]
    matchups = [_parse_matchup(item) for item in args.matchup]
    seeds = get_seed_sequence(args.seed_count)

    batch = BatchRunner.from_matrix(
        scenarios=scenarios,
        matchups=matchups,
        seeds=seeds,
        output_dir=args.output_dir,
        runner_factory=lambda condition: _build_runner(
            condition=condition,
            white_cell_spec=args.white_cell,
            stochastic_combat=args.stochastic_combat,
            noise_std=args.noise_std,
            visibility_radius=args.visibility_radius,
            identification_radius=args.identification_radius,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            context_window=args.context_window,
            backend=args.backend,
        ),
    )
    results = batch.run()
    log_paths = [run.log_path for run in results.values() if run.log_path is not None]
    summary = summarize_runs(log_paths)
    print(
        json.dumps(
            {
                "run_count": summary.run_count,
                "blue_win_rate": summary.blue_win_rate,
                "red_win_rate": summary.red_win_rate,
                "mean_action_entropy": summary.mean_action_entropy,
                "mean_escalation_sensitivity_index": summary.mean_escalation_sensitivity_index,
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_matchup(value: str) -> MatchupSpec:
    """Parse a CLI matchup token."""

    blue_spec, red_spec = [item.strip() for item in value.split(",", maxsplit=1)]
    name = f"{blue_spec}-vs-{red_spec}".replace(":", "-")
    return MatchupSpec(
        name=name,
        blue_agent_name=blue_spec,
        red_agent_name=red_spec,
    )


def _build_runner(
    *,
    condition,
    white_cell_spec: str,
    stochastic_combat: bool,
    noise_std: float,
    visibility_radius: int = 3,
    identification_radius: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 512,
    context_window: int = 2048,
    backend: str = "auto",
) -> ExperimentRunner:
    """Build one ExperimentRunner for a batch matrix condition."""

    scenario = condition.scenario
    grid = build_grid(scenario)
    fog = FogOfWarFilter(
        visibility_radius=visibility_radius,
        identification_radius=identification_radius,
    )
    engine = SimulationEngine(
        state_manager=StateManager(initial_state=scenario_to_game_state(scenario), fog_of_war=fog),
        combat_resolver=LanchesterResolver(LanchesterConfig(stochastic=stochastic_combat, noise_std=noise_std)),
        fog_of_war=fog,
        grid=grid,
    )
    agent_kwargs = dict(temperature=temperature, max_tokens=max_tokens, context_window=context_window, backend=backend)
    logger = JsonlLogger(path=condition.log_path or Path(f"{condition.run_id}.jsonl"))
    loop = TurnLoop(
        engine=engine,
        blue_agent=_build_agent(condition.matchup.blue_agent_name, faction=Faction.BLUE, grid=grid, **agent_kwargs),
        red_agent=_build_agent(condition.matchup.red_agent_name, faction=Faction.RED, grid=grid, **agent_kwargs),
        renderer=StateRenderer(),
        parser=ActionParser(grid=grid),
        white_cell=_build_white_cell(white_cell_spec),
        logger=logger,
    )
    return ExperimentRunner(
        scenario=scenario,
        turn_loop=loop,
        logger=logger,
        seed=condition.seed,
        run_id=condition.run_id,
        blue_agent_name=condition.matchup.blue_agent_name,
        red_agent_name=condition.matchup.red_agent_name,
        extra_context={"matchup": condition.matchup.name},
    )


if __name__ == "__main__":
    raise SystemExit(main())
