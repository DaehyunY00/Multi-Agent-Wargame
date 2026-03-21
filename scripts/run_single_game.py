"""CLI entrypoint for running a single wargame scenario."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wargame.agents import (  # noqa: E402
    ActionParser,
    BlueAgent,
    HeuristicWhiteCellAgent,
    LocalLLMConfig,
    RandomAgent,
    RedAgent,
    RuleBasedAgent,
    ScriptAgent,
    ScriptBehavior,
    StateRenderer,
)
from wargame.combat import LanchesterConfig, LanchesterResolver  # noqa: E402
from wargame.core import Faction  # noqa: E402
from wargame.engine import FogOfWarFilter, SimulationEngine, StateManager  # noqa: E402
from wargame.experiments import ExperimentRunner  # noqa: E402
from wargame.logging import JsonlLogger  # noqa: E402
from wargame.orchestrator import TurnLoop  # noqa: E402
from wargame.scenarios import build_grid, load_scenario, scenario_to_game_state  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    """Run a single configured game and print a compact JSON summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, help="Scenario id or YAML path.")
    parser.add_argument("--blue-agent", default="rule", help="Agent spec for blue.")
    parser.add_argument("--red-agent", default="rule", help="Agent spec for red.")
    parser.add_argument("--white-cell", default="heuristic", help="White-cell spec or 'none'.")
    parser.add_argument("--seed", type=int, default=0, help="Base experiment seed.")
    parser.add_argument("--output", type=Path, default=Path("runs/single_game.jsonl"), help="JSONL log path.")
    parser.add_argument("--visibility-radius", type=int, default=3)
    parser.add_argument("--identification-radius", type=int, default=1)
    parser.add_argument("--stochastic-combat", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--context-window", type=int, default=2048)
    parser.add_argument("--backend", choices=["auto", "mlx", "vllm"], default="auto")
    args = parser.parse_args(list(argv) if argv is not None else None)

    scenario = load_scenario(_resolve_scenario_path(args.scenario))
    grid = build_grid(scenario)
    initial_state = scenario_to_game_state(scenario)
    fog = FogOfWarFilter(
        visibility_radius=args.visibility_radius,
        identification_radius=args.identification_radius,
    )
    engine = SimulationEngine(
        state_manager=StateManager(initial_state=initial_state, fog_of_war=fog),
        combat_resolver=LanchesterResolver(
            LanchesterConfig(stochastic=args.stochastic_combat, noise_std=args.noise_std)
        ),
        fog_of_war=fog,
        grid=grid,
    )
    logger = JsonlLogger(path=args.output)
    loop = TurnLoop(
        engine=engine,
        blue_agent=_build_agent(args.blue_agent, faction=Faction.BLUE, grid=grid, temperature=args.temperature, max_tokens=args.max_tokens, context_window=args.context_window, backend=args.backend),
        red_agent=_build_agent(args.red_agent, faction=Faction.RED, grid=grid, temperature=args.temperature, max_tokens=args.max_tokens, context_window=args.context_window, backend=args.backend),
        renderer=StateRenderer(),
        parser=ActionParser(grid=grid),
        white_cell=_build_white_cell(args.white_cell),
        logger=logger,
    )
    runner = ExperimentRunner(
        scenario=scenario,
        turn_loop=loop,
        logger=logger,
        seed=args.seed,
    )
    run = runner.run()
    payload = {
        "run_id": run.context.run_id,
        "scenario_id": run.context.scenario_id,
        "turns": len(run.turns),
        "terminal": run.terminal,
        "log_path": str(run.log_path) if run.log_path is not None else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _build_agent(
    spec: str,
    *,
    faction: Faction,
    grid,
    temperature: float = 0.7,
    max_tokens: int = 512,
    context_window: int = 2048,
    backend: str = "auto",
):
    """Build a baseline or LLM agent from a compact CLI spec.

    Supported specs:
      rule, random, script, script:<behavior>
      local_llm:<model_path>
    """
    normalized = spec.strip().lower()
    if normalized == "rule":
        return RuleBasedAgent(grid=grid, faction=faction, name=f"{faction.value}_rule")
    if normalized == "random":
        return RandomAgent(grid=grid, faction=faction, name=f"{faction.value}_random")
    if normalized == "script":
        return ScriptAgent(grid=grid, faction=faction, behavior=ScriptBehavior.FRONTAL_ASSAULT, name=f"{faction.value}_script")
    if normalized.startswith("script:"):
        behavior = ScriptBehavior(normalized.split(":", maxsplit=1)[1])
        return ScriptAgent(grid=grid, faction=faction, behavior=behavior, name=f"{faction.value}_{behavior.value}")
    if normalized.startswith("local_llm:"):
        model_path = spec.strip()[len("local_llm:"):]
        llm_backend = _select_backend(model_path, backend)
        config = LocalLLMConfig(
            model_name=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
        )
        parser = ActionParser(grid=grid)
        agent_cls = BlueAgent if faction is Faction.BLUE else RedAgent
        return agent_cls(backend=llm_backend, config=config, parser=parser)
    raise ValueError(f"Unsupported agent spec: {spec!r}")


def _select_backend(model_path: str, backend_hint: str):
    """Resolve an LLM backend instance from a model path and explicit hint."""

    use_mlx = backend_hint == "mlx" or (
        backend_hint == "auto" and _looks_like_mlx_model(model_path)
    )
    if use_mlx:
        from wargame.agents.backends import MLXLocalLLMBackend  # noqa: PLC0415

        return MLXLocalLLMBackend(model_path=model_path)
    if backend_hint == "vllm":
        from wargame.agents.backends import VLLMLocalLLMBackend  # noqa: PLC0415

        return VLLMLocalLLMBackend(model_path=model_path)
    raise ValueError(
        f"Cannot auto-select a backend for {model_path!r}. "
        "Pass --backend mlx (Apple Silicon) or --backend vllm."
    )


def _looks_like_mlx_model(model_path: str) -> bool:
    lower = model_path.lower()
    return "mlx-community/" in lower or lower.startswith("mlx")


def _build_white_cell(spec: str):
    """Build an optional white-cell evaluator."""

    normalized = spec.strip().lower()
    if normalized in {"none", "off"}:
        return None
    if normalized == "heuristic":
        return HeuristicWhiteCellAgent()
    raise ValueError(f"Unsupported white-cell spec: {spec!r}")


def _resolve_scenario_path(value: str) -> Path:
    """Resolve a preset id or explicit path into a YAML file path."""

    candidate = Path(value)
    if candidate.exists():
        return candidate
    preset_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "wargame"
        / "scenarios"
        / "presets"
        / f"{value}.yaml"
    )
    if preset_path.exists():
        return preset_path
    raise FileNotFoundError(f"Unknown scenario preset or path: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
