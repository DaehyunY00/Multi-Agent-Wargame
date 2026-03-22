"""Run repeated games and summarize local-LLM output stability."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wargame.analysis import load_jsonl_records  # noqa: E402
from wargame.experiments import get_seed_sequence  # noqa: E402


@dataclass(frozen=True, slots=True)
class LocalLLMDecisionStats:
    """Aggregated local-LLM parser stability signals from one log."""

    decision_count: int = 0
    fallback_count: int = 0
    parse_success_count: int = 0


@dataclass(frozen=True, slots=True)
class StabilityRunRecord:
    """Structured result for one repeated single-game execution."""

    seed: int
    completed: bool
    abnormal_termination: bool
    output_path: str | None
    run_id: str | None = None
    terminal: bool = False
    turns: int = 0
    fallback_count: int = 0
    parse_success_count: int = 0
    llm_decision_count: int = 0
    return_code: int = 0
    error: str | None = None


def main(argv: Sequence[str] | None = None) -> int:
    """Run repeated single-game experiments and save stability summaries."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, help="Scenario id or YAML path.")
    parser.add_argument("--blue-agent", required=True, help="Blue agent spec.")
    parser.add_argument("--red-agent", required=True, help="Red agent spec.")
    parser.add_argument("--white-cell", default="heuristic", help="White-cell spec or 'none'.")
    parser.add_argument("--seed-count", type=int, default=5, help="Number of repeated runs.")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed value.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/llm_stability"),
        help="Directory for per-run JSONL logs and stability summaries.",
    )
    parser.add_argument("--visibility-radius", type=int, default=None)
    parser.add_argument("--identification-radius", type=int, default=None)
    parser.add_argument(
        "--fog-preset",
        choices=["auto", "baseline", "llm"],
        default="auto",
        help="Forwarded to scripts/run_single_game.py when explicit fog radii are omitted.",
    )
    parser.add_argument("--stochastic-combat", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--context-window", type=int, default=2048)
    parser.add_argument(
        "--backend",
        choices=["auto", "mlx", "vllm"],
        default="auto",
        help="Forwarded backend hint for local_llm agent specs.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(args.seed_start + seed for seed in get_seed_sequence(args.seed_count))
    run_results = [
        _run_single_game(
            scenario=args.scenario,
            blue_agent=args.blue_agent,
            red_agent=args.red_agent,
            white_cell=args.white_cell,
            seed=seed,
            output_dir=args.output_dir,
            visibility_radius=args.visibility_radius,
            identification_radius=args.identification_radius,
            fog_preset=args.fog_preset,
            stochastic_combat=args.stochastic_combat,
            noise_std=args.noise_std,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            context_window=args.context_window,
            backend=args.backend,
        )
        for seed in seeds
    ]

    summary = build_stability_summary(run_results)
    summary["scenario"] = args.scenario
    summary["blue_agent"] = args.blue_agent
    summary["red_agent"] = args.red_agent
    summary["seed_count"] = args.seed_count
    summary["seed_start"] = args.seed_start
    summary["output_dir"] = str(args.output_dir)

    json_path = args.output_dir / "stability_summary.json"
    text_path = args.output_dir / "stability_summary.txt"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text_path.write_text(format_stability_summary(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def collect_local_llm_stats(path: str | Path) -> LocalLLMDecisionStats:
    """Count local-LLM fallbacks and parse successes from one JSONL log."""

    decisions = 0
    fallbacks = 0
    parse_successes = 0
    for record in load_jsonl_records(path):
        for decision in _iter_action_agent_decisions(record):
            metadata = decision.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            if metadata.get("decision_source") != "local_llm" and "json_parse_success" not in metadata:
                continue
            decisions += 1
            if decision.get("used_fallback") is True:
                fallbacks += 1
            if metadata.get("json_parse_success") is True:
                parse_successes += 1
            elif "json_parse_success" not in metadata and decision.get("used_fallback") is False:
                parse_successes += 1
    return LocalLLMDecisionStats(
        decision_count=decisions,
        fallback_count=fallbacks,
        parse_success_count=parse_successes,
    )


def build_stability_summary(run_results: Sequence[StabilityRunRecord]) -> dict[str, Any]:
    """Aggregate repeated-run outcomes into one JSON-friendly summary."""

    total_runs = len(run_results)
    completed_runs = sum(1 for run in run_results if run.completed)
    abnormal_termination_count = sum(1 for run in run_results if run.abnormal_termination)
    llm_decision_count = sum(run.llm_decision_count for run in run_results)
    fallback_count = sum(run.fallback_count for run in run_results)
    parse_success_count = sum(run.parse_success_count for run in run_results)
    return {
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "abnormal_termination_count": abnormal_termination_count,
        "llm_decision_count": llm_decision_count,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / llm_decision_count if llm_decision_count else 0.0,
        "parse_success_rate": parse_success_count / llm_decision_count if llm_decision_count else 0.0,
        "per_run_output_paths": [
            run.output_path
            for run in run_results
            if run.output_path is not None
        ],
        "runs": [asdict(run) for run in run_results],
    }


def format_stability_summary(summary: Mapping[str, Any]) -> str:
    """Render a compact human-readable stability summary."""

    lines = [
        "Local LLM Stability Summary",
        f"Scenario: {summary.get('scenario', 'n/a')}",
        f"Blue agent: {summary.get('blue_agent', 'n/a')}",
        f"Red agent: {summary.get('red_agent', 'n/a')}",
        f"Total runs: {summary.get('total_runs', 0)}",
        f"Completed runs: {summary.get('completed_runs', 0)}",
        f"Abnormal terminations: {summary.get('abnormal_termination_count', 0)}",
        f"Fallback count: {summary.get('fallback_count', 0)}",
        f"Fallback rate: {float(summary.get('fallback_rate', 0.0)):.3f}",
        f"Parse success rate: {float(summary.get('parse_success_rate', 0.0)):.3f}",
        "Per-run outputs:",
    ]
    output_paths = summary.get("per_run_output_paths", [])
    if isinstance(output_paths, Sequence):
        lines.extend(
            f"- {path}"
            for path in output_paths
            if isinstance(path, str)
        )
    return "\n".join(lines) + "\n"


def _run_single_game(
    *,
    scenario: str,
    blue_agent: str,
    red_agent: str,
    white_cell: str,
    seed: int,
    output_dir: Path,
    visibility_radius: int | None,
    identification_radius: int | None,
    fog_preset: str,
    stochastic_combat: bool,
    noise_std: float,
    temperature: float,
    max_tokens: int,
    context_window: int,
    backend: str,
) -> StabilityRunRecord:
    """Execute one existing single-game CLI run and recover stability stats."""

    script_path = Path(__file__).resolve().with_name("run_single_game.py")
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"seed_{seed}.jsonl"
    command = [
        sys.executable,
        str(script_path),
        "--scenario",
        scenario,
        "--blue-agent",
        blue_agent,
        "--red-agent",
        red_agent,
        "--white-cell",
        white_cell,
        "--seed",
        str(seed),
        "--output",
        str(log_path),
        "--fog-preset",
        fog_preset,
        "--noise-std",
        str(noise_std),
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
        "--context-window",
        str(context_window),
        "--backend",
        backend,
    ]
    if visibility_radius is not None:
        command.extend(["--visibility-radius", str(visibility_radius)])
    if identification_radius is not None:
        command.extend(["--identification-radius", str(identification_radius)])
    if stochastic_combat:
        command.append("--stochastic-combat")

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=script_path.resolve().parents[1],
    )
    if completed.returncode != 0:
        return StabilityRunRecord(
            seed=seed,
            completed=False,
            abnormal_termination=True,
            output_path=str(log_path) if log_path.exists() else None,
            return_code=completed.returncode,
            error=completed.stderr.strip() or completed.stdout.strip() or "run_single_game.py failed",
        )

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return StabilityRunRecord(
            seed=seed,
            completed=False,
            abnormal_termination=True,
            output_path=str(log_path) if log_path.exists() else None,
            return_code=completed.returncode,
            error=f"Invalid JSON summary from run_single_game.py: {exc.msg}.",
        )

    stats = collect_local_llm_stats(log_path) if log_path.exists() else LocalLLMDecisionStats()
    is_completed = bool(payload.get("terminal")) and log_path.exists()
    return StabilityRunRecord(
        seed=seed,
        completed=is_completed,
        abnormal_termination=not is_completed,
        output_path=str(log_path) if log_path.exists() else payload.get("log_path"),
        run_id=payload.get("run_id"),
        terminal=bool(payload.get("terminal")),
        turns=int(payload.get("turns", 0)),
        fallback_count=stats.fallback_count,
        parse_success_count=stats.parse_success_count,
        llm_decision_count=stats.decision_count,
        return_code=completed.returncode,
        error=None if is_completed else "Run did not reach a terminal state.",
    )


def _iter_action_agent_decisions(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Yield blue/red action decisions from one logged turn record."""

    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return []
    return [
        decision
        for key in ("blue", "red")
        if isinstance((decision := metadata.get(key)), Mapping)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
