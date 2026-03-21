"""CLI entrypoint for analyzing generated experiment logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wargame.analysis import (  # noqa: E402
    build_experiment_plots,
    doctrine_compliance_rate,
    load_jsonl_records,
    summarize_runs,
    tactical_rationality_score,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate one or more JSONL logs and print aggregate metrics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Log files or directories containing JSONL files.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Optional directory for per-run SVG plots.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    log_paths = _collect_log_paths(args.paths)
    summary = summarize_runs(log_paths)
    doctrine_scores = [doctrine_compliance_rate(path) for path in log_paths]
    rationality_scores = [tactical_rationality_score(path) for path in log_paths]

    plot_files: list[str] = []
    if args.plot_dir is not None:
        for log_path in log_paths:
            plot_bundle = build_experiment_plots(
                load_jsonl_records(log_path),
                args.plot_dir / log_path.stem,
            )
            plot_files.extend(str(path) for path in plot_bundle.files)

    print(
        json.dumps(
            {
                "run_count": summary.run_count,
                "blue_win_rate": summary.blue_win_rate,
                "red_win_rate": summary.red_win_rate,
                "mean_action_entropy": summary.mean_action_entropy,
                "mean_escalation_sensitivity_index": summary.mean_escalation_sensitivity_index,
                "mean_doctrine_compliance_rate": _mean(doctrine_scores),
                "mean_tactical_rationality_score": _mean(rationality_scores),
                "plot_files": plot_files,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _collect_log_paths(values: Sequence[str]) -> list[Path]:
    """Expand CLI inputs into concrete JSONL log paths."""

    paths: list[Path] = []
    for value in values:
        candidate = Path(value)
        if candidate.is_dir():
            paths.extend(sorted(candidate.rglob("*.jsonl")))
        elif candidate.suffix == ".jsonl":
            paths.append(candidate)
    return paths


def _mean(values: Sequence[float]) -> float:
    """Return a zero-safe mean."""

    if not values:
        return 0.0
    return sum(values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
