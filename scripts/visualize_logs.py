"""CLI for aggregate visualization of experiment JSONL logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wargame.analysis import build_aggregate_visualizations  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    """Generate aggregate plots and summaries for one or more JSONL logs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more JSONL log files or directories to scan recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/visualizations"),
        help="Directory where aggregate summaries and plots are written.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        log_paths = _collect_log_paths(args.paths)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    try:
        bundle = build_aggregate_visualizations(log_paths, args.output_dir)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output_dir": str(bundle.output_dir),
                "run_count": len(bundle.log_paths),
                "written_files": [str(path) for path in bundle.files],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _collect_log_paths(values: Sequence[str]) -> list[Path]:
    """Expand file and directory inputs into a sorted list of JSONL logs."""

    collected: set[Path] = set()
    for value in values:
        candidate = Path(value)
        if not candidate.exists():
            raise FileNotFoundError(f"Input path does not exist: {candidate}")
        if candidate.is_dir():
            collected.update(path.resolve() for path in candidate.rglob("*.jsonl"))
            continue
        if candidate.is_file():
            if candidate.suffix != ".jsonl":
                raise ValueError(f"Input file must end with .jsonl: {candidate}")
            collected.add(candidate.resolve())
            continue
        raise ValueError(f"Unsupported input path: {candidate}")

    paths = sorted(collected)
    if not paths:
        raise ValueError("No JSONL log files were found in the provided inputs.")
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
