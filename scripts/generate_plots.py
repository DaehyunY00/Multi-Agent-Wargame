"""CLI for generating publication-quality experiment plots.

Usage:
  python scripts/generate_plots.py \\
    --input-dirs runs/phase4/qwen/ runs/phase4/baseline/rule/ \\
    --labels "Qwen2.5-7B" "Rule-Based" \\
    --output-dir runs/final_plots/

Each --input-dir must contain *.jsonl run logs (searched recursively).
One label must be supplied per input directory (via --labels).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wargame.analysis.metrics import (  # noqa: E402
    action_entropy,
    doctrine_compliance_rate,
    escalation_sensitivity_index,
    load_jsonl_records,
    tactical_rationality_score,
    win_rate,
)
from wargame.analysis.plots import (  # noqa: E402
    PlotBundle,
    build_experiment_plots,
    plot_action_entropy_comparison,
    plot_dcr_distribution,
    plot_esi_comparison,
    plot_force_curve,
    plot_win_rate_by_scenario,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate all configured plots and print a summary of written files."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directories containing *.jsonl run logs (searched recursively).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        help="Display label for each input directory (must match --input-dirs count).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/final_plots"),
    )
    parser.add_argument(
        "--scenario-id",
        default=None,
        metavar="ID",
        help="Optional scenario identifier embedded in force-curve plot titles.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    labels = args.labels or [d.name for d in args.input_dirs]
    if len(labels) != len(args.input_dirs):
        parser.error(
            f"--labels count ({len(labels)}) must match --input-dirs count ({len(args.input_dirs)})."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_written: list[Path] = []

    # -----------------------------------------------------------------------
    # Collect per-run metrics for each group
    # -----------------------------------------------------------------------
    groups: list[dict] = []
    for label, dir_path in zip(labels, args.input_dirs):
        groups.append(_collect_group(label, dir_path))

    # -----------------------------------------------------------------------
    # SVG fallback — one force-curve per run (no matplotlib needed)
    # -----------------------------------------------------------------------
    for group in groups:
        for idx, records in enumerate(group["records"]):
            svg_dir = args.output_dir / "svg" / group["label"]
            bundle: PlotBundle = build_experiment_plots(records, svg_dir / f"run_{idx:03d}")
            all_written.extend(bundle.files)

    # -----------------------------------------------------------------------
    # Matplotlib charts (skipped gracefully if matplotlib is absent)
    # -----------------------------------------------------------------------
    try:
        _generate_matplotlib_plots(groups, args.output_dir, all_written, args.scenario_id)
    except ImportError as exc:
        print(f"[warn] matplotlib unavailable — skipping chart generation. ({exc})", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"Written {len(all_written)} file(s) to {args.output_dir}")
    for path in sorted(all_written):
        print(f"  {path}")
    return 0


def _collect_group(label: str, dir_path: Path) -> dict:
    """Load all JSONL runs under *dir_path* and compute per-run metric vectors."""
    jsonl_files = sorted(dir_path.glob("**/*.jsonl"))
    records_list: list[list] = []
    dcr_list: list[float] = []
    trs_list: list[float] = []
    esi_list: list[float] = []
    entropy_list: list[float] = []

    for path in jsonl_files:
        try:
            records = load_jsonl_records(path)
        except Exception:  # noqa: BLE001
            continue
        if not records:
            continue
        records_list.append(records)
        dcr_list.append(doctrine_compliance_rate(records))
        trs_list.append(tactical_rationality_score(records))
        esi_list.append(escalation_sensitivity_index(records))
        entropy_list.append(action_entropy(records))

    return {
        "label": label,
        "records": records_list,
        "dcr": dcr_list,
        "trs": trs_list,
        "esi": esi_list,
        "entropy": entropy_list,
    }


def _generate_matplotlib_plots(
    groups: list[dict],
    output_dir: Path,
    written: list[Path],
    scenario_id: str | None,
) -> None:
    """Generate all matplotlib charts; appends written paths to *written*."""

    labels = [g["label"] for g in groups]

    # Action entropy boxplot
    entropy_data = {g["label"]: g["entropy"] for g in groups if g["entropy"]}
    if entropy_data:
        written.extend(
            plot_action_entropy_comparison(entropy_data, output_dir / "action_entropy")
        )

    # ESI comparison boxplot
    esi_data = {g["label"]: g["esi"] for g in groups if g["esi"]}
    if esi_data:
        written.extend(plot_esi_comparison(esi_data, output_dir / "esi_comparison"))

    # DCR distribution histogram
    dcr_data = {g["label"]: g["dcr"] for g in groups if g["dcr"]}
    if dcr_data:
        written.extend(
            plot_dcr_distribution(dcr_data, output_dir / "dcr_distribution")
        )

    # Win-rate bar chart — one synthetic "scenario" per group label
    # (groups may span different scenarios; use label as key when no scenario axis)
    win_rate_data: dict[str, dict[str, float]] = {}
    scenario_key = scenario_id or "all"
    for group in groups:
        if group["records"]:
            wr = win_rate(group["records"])
            win_rate_data.setdefault(scenario_key, {})[group["label"]] = wr
    if win_rate_data:
        written.extend(
            plot_win_rate_by_scenario(win_rate_data, output_dir / "win_rate_by_scenario")
        )

    # Force curves — first run of each group only (to avoid hundreds of files)
    for group in groups:
        if not group["records"]:
            continue
        title = f"Force Curve — {group['label']}"
        if scenario_id:
            title += f" ({scenario_id})"
        written.extend(
            plot_force_curve(
                group["records"][0],
                output_dir / f"force_curve_{_slugify(group['label'])}",
                title=title,
            )
        )


def _slugify(text: str) -> str:
    """Convert a label to a filesystem-safe slug."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
