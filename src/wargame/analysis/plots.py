"""Lightweight plot generation for experiment analysis.

The original SVG/JSON generation works without any third-party libraries.
The matplotlib-based functions below are gated behind a lazy import: if
matplotlib is not installed they raise ``ImportError`` with a clear message
rather than crashing at module import time.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

from wargame.core.enums import Faction

from .metrics import (
    action_entropy,
    escalation_sensitivity_index,
    json_parsing_success_rate,
    load_jsonl_records,
    win_rate,
)
from .stats import SummaryStatistics, summarize_runs


# ---------------------------------------------------------------------------
# Public data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlotBundle:
    """Container describing generated experiment-analysis artifacts."""

    output_dir: Path
    files: list[Path] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class VisualizationBundle:
    """Container describing generated aggregate visualization artifacts."""

    output_dir: Path
    log_paths: list[Path] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Original SVG-based interface (no external dependencies)
# ---------------------------------------------------------------------------


def build_experiment_plots(
    results: Sequence[Mapping[str, Any]] | Sequence[Any],
    output_dir: Path,
) -> PlotBundle:
    """Generate simple SVG/JSON artifacts from serialized turn records."""

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = [dict(record) for record in results if isinstance(record, Mapping)]
    files: list[Path] = []

    summary_path = output_dir / "turn_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "turns": len(normalized),
                "blue_force_totals": _force_totals(normalized, "blue"),
                "red_force_totals": _force_totals(normalized, "red"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    files.append(summary_path)

    svg_path = output_dir / "force_totals.svg"
    svg_path.write_text(_build_force_totals_svg(normalized), encoding="utf-8")
    files.append(svg_path)

    return PlotBundle(output_dir=output_dir, files=files)


def build_battlefield_replay(
    results: Sequence[Mapping[str, Any]] | Sequence[Any],
    output_dir: Path,
    *,
    run_label: str = "run",
    title: str | None = None,
) -> PlotBundle:
    """Generate a self-contained battlefield replay as SVG frames plus HTML."""

    normalized = [dict(record) for record in results if isinstance(record, Mapping)]
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    replay = _build_replay_payload(normalized, run_label=run_label, title=title)
    for frame in replay["frames"]:
        svg_path = output_dir / str(frame["file_name"])
        svg_path.write_text(_build_battlefield_frame_svg(frame, replay), encoding="utf-8")
        files.append(svg_path)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files.append(manifest_path)

    html_path = output_dir / "index.html"
    html_path.write_text(_build_replay_html(replay), encoding="utf-8")
    files.append(html_path)

    return PlotBundle(output_dir=output_dir, files=sorted(files))


def build_aggregate_visualizations(
    log_paths: Sequence[str | Path],
    output_dir: Path,
) -> VisualizationBundle:
    """Generate aggregate summaries and comparison charts for many JSONL logs."""

    _require_matplotlib()

    normalized_paths = [Path(path) for path in log_paths]
    if not normalized_paths:
        raise ValueError("At least one JSONL log path is required.")

    output_dir.mkdir(parents=True, exist_ok=True)
    loaded_runs = [
        (path, load_jsonl_records(path))
        for path in normalized_paths
    ]
    summary = summarize_runs(normalized_paths)
    scenario_ids = _scenario_ids(loaded_runs)
    agent_names = _agent_names(loaded_runs)
    written: list[Path] = []

    summary_payload = _build_summary_payload(
        summary,
        input_log_count=len(normalized_paths),
        scenario_ids=scenario_ids,
        agent_names=agent_names,
    )
    summary_json = output_dir / "aggregate_summary.json"
    summary_json.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written.append(summary_json)

    summary_csv = output_dir / "aggregate_summary.csv"
    _write_summary_csv(summary_csv, summary_payload)
    written.append(summary_csv)

    win_rate_data = _aggregate_win_rates_by_scenario(loaded_runs)
    if win_rate_data:
        written.extend(
            plot_win_rate_by_scenario(
                win_rate_data,
                output_dir / "win_rate_by_scenario",
            )
        )

    action_entropy_data = _collect_metric_by_agent(
        loaded_runs,
        metric=action_entropy,
    )
    if action_entropy_data:
        written.extend(
            plot_action_entropy_comparison(
                action_entropy_data,
                output_dir / "action_entropy_by_agent",
            )
        )

    escalation_data = _collect_metric_by_agent(
        loaded_runs,
        metric=escalation_sensitivity_index,
    )
    if escalation_data:
        written.extend(
            plot_esi_comparison(
                escalation_data,
                output_dir / "escalation_sensitivity_by_agent",
            )
        )

    parsing_data = _collect_metric_by_agent(
        loaded_runs,
        metric=json_parsing_success_rate,
    )
    if parsing_data:
        written.extend(
            plot_parsing_success_by_agent(
                parsing_data,
                output_dir / "parsing_success_by_agent",
            )
        )

    written.extend(_write_force_curves(loaded_runs, output_dir / "force_curves"))
    written.extend(_write_battlefield_replays(loaded_runs, output_dir / "battlefield_replays"))

    return VisualizationBundle(
        output_dir=output_dir,
        log_paths=normalized_paths,
        files=sorted(written),
    )


# ---------------------------------------------------------------------------
# Matplotlib-based chart functions (optional dependency)
# ---------------------------------------------------------------------------


def plot_action_entropy_comparison(
    data: dict[str, list[float]],
    output_path: Path,
) -> list[Path]:
    """Boxplot of per-run action entropy, one box per agent label.

    Parameters
    ----------
    data:
        Mapping of ``{agent_label: [entropy_per_run, …]}``.
    output_path:
        Destination without extension — ``.png`` and ``.svg`` are written.

    Returns
    -------
    List of paths actually written.
    """
    mpl, plt = _require_matplotlib()
    _apply_academic_style(mpl)

    fig, ax = plt.subplots(figsize=(max(4, len(data) * 1.2 + 2), 4))
    labels = list(data.keys())
    values = [data[k] for k in labels]

    bp = _boxplot(ax, values, labels)
    _apply_bw_boxplot(bp)

    ax.set_xlabel("Agent")
    ax.set_ylabel("Action Entropy (bits)")
    ax.set_title("Action Entropy by Agent")
    fig.tight_layout()

    return _save_figure(fig, plt, output_path)


def plot_win_rate_by_scenario(
    data: dict[str, dict[str, float]],
    output_path: Path,
) -> list[Path]:
    """Grouped bar chart of win rates per scenario.

    Parameters
    ----------
    data:
        ``{scenario_id: {agent_label: win_rate}}``.
    output_path:
        Destination without extension.
    """
    mpl, plt = _require_matplotlib()
    import numpy as np  # noqa: PLC0415

    _apply_academic_style(mpl)

    scenarios = list(data.keys())
    agents = sorted({a for s in data.values() for a in s})
    n_scenarios = len(scenarios)
    n_agents = len(agents)
    x = np.arange(n_scenarios)
    width = 0.7 / max(n_agents, 1)
    hatches = _hatch_cycle()

    fig, ax = plt.subplots(figsize=(max(5, n_scenarios * 1.5 + 2), 4))
    for i, agent in enumerate(agents):
        heights = [data[s].get(agent, 0.0) for s in scenarios]
        offset = (i - (n_agents - 1) / 2) * width
        bars = ax.bar(x + offset, heights, width, label=agent, hatch=next(hatches), color="white", edgecolor="black")
        _ = bars  # suppress lint

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Win Rate by Scenario")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    return _save_figure(fig, plt, output_path)


def plot_esi_comparison(
    data: dict[str, list[float]],
    output_path: Path,
) -> list[Path]:
    """Boxplot of Escalation Sensitivity Index across agent/model groups.

    Parameters
    ----------
    data:
        ``{agent_label: [esi_per_run, …]}``.
    output_path:
        Destination without extension.
    """
    mpl, plt = _require_matplotlib()
    _apply_academic_style(mpl)

    labels = list(data.keys())
    values = [data[k] for k in labels]

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2 + 2), 4))
    bp = _boxplot(ax, values, labels)
    _apply_bw_boxplot(bp)

    ax.set_xlabel("Agent / Model")
    ax.set_ylabel("Escalation Sensitivity Index")
    ax.set_title("ESI Comparison")
    fig.tight_layout()

    return _save_figure(fig, plt, output_path)


def plot_force_curve(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    title: str = "Force Strength Over Turns",
) -> list[Path]:
    """Line chart of blue/red force totals across turns for a single run.

    Parameters
    ----------
    records:
        List of turn log records (same format used by ``build_experiment_plots``).
    output_path:
        Destination without extension.
    title:
        Optional chart title (e.g. scenario + matchup identifier).
    """
    mpl, plt = _require_matplotlib()
    _apply_academic_style(mpl)

    blue = _force_totals(list(records), "blue")
    red = _force_totals(list(records), "red")
    turns = list(range(1, max(len(blue), len(red)) + 1))

    fig, ax = plt.subplots(figsize=(6, 4))
    if blue:
        ax.plot(turns[: len(blue)], blue, label="Blue", linestyle="-", marker="o", markersize=3, color="black")
    if red:
        ax.plot(turns[: len(red)], red, label="Red", linestyle="--", marker="s", markersize=3, color="dimgray")

    ax.set_xlabel("Turn")
    ax.set_ylabel("Remaining Strength")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return _save_figure(fig, plt, output_path)


def plot_parsing_success_by_agent(
    data: dict[str, list[float]],
    output_path: Path,
) -> list[Path]:
    """Boxplot of per-run JSON parsing success rates by agent."""

    mpl, plt = _require_matplotlib()
    _apply_academic_style(mpl)

    labels = list(data.keys())
    values = [data[k] for k in labels]

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2 + 2), 4))
    bp = _boxplot(ax, values, labels)
    _apply_bw_boxplot(bp)

    ax.set_xlabel("Agent")
    ax.set_ylabel("JSON Parsing Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Parsing Success by Agent")
    fig.tight_layout()

    return _save_figure(fig, plt, output_path)


def plot_dcr_distribution(
    data: dict[str, list[float]],
    output_path: Path,
    *,
    bins: int = 10,
) -> list[Path]:
    """Overlapping histogram of DCR distributions per agent/model.

    Parameters
    ----------
    data:
        ``{agent_label: [dcr_per_run, …]}``.
    output_path:
        Destination without extension.
    bins:
        Number of histogram bins (default 10).
    """
    mpl, plt = _require_matplotlib()
    _apply_academic_style(mpl)

    linestyles = _linestyle_cycle()
    fig, ax = plt.subplots(figsize=(6, 4))

    for label, values in data.items():
        if not values:
            continue
        ax.hist(
            values,
            bins=bins,
            range=(0, 1),
            alpha=0.55,
            label=label,
            histtype="stepfilled",
            linestyle=next(linestyles),
            edgecolor="black",
            linewidth=0.8,
            color="white",
        )

    ax.set_xlabel("Doctrine Compliance Rate")
    ax.set_ylabel("Count")
    ax.set_title("DCR Distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()

    return _save_figure(fig, plt, output_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_summary_payload(
    summary: SummaryStatistics,
    *,
    input_log_count: int,
    scenario_ids: list[str],
    agent_names: list[str],
) -> dict[str, Any]:
    """Flatten summary statistics and attach aggregate metadata."""

    return {
        "input_log_count": input_log_count,
        "scenario_ids": scenario_ids,
        "agent_names": agent_names,
        "run_count": summary.run_count,
        "mean_turns": summary.mean_turns,
        "blue_win_rate": summary.blue_win_rate,
        "red_win_rate": summary.red_win_rate,
        "mean_blue_remaining_force_ratio": summary.mean_blue_remaining_force_ratio,
        "mean_red_remaining_force_ratio": summary.mean_red_remaining_force_ratio,
        "mean_action_entropy": summary.mean_action_entropy,
        "mean_escalation_sensitivity_index": summary.mean_escalation_sensitivity_index,
        "mean_tactic_transition_frequency": summary.mean_tactic_transition_frequency,
        "mean_json_parsing_success_rate": summary.mean_json_parsing_success_rate,
        "inference_timing": {
            "sample_count": summary.inference_timing.sample_count,
            "mean_seconds": summary.inference_timing.mean_seconds,
            "max_seconds": summary.inference_timing.max_seconds,
        },
    }


def _write_summary_csv(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist one flat aggregate summary row as CSV."""

    inference = payload.get("inference_timing")
    inference_payload = inference if isinstance(inference, Mapping) else {}
    row = {
        "input_log_count": payload.get("input_log_count", 0),
        "scenario_ids": ";".join(_string_list(payload.get("scenario_ids"))),
        "agent_names": ";".join(_string_list(payload.get("agent_names"))),
        "run_count": payload.get("run_count", 0),
        "mean_turns": payload.get("mean_turns", 0.0),
        "blue_win_rate": payload.get("blue_win_rate", 0.0),
        "red_win_rate": payload.get("red_win_rate", 0.0),
        "mean_blue_remaining_force_ratio": payload.get("mean_blue_remaining_force_ratio", 0.0),
        "mean_red_remaining_force_ratio": payload.get("mean_red_remaining_force_ratio", 0.0),
        "mean_action_entropy": payload.get("mean_action_entropy", 0.0),
        "mean_escalation_sensitivity_index": payload.get(
            "mean_escalation_sensitivity_index",
            0.0,
        ),
        "mean_tactic_transition_frequency": payload.get(
            "mean_tactic_transition_frequency",
            0.0,
        ),
        "mean_json_parsing_success_rate": payload.get(
            "mean_json_parsing_success_rate",
            0.0,
        ),
        "inference_sample_count": inference_payload.get("sample_count", 0),
        "inference_mean_seconds": inference_payload.get("mean_seconds", 0.0),
        "inference_max_seconds": inference_payload.get("max_seconds", 0.0),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _scenario_ids(
    loaded_runs: Sequence[tuple[Path, Sequence[Mapping[str, Any]]]],
) -> list[str]:
    """Return the sorted scenario identifiers seen in the loaded runs."""

    scenario_ids = {
        _scenario_id_for_run(path, records)
        for path, records in loaded_runs
    }
    return sorted(scenario_ids)


def _agent_names(
    loaded_runs: Sequence[tuple[Path, Sequence[Mapping[str, Any]]]],
) -> list[str]:
    """Return the sorted agent names seen across blue/red run contexts."""

    names: set[str] = set()
    for _, records in loaded_runs:
        context = _run_context(records)
        for key in ("blue_agent", "red_agent"):
            if isinstance((value := context.get(key)), str) and value:
                names.add(value)
    return sorted(names)


def _aggregate_win_rates_by_scenario(
    loaded_runs: Sequence[tuple[Path, Sequence[Mapping[str, Any]]]],
) -> dict[str, dict[str, float]]:
    """Aggregate per-scenario win rates for agents across both factions."""

    grouped: dict[str, dict[str, dict[Faction, list[Sequence[Mapping[str, Any]]]]]] = defaultdict(
        lambda: defaultdict(lambda: {Faction.BLUE: [], Faction.RED: []})
    )

    for path, records in loaded_runs:
        context = _run_context(records)
        scenario_id = _scenario_id_for_run(path, records)

        if isinstance((blue_agent := context.get("blue_agent")), str) and blue_agent:
            grouped[scenario_id][blue_agent][Faction.BLUE].append(records)
        if isinstance((red_agent := context.get("red_agent")), str) and red_agent:
            grouped[scenario_id][red_agent][Faction.RED].append(records)

    result: dict[str, dict[str, float]] = {}
    for scenario_id in sorted(grouped):
        result[scenario_id] = {}
        for agent_name in sorted(grouped[scenario_id]):
            blue_runs = grouped[scenario_id][agent_name][Faction.BLUE]
            red_runs = grouped[scenario_id][agent_name][Faction.RED]
            total_runs = len(blue_runs) + len(red_runs)
            if total_runs == 0:
                continue

            blue_wins = (
                win_rate(blue_runs, faction=Faction.BLUE) * len(blue_runs)
                if blue_runs
                else 0.0
            )
            red_wins = (
                win_rate(red_runs, faction=Faction.RED) * len(red_runs)
                if red_runs
                else 0.0
            )
            result[scenario_id][agent_name] = (blue_wins + red_wins) / total_runs
    return result


def _collect_metric_by_agent(
    loaded_runs: Sequence[tuple[Path, Sequence[Mapping[str, Any]]]],
    *,
    metric: Callable[[Sequence[Mapping[str, Any]]], float],
) -> dict[str, list[float]]:
    """Collect one per-run metric series for each agent across both factions."""

    series: dict[str, list[float]] = defaultdict(list)
    for _, records in loaded_runs:
        context = _run_context(records)
        for faction_key, context_key in (("blue", "blue_agent"), ("red", "red_agent")):
            if not isinstance((agent_name := context.get(context_key)), str) or not agent_name:
                continue
            decision_records = _decision_records_for_faction(records, faction_key)
            if not decision_records:
                continue
            series[agent_name].append(metric(decision_records))
    return {
        label: values
        for label, values in sorted(series.items())
        if values
    }


def _decision_records_for_faction(
    records: Sequence[Mapping[str, Any]],
    faction_key: str,
) -> list[dict[str, Any]]:
    """Project one run's turn records down to one faction's decisions only."""

    projected: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        decision = metadata.get(faction_key)
        if not isinstance(decision, Mapping):
            continue

        raw_actions = decision.get("actions")
        actions = (
            [dict(action) for action in raw_actions if isinstance(action, Mapping)]
            if isinstance(raw_actions, list)
            else []
        )
        decision_metadata = decision.get("metadata")
        projected.append(
            {
                "turn": record.get("turn", 0),
                "actions": actions,
                "metadata": {
                    faction_key: {
                        "used_fallback": bool(decision.get("used_fallback", False)),
                        "metadata": (
                            dict(decision_metadata)
                            if isinstance(decision_metadata, Mapping)
                            else {}
                        ),
                    }
                },
            }
        )
    return projected


def _write_force_curves(
    loaded_runs: Sequence[tuple[Path, Sequence[Mapping[str, Any]]]],
    output_dir: Path,
) -> list[Path]:
    """Write one PNG/SVG force curve per run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    seen_names: dict[str, int] = {}
    written: list[Path] = []

    for path, records in loaded_runs:
        run_label = _unique_run_label(_run_label_for_output(path, records), seen_names)
        title = _force_curve_title(path, records, run_label)
        written.extend(
            plot_force_curve(
                records,
                output_dir / _slugify(run_label),
                title=title,
            )
        )
    return written


def _write_battlefield_replays(
    loaded_runs: Sequence[tuple[Path, Sequence[Mapping[str, Any]]]],
    output_dir: Path,
) -> list[Path]:
    """Write one battlefield replay directory per run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    seen_names: dict[str, int] = {}
    written: list[Path] = []

    for path, records in loaded_runs:
        run_label = _unique_run_label(_run_label_for_output(path, records), seen_names)
        title = _force_curve_title(path, records, run_label)
        bundle = build_battlefield_replay(
            records,
            output_dir / _slugify(run_label),
            run_label=run_label,
            title=title,
        )
        written.extend(bundle.files)
    return written


def _build_replay_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    run_label: str,
    title: str | None,
) -> dict[str, Any]:
    """Normalize one run's records into replay-friendly frame data."""

    terrain_by_hex = _replay_terrain_lookup(records)
    objective_hexes = _replay_objective_hexes(records)
    bounds = _replay_bounds(records, terrain_by_hex, objective_hexes)
    frames: list[dict[str, Any]] = []
    previous_positions: dict[str, dict[str, int]] = {}

    ordered_records = sorted(records, key=lambda record: int(record.get("turn", 0)))
    for record in ordered_records:
        turn = int(record.get("turn", 0))
        units = _replay_units(record)
        unit_lookup = {unit["unit_id"]: unit for unit in units}
        current_positions = {
            unit["unit_id"]: dict(unit["position"])
            for unit in units
            if isinstance(unit.get("position"), Mapping)
        }
        combat = record.get("combat")
        combat_mapping = combat if isinstance(combat, Mapping) else {}
        raw_casualties = combat_mapping.get("casualties_by_unit")
        casualty_mapping = raw_casualties if isinstance(raw_casualties, Mapping) else {}
        casualties_by_unit = {
            str(unit_id): int(loss)
            for unit_id, loss in casualty_mapping.items()
            if isinstance(loss, (int, float))
        }
        engaged_unit_ids = sorted(
            {
                unit_id
                for key in ("attacker_ids", "defender_ids")
                for unit_id in combat_mapping.get(key, [])
                if isinstance(combat_mapping.get(key), list) and isinstance(unit_id, str)
            }
        )
        moved_unit_ids = sorted(
            unit_id
            for unit_id, position in current_positions.items()
            if previous_positions.get(unit_id) is not None and previous_positions[unit_id] != position
        )
        frame = {
            "turn": turn,
            "file_name": f"turn_{turn:03d}.svg",
            "caption": _replay_caption(turn, units, casualties_by_unit, combat_mapping),
            "units": units,
            "engaged_unit_ids": engaged_unit_ids,
            "moved_unit_ids": moved_unit_ids,
            "casualties_by_unit": casualties_by_unit,
            "combat_summary": str(combat_mapping.get("summary", "")),
            "winner": combat_mapping.get("winner"),
            "notes": [
                str(note)
                for note in record.get("notes", [])
                if isinstance(record.get("notes"), list) and isinstance(note, str)
            ],
            "action_overlays": _replay_action_overlays(
                record,
                unit_lookup=unit_lookup,
                previous_positions=previous_positions,
            ),
        }
        frames.append(frame)
        previous_positions = current_positions

    return {
        "run_label": run_label,
        "title": title or run_label,
        "frame_count": len(frames),
        "bounds": bounds,
        "terrain_by_hex": dict(sorted(terrain_by_hex.items())),
        "objective_hexes": objective_hexes,
        "frames": frames,
    }


def _replay_terrain_lookup(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Merge terrain cells from serialized turn states into one lookup table."""

    terrain_by_hex: dict[str, str] = {}
    for record in records:
        state = record.get("state")
        if not isinstance(state, Mapping):
            continue
        raw_terrain = state.get("terrain_by_hex")
        if not isinstance(raw_terrain, Mapping):
            continue
        for key, value in raw_terrain.items():
            position = _serialized_position(key)
            if position is None:
                continue
            terrain_by_hex[_position_key(position["q"], position["r"])] = (
                str(value) if isinstance(value, str) else "open"
            )
    return terrain_by_hex


def _replay_objective_hexes(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, int]]]:
    """Recover per-faction objective hexes from serialized state metadata."""

    objectives: dict[str, list[dict[str, int]]] = {"blue": [], "red": []}
    for record in records:
        state = record.get("state")
        if not isinstance(state, Mapping):
            continue
        metadata = state.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        raw_objectives = metadata.get("objective_hexes")
        if not isinstance(raw_objectives, Mapping):
            continue
        for faction in ("blue", "red"):
            payload = raw_objectives.get(faction)
            if not isinstance(payload, list):
                continue
            if objectives[faction]:
                continue
            objectives[faction] = [
                position
                for item in payload
                if isinstance((position := _serialized_position(item)), dict)
            ]
    return objectives


def _replay_bounds(
    records: Sequence[Mapping[str, Any]],
    terrain_by_hex: Mapping[str, str],
    objective_hexes: Mapping[str, Sequence[Mapping[str, int]]],
) -> dict[str, int]:
    """Infer a stable replay viewport from terrain, units, objectives, and actions."""

    positions: list[dict[str, int]] = []
    for key in terrain_by_hex:
        if isinstance((position := _serialized_position(key)), dict):
            positions.append(position)
    for items in objective_hexes.values():
        positions.extend(
            position
            for position in items
            if isinstance(position, Mapping)
        )
    for record in records:
        positions.extend(_positions_from_record(record))

    if not positions:
        return {"min_q": 0, "max_q": 0, "min_r": 0, "max_r": 0}

    min_q = min(position["q"] for position in positions)
    max_q = max(position["q"] for position in positions)
    min_r = min(position["r"] for position in positions)
    max_r = max(position["r"] for position in positions)
    return {
        "min_q": max(0, min_q - 1),
        "max_q": max_q + 1,
        "min_r": max(0, min_r - 1),
        "max_r": max_r + 1,
    }


def _positions_from_record(record: Mapping[str, Any]) -> list[dict[str, int]]:
    """Collect all replay-relevant positions from one serialized turn record."""

    positions: list[dict[str, int]] = []
    state = record.get("state")
    if isinstance(state, Mapping):
        units = state.get("units")
        if isinstance(units, Mapping):
            for unit in units.values():
                if isinstance(unit, Mapping) and isinstance((position := _serialized_position(unit.get("position"))), dict):
                    positions.append(position)
    actions = record.get("actions")
    if isinstance(actions, list):
        for action in actions:
            if isinstance(action, Mapping) and isinstance((position := _serialized_position(action.get("target_hex"))), dict):
                positions.append(position)
    return positions


def _replay_units(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract a normalized unit list from one serialized state snapshot."""

    state = record.get("state")
    if not isinstance(state, Mapping):
        return []
    units = state.get("units")
    if not isinstance(units, Mapping):
        return []

    normalized: list[dict[str, Any]] = []
    for unit_id, unit in sorted(units.items()):
        if not isinstance(unit_id, str) or not isinstance(unit, Mapping):
            continue
        position = _serialized_position(unit.get("position"))
        normalized.append(
            {
                "unit_id": unit_id,
                "faction": str(unit.get("faction", "")),
                "position": position,
                "strength": int(unit.get("strength", 0)) if isinstance(unit.get("strength"), (int, float)) else 0,
                "status": str(unit.get("status", "")),
                "posture": str(unit.get("posture", "")),
            }
        )
    return normalized


def _replay_action_overlays(
    record: Mapping[str, Any],
    *,
    unit_lookup: Mapping[str, Mapping[str, Any]],
    previous_positions: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    """Build line overlays that show unit intent on the battlefield replay."""

    overlays: list[dict[str, Any]] = []
    actions = record.get("actions")
    if not isinstance(actions, list):
        return overlays

    for action in actions:
        if not isinstance(action, Mapping):
            continue
        unit_id = action.get("unit_id")
        target = _serialized_position(action.get("target_hex"))
        if not isinstance(unit_id, str) or target is None:
            continue
        origin = previous_positions.get(unit_id)
        if origin is None and isinstance((unit := unit_lookup.get(unit_id)), Mapping):
            unit_position = unit.get("position")
            origin = dict(unit_position) if isinstance(unit_position, Mapping) else None
        if origin is None:
            continue
        unit = unit_lookup.get(unit_id, {})
        overlays.append(
            {
                "unit_id": unit_id,
                "faction": str(unit.get("faction", "")),
                "action_type": str(action.get("action_type", "")),
                "origin": dict(origin),
                "target": target,
            }
        )
    return overlays


def _replay_caption(
    turn: int,
    units: Sequence[Mapping[str, Any]],
    casualties_by_unit: Mapping[str, int],
    combat: Mapping[str, Any],
) -> str:
    """Build a concise caption line for one replay frame."""

    blue_total = sum(
        int(unit.get("strength", 0))
        for unit in units
        if unit.get("faction") == Faction.BLUE.value
    )
    red_total = sum(
        int(unit.get("strength", 0))
        for unit in units
        if unit.get("faction") == Faction.RED.value
    )
    total_casualties = sum(casualties_by_unit.values())
    winner = combat.get("winner")
    winner_text = f" | winner={winner}" if isinstance(winner, str) and winner else ""
    return (
        f"Turn {turn} | blue={blue_total} | red={red_total} "
        f"| casualties={total_casualties}{winner_text}"
    )


def _build_battlefield_frame_svg(frame: Mapping[str, Any], replay: Mapping[str, Any]) -> str:
    """Render one turn snapshot as an SVG battlefield frame."""

    bounds = replay.get("bounds", {})
    min_q = int(bounds.get("min_q", 0))
    max_q = int(bounds.get("max_q", 0))
    min_r = int(bounds.get("min_r", 0))
    max_r = int(bounds.get("max_r", 0))
    size = 22.0
    margin = 36.0
    side_panel = 320.0
    cells = [
        {"q": q, "r": r}
        for r in range(min_r, max_r + 1)
        for q in range(min_q, max_q + 1)
    ]
    centers = [_axial_to_pixel(cell["q"], cell["r"], size=size) for cell in cells]
    min_x = min((center["x"] for center in centers), default=0.0) - size
    max_x = max((center["x"] for center in centers), default=0.0) + size
    min_y = min((center["y"] for center in centers), default=0.0) - size
    max_y = max((center["y"] for center in centers), default=0.0) + size
    grid_width = max_x - min_x
    grid_height = max_y - min_y
    width = grid_width + margin * 2 + side_panel
    height = max(grid_height + margin * 2, 480.0)
    offset_x = margin - min_x
    offset_y = margin - min_y
    terrain_by_hex = replay.get("terrain_by_hex", {})
    objective_hexes = replay.get("objective_hexes", {})
    units = frame.get("units", [])

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        "<defs>",
        '<marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#1d4ed8" /></marker>',
        '<marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#b91c1c" /></marker>',
        "</defs>",
        '<rect width="100%" height="100%" fill="#f7f7f2"/>',
    ]

    for cell in cells:
        center = _axial_to_pixel(cell["q"], cell["r"], size=size, offset_x=offset_x, offset_y=offset_y)
        key = _position_key(cell["q"], cell["r"])
        terrain = terrain_by_hex.get(key, "open")
        points = _hex_polygon_points(center["x"], center["y"], size=size)
        elements.append(
            f'<polygon points="{points}" fill="{_terrain_fill(terrain)}" stroke="#adb0aa" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{center["x"]:.1f}" y="{center["y"] + 4:.1f}" text-anchor="middle" font-size="8" fill="#6b7280">{cell["q"]},{cell["r"]}</text>'
        )

    for faction, color in ((Faction.BLUE.value, "#1d4ed8"), (Faction.RED.value, "#b91c1c")):
        payload = objective_hexes.get(faction, [])
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            center = _axial_to_pixel(
                int(item.get("q", 0)),
                int(item.get("r", 0)),
                size=size,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            points = _hex_polygon_points(center["x"], center["y"], size=size - 3)
            elements.append(
                f'<polygon points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="6 3"/>'
            )

    for overlay in frame.get("action_overlays", []):
        if not isinstance(overlay, Mapping):
            continue
        origin = overlay.get("origin")
        target = overlay.get("target")
        if not isinstance(origin, Mapping) or not isinstance(target, Mapping):
            continue
        start = _axial_to_pixel(
            int(origin.get("q", 0)),
            int(origin.get("r", 0)),
            size=size,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        end = _axial_to_pixel(
            int(target.get("q", 0)),
            int(target.get("r", 0)),
            size=size,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        faction = str(overlay.get("faction", ""))
        color = "#1d4ed8" if faction == Faction.BLUE.value else "#b91c1c"
        marker = "arrow-blue" if faction == Faction.BLUE.value else "arrow-red"
        dash = "5 3" if str(overlay.get("action_type", "")) in {"attack", "support_by_fire"} else "2 3"
        elements.append(
            f'<line x1="{start["x"]:.1f}" y1="{start["y"]:.1f}" x2="{end["x"]:.1f}" y2="{end["y"]:.1f}" '
            f'stroke="{color}" stroke-width="2" stroke-dasharray="{dash}" opacity="0.75" marker-end="url(#{marker})"/>'
        )

    units_by_hex: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for unit in units:
        if isinstance(unit, Mapping) and isinstance((position := unit.get("position")), Mapping):
            units_by_hex[_position_key(int(position.get("q", 0)), int(position.get("r", 0)))].append(unit)

    casualties = frame.get("casualties_by_unit", {})
    engaged_unit_ids = {
        unit_id
        for unit_id in frame.get("engaged_unit_ids", [])
        if isinstance(frame.get("engaged_unit_ids"), list) and isinstance(unit_id, str)
    }

    for key, stacked_units in sorted(units_by_hex.items()):
        position = _serialized_position(key)
        if position is None:
            continue
        center = _axial_to_pixel(
            position["q"],
            position["r"],
            size=size,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        for unit, slot in zip(stacked_units, _stacked_unit_offsets(len(stacked_units)), strict=False):
            faction = str(unit.get("faction", ""))
            fill = "#2563eb" if faction == Faction.BLUE.value else "#dc2626"
            circle_x = center["x"] + slot["dx"]
            circle_y = center["y"] + slot["dy"]
            unit_id = str(unit.get("unit_id", "unit"))
            stroke = "#111827"
            stroke_width = 2
            if unit_id in engaged_unit_ids:
                stroke = "#f59e0b"
                stroke_width = 3
            if isinstance(casualties, Mapping) and int(casualties.get(unit_id, 0)) > 0:
                stroke = "#111827"
                stroke_width = 4
            elements.append(
                f'<circle cx="{circle_x:.1f}" cy="{circle_y:.1f}" r="12" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
            )
            elements.append(
                f'<text x="{circle_x:.1f}" y="{circle_y + 4:.1f}" text-anchor="middle" font-size="9" font-weight="700" fill="#ffffff">{escape(_short_unit_label(unit_id))}</text>'
            )
            elements.append(
                f'<text x="{circle_x:.1f}" y="{circle_y + 22:.1f}" text-anchor="middle" font-size="9" fill="#1f2937">{int(unit.get("strength", 0))}</text>'
            )

    panel_x = grid_width + margin * 2 + 18
    title_text = escape(str(replay.get("title", "Battlefield Replay")))
    caption_text = escape(str(frame.get("caption", "")))
    elements.extend(
        [
            f'<text x="{panel_x}" y="44" font-size="22" font-weight="700" fill="#111827">{title_text}</text>',
            f'<text x="{panel_x}" y="74" font-size="14" fill="#374151">{caption_text}</text>',
            f'<text x="{panel_x}" y="114" font-size="14" font-weight="700" fill="#111827">Legend</text>',
            f'<circle cx="{panel_x + 14}" cy="138" r="10" fill="#2563eb" stroke="#111827" stroke-width="2"/>',
            f'<text x="{panel_x + 34}" y="143" font-size="12" fill="#374151">Blue unit</text>',
            f'<circle cx="{panel_x + 14}" cy="164" r="10" fill="#dc2626" stroke="#111827" stroke-width="2"/>',
            f'<text x="{panel_x + 34}" y="169" font-size="12" fill="#374151">Red unit</text>',
            f'<circle cx="{panel_x + 14}" cy="190" r="10" fill="#dc2626" stroke="#f59e0b" stroke-width="3"/>',
            f'<text x="{panel_x + 34}" y="195" font-size="12" fill="#374151">Engaged this turn</text>',
        ]
    )

    sidebar_lines = _replay_sidebar_lines(frame)
    base_y = 230
    for index, line in enumerate(sidebar_lines):
        elements.append(
            f'<text x="{panel_x}" y="{base_y + index * 18}" font-size="12" fill="#374151">{escape(line)}</text>'
        )

    elements.append("</svg>")
    return "\n".join(elements)


def _replay_sidebar_lines(frame: Mapping[str, Any]) -> list[str]:
    """Build readable sidebar text for one replay frame."""

    lines: list[str] = []
    winner = frame.get("winner")
    if isinstance(winner, str) and winner:
        lines.append(f"Winner: {winner}")

    casualties = frame.get("casualties_by_unit", {})
    if isinstance(casualties, Mapping) and casualties:
        lines.append("Casualties:")
        for unit_id, loss in sorted(casualties.items()):
            lines.append(f"  {unit_id}: -{int(loss)}")

    combat_summary = frame.get("combat_summary")
    if isinstance(combat_summary, str) and combat_summary:
        lines.append("Combat:")
        lines.extend(_wrap_text(combat_summary, width=40))

    notes = frame.get("notes", [])
    if isinstance(notes, list) and notes:
        lines.append("Notes:")
        for note in notes:
            if isinstance(note, str):
                lines.extend(_wrap_text(note, width=40))

    if not lines:
        lines.append("No detailed notes recorded for this turn.")
    return lines


def _build_replay_html(replay: Mapping[str, Any]) -> str:
    """Generate a small self-contained HTML viewer for replay SVG frames."""

    payload = json.dumps(replay, ensure_ascii=False).replace("</", "<\\/")
    title = escape(str(replay.get("title", "Battlefield Replay")))
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3f1ea;
      --panel: #fffdf8;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #d6d3cd;
      --accent: #1d4ed8;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f7f5ef 0%, #ece8dd 100%);
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 12px 30px rgba(17, 24, 39, 0.08);
      overflow: hidden;
    }}
    .header {{
      padding: 20px 24px 8px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
    }}
    .muted {{
      color: var(--muted);
      font-size: 14px;
    }}
    .controls {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      padding: 16px 24px;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.7);
    }}
    button {{
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 999px;
      padding: 9px 14px;
      font-size: 14px;
      cursor: pointer;
    }}
    button:hover {{
      border-color: var(--accent);
      color: var(--accent);
    }}
    input[type="range"] {{
      flex: 1 1 260px;
    }}
    .viewer {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      gap: 0;
    }}
    .frame {{
      background: #f7f7f2;
      min-height: 480px;
    }}
    .frame img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .sidebar {{
      border-left: 1px solid var(--line);
      padding: 20px;
      background: rgba(255, 255, 255, 0.86);
    }}
    .sidebar h2 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .sidebar pre {{
      white-space: pre-wrap;
      word-break: keep-all;
      font-family: inherit;
      font-size: 13px;
      line-height: 1.6;
      margin: 0;
      color: var(--ink);
    }}
    @media (max-width: 960px) {{
      .viewer {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        border-left: 0;
        border-top: 1px solid var(--line);
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="header">
        <h1>{title}</h1>
        <div class="muted">턴별 상태 스냅샷, 이동 의도선, 교전/피해 강조를 포함한 전장 replay</div>
      </div>
      <div class="controls">
        <button id="prev-btn" type="button">이전</button>
        <button id="play-btn" type="button">재생</button>
        <button id="next-btn" type="button">다음</button>
        <input id="turn-slider" type="range" min="0" max="0" value="0" />
        <span id="turn-label" class="muted"></span>
      </div>
      <div class="viewer">
        <div class="frame">
          <img id="frame-img" alt="Battlefield replay frame" />
        </div>
        <aside class="sidebar">
          <h2 id="frame-title">Turn</h2>
          <pre id="frame-notes"></pre>
        </aside>
      </div>
    </div>
  </div>
  <script id="replay-data" type="application/json">{payload}</script>
  <script>
    const replay = JSON.parse(document.getElementById("replay-data").textContent);
    const frames = Array.isArray(replay.frames) ? replay.frames : [];
    const slider = document.getElementById("turn-slider");
    const turnLabel = document.getElementById("turn-label");
    const frameTitle = document.getElementById("frame-title");
    const frameNotes = document.getElementById("frame-notes");
    const frameImg = document.getElementById("frame-img");
    const playBtn = document.getElementById("play-btn");
    let current = 0;
    let timer = null;

    slider.max = Math.max(frames.length - 1, 0);

    function describe(frame) {{
      const lines = [];
      if (frame.caption) lines.push(frame.caption);
      if (frame.winner) lines.push(`winner: ${{frame.winner}}`);
      const casualties = frame.casualties_by_unit || {{}};
      const casualtyLines = Object.entries(casualties).map(([unitId, loss]) => `${{unitId}}: -${{loss}}`);
      if (casualtyLines.length) {{
        lines.push("");
        lines.push("casualties");
        lines.push(...casualtyLines);
      }}
      if (frame.combat_summary) {{
        lines.push("");
        lines.push("combat");
        lines.push(frame.combat_summary);
      }}
      if (Array.isArray(frame.notes) && frame.notes.length) {{
        lines.push("");
        lines.push("notes");
        lines.push(...frame.notes);
      }}
      return lines.join("\\n");
    }}

    function render(index) {{
      if (!frames.length) {{
        frameTitle.textContent = "No frames";
        frameNotes.textContent = "This replay did not contain any turn snapshots.";
        frameImg.removeAttribute("src");
        turnLabel.textContent = "0 / 0";
        return;
      }}
      current = Math.min(Math.max(index, 0), frames.length - 1);
      const frame = frames[current];
      slider.value = String(current);
      turnLabel.textContent = `${{current + 1}} / ${{frames.length}}`;
      frameTitle.textContent = `Turn ${{frame.turn}}`;
      frameNotes.textContent = describe(frame);
      frameImg.src = frame.file_name;
    }}

    function togglePlay() {{
      if (timer !== null) {{
        clearInterval(timer);
        timer = null;
        playBtn.textContent = "재생";
        return;
      }}
      playBtn.textContent = "정지";
      timer = setInterval(() => {{
        if (current >= frames.length - 1) {{
          clearInterval(timer);
          timer = null;
          playBtn.textContent = "재생";
          return;
        }}
        render(current + 1);
      }}, 1200);
    }}

    document.getElementById("prev-btn").addEventListener("click", () => render(current - 1));
    document.getElementById("next-btn").addEventListener("click", () => render(current + 1));
    document.getElementById("play-btn").addEventListener("click", togglePlay);
    slider.addEventListener("input", (event) => render(Number(event.target.value)));

    render(0);
  </script>
</body>
</html>
"""


def _wrap_text(text: str, *, width: int) -> list[str]:
    """Wrap sidebar text into approximately equal-width lines."""

    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return lines


def _terrain_fill(terrain: str) -> str:
    """Map terrain names to replay-friendly cell colors."""

    return {
        "open": "#ece7d8",
        "forest": "#b9d3a9",
        "mountain": "#c7c7c7",
        "urban": "#d9c4b0",
        "river": "#9ac7ec",
    }.get(terrain, "#ece7d8")


def _axial_to_pixel(
    q: int,
    r: int,
    *,
    size: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> dict[str, float]:
    """Convert axial coordinates into pointy-top hex pixel centers."""

    x = size * math.sqrt(3) * (q + r / 2) + offset_x
    y = size * 1.5 * r + offset_y
    return {"x": x, "y": y}


def _hex_polygon_points(cx: float, cy: float, *, size: float) -> str:
    """Return a polygon point string for a pointy-top hex."""

    points = []
    for index in range(6):
        angle = math.radians(60 * index - 30)
        x = cx + size * math.cos(angle)
        y = cy + size * math.sin(angle)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _stacked_unit_offsets(count: int) -> list[dict[str, float]]:
    """Return small offsets so multiple units in one hex remain legible."""

    if count <= 1:
        return [{"dx": 0.0, "dy": 0.0}]
    if count == 2:
        return [{"dx": -10.0, "dy": 0.0}, {"dx": 10.0, "dy": 0.0}]
    if count == 3:
        return [{"dx": -10.0, "dy": 6.0}, {"dx": 10.0, "dy": 6.0}, {"dx": 0.0, "dy": -10.0}]
    offsets: list[dict[str, float]] = []
    radius = 12.0
    for index in range(count):
        angle = 2 * math.pi * index / count
        offsets.append({"dx": radius * math.cos(angle), "dy": radius * math.sin(angle)})
    return offsets


def _short_unit_label(unit_id: str) -> str:
    """Compress a unit id into a short on-map label."""

    if "-" in unit_id:
        return unit_id.split("-", maxsplit=1)[1][:4]
    return unit_id[:4]


def _serialized_position(value: Any) -> dict[str, int] | None:
    """Normalize position-like payloads into ``{q, r}`` dictionaries."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        q = value.get("q")
        r = value.get("r")
        if isinstance(q, int) and isinstance(r, int):
            return {"q": q, "r": r}
        return None
    if hasattr(value, "q") and hasattr(value, "r"):
        try:
            return {"q": int(value.q), "r": int(value.r)}
        except (TypeError, ValueError):
            return None
    if isinstance(value, str) and "," in value:
        q_text, r_text = value.split(",", maxsplit=1)
        try:
            return {"q": int(q_text), "r": int(r_text)}
        except ValueError:
            return None
    return None


def _position_key(q: int, r: int) -> str:
    """Convert one axial coordinate into the serialized terrain-key form."""

    return f"{q},{r}"


def _run_context(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Return the first available run context from serialized turn records."""

    for record in records:
        context = record.get("context")
        if isinstance(context, Mapping):
            return context
    return {}


def _scenario_id_for_run(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    """Recover a scenario identifier from run context or file path."""

    context = _run_context(records)
    if isinstance((scenario_id := context.get("scenario_id")), str) and scenario_id:
        return scenario_id
    return path.parent.name or path.stem


def _run_label_for_output(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    """Choose a stable run label for output filenames."""

    context = _run_context(records)
    if isinstance((run_id := context.get("run_id")), str) and run_id:
        return run_id
    return path.stem


def _force_curve_title(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    run_label: str,
) -> str:
    """Build a readable force-curve title from run metadata."""

    context = _run_context(records)
    scenario_name = context.get("scenario_name")
    if isinstance(scenario_name, str) and scenario_name:
        return f"Force Strength Over Turns - {scenario_name} ({run_label})"
    scenario_id = _scenario_id_for_run(path, records)
    return f"Force Strength Over Turns - {scenario_id} ({run_label})"


def _unique_run_label(label: str, seen_names: dict[str, int]) -> str:
    """Deduplicate a preferred run label without losing readability."""

    count = seen_names.get(label, 0)
    seen_names[label] = count + 1
    if count == 0:
        return label
    return f"{label}-{count + 1}"


def _string_list(value: Any) -> list[str]:
    """Normalize a JSON-like list payload into strings."""

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _require_matplotlib():
    """Import matplotlib; raise a helpful error when absent."""
    try:
        import matplotlib as mpl  # noqa: PLC0415
        import matplotlib.pyplot as plt  # noqa: PLC0415

        return mpl, plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for chart generation. "
            "Install with: pip install matplotlib"
        ) from exc


def _apply_academic_style(mpl) -> None:
    """Apply publication-friendly rcParams (greyscale-safe, 300 DPI)."""
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            # Prefer system fonts with Korean glyph coverage; fall back gracefully.
            "font.sans-serif": [
                "Apple SD Gothic Neo",
                "Noto Sans CJK KR",
                "Malgun Gothic",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.4,
            "figure.constrained_layout.use": False,
        }
    )


def _apply_bw_boxplot(bp) -> None:
    """Make a boxplot print-safe (white fill, black edges, grey median)."""
    for patch in bp.get("boxes", []):
        patch.set_facecolor("white")
        patch.set_edgecolor("black")
    for median in bp.get("medians", []):
        median.set_color("black")
        median.set_linewidth(1.5)
    for element in ("whiskers", "caps", "fliers"):
        for item in bp.get(element, []):
            item.set_color("black")


def _boxplot(ax, values: Sequence[Sequence[float]], labels: Sequence[str]):
    """Call ``Axes.boxplot`` across old/new matplotlib label parameter names."""

    try:
        return ax.boxplot(values, tick_labels=labels, patch_artist=True, notch=False)
    except TypeError:
        return ax.boxplot(values, labels=labels, patch_artist=True, notch=False)


def _save_figure(fig, plt, base_path: Path) -> list[Path]:
    """Save *fig* as both PNG and SVG next to *base_path* stem."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in ("png", "svg"):
        path = base_path.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def _hatch_cycle():
    """Infinite iterator over print-safe hatch patterns."""
    import itertools  # noqa: PLC0415

    return itertools.cycle(["", "///", "xxx", "...", "---", "|||"])


def _linestyle_cycle():
    """Infinite iterator over print-safe linestyles."""
    import itertools  # noqa: PLC0415

    return itertools.cycle(["-", "--", "-.", ":"])


def _slugify(text: str) -> str:
    """Convert a label to a filesystem-safe slug."""

    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text).strip("_")


# ---------------------------------------------------------------------------
# SVG helpers (no dependencies)
# ---------------------------------------------------------------------------


def _force_totals(records: Sequence[Mapping[str, Any]], faction: str) -> list[int]:
    """Recover per-turn force totals from serialized state snapshots."""

    totals: list[int] = []
    for record in records:
        state = record.get("state")
        if not isinstance(state, Mapping):
            continue
        units = state.get("units")
        if not isinstance(units, Mapping):
            continue
        totals.append(
            sum(
                int(unit.get("strength", 0))
                for unit in units.values()
                if isinstance(unit, Mapping) and unit.get("faction") == faction
            )
        )
    return totals


def _build_force_totals_svg(records: Sequence[Mapping[str, Any]]) -> str:
    """Build a lightweight SVG line chart for blue/red force totals."""

    blue = _force_totals(records, "blue")
    red = _force_totals(records, "red")
    width = 640
    height = 320
    padding = 40
    points = max(len(blue), len(red), 1)
    max_total = max(blue + red + [1])

    def project(series: list[int]) -> str:
        if not series:
            return ""
        coordinates = []
        for index, value in enumerate(series):
            x = padding + ((width - 2 * padding) * index / max(points - 1, 1))
            y = height - padding - ((height - 2 * padding) * value / max_total)
            coordinates.append(f"{x:.1f},{y:.1f}")
        return " ".join(coordinates)

    blue_path = project(blue)
    red_path = project(red)
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f8f8f4"/>',
            f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#444" />',
            f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#444" />',
            '<text x="40" y="24" font-size="16" fill="#222">Force totals by turn</text>',
            f'<polyline fill="none" stroke="#0b6efd" stroke-width="3" points="{blue_path}" />',
            f'<polyline fill="none" stroke="#d63384" stroke-width="3" points="{red_path}" />',
            '<text x="520" y="30" font-size="12" fill="#0b6efd">Blue</text>',
            '<text x="520" y="48" font-size="12" fill="#d63384">Red</text>',
            "</svg>",
        ]
    )
