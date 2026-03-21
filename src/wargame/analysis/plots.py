"""Lightweight plot generation for experiment analysis.

The original SVG/JSON generation works without any third-party libraries.
The matplotlib-based functions below are gated behind a lazy import: if
matplotlib is not installed they raise ``ImportError`` with a clear message
rather than crashing at module import time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Public data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlotBundle:
    """Container describing generated experiment-analysis artifacts."""

    output_dir: Path
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

    bp = ax.boxplot(values, labels=labels, patch_artist=True, notch=False)
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
    bp = ax.boxplot(values, labels=labels, patch_artist=True, notch=False)
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
