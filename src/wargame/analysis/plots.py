"""Lightweight plot generation for experiment analysis."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PlotBundle:
    """Container describing generated experiment-analysis artifacts."""

    output_dir: Path
    files: list[Path] = field(default_factory=list)


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
