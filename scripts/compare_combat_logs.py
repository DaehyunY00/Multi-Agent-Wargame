"""CLI utility for comparing deterministic and stochastic combat logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wargame.analysis import (  # noqa: E402
    aggregate_casualties_by_unit,
    combat_turn_count,
    load_jsonl_records,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Compare two JSONL runs and print a combat-focused JSON summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deterministic_log", help="Path to the deterministic combat JSONL log.")
    parser.add_argument("stochastic_log", help="Path to the stochastic combat JSONL log.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    deterministic = _build_log_summary(Path(args.deterministic_log))
    stochastic = _build_log_summary(Path(args.stochastic_log))
    comparison = {
        "deterministic": deterministic,
        "stochastic": stochastic,
        "delta": {
            "turn_count": stochastic["turn_count"] - deterministic["turn_count"],
            "combat_turn_count": stochastic["combat_turn_count"] - deterministic["combat_turn_count"],
            "total_casualties": stochastic["total_casualties"] - deterministic["total_casualties"],
            "blue_final_strength": (
                stochastic["final_force_totals"]["blue"]
                - deterministic["final_force_totals"]["blue"]
            ),
            "red_final_strength": (
                stochastic["final_force_totals"]["red"]
                - deterministic["final_force_totals"]["red"]
            ),
            "casualty_delta_by_unit": _subtract_casualties(
                deterministic["casualties_by_unit"],
                stochastic["casualties_by_unit"],
            ),
        },
    }
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


def _build_log_summary(path: Path) -> dict[str, Any]:
    """Build a compact combat summary from one JSONL log."""

    records = load_jsonl_records(path)
    casualties_by_unit = aggregate_casualties_by_unit(records)
    final_force_totals = _final_force_totals(records)
    return {
        "path": str(path),
        "turn_count": len(records),
        "combat_turn_count": combat_turn_count(records),
        "total_casualties": sum(casualties_by_unit.values()),
        "casualties_by_unit": casualties_by_unit,
        "final_force_totals": final_force_totals,
    }


def _final_force_totals(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Recover the last logged total strengths for blue and red."""

    if not records:
        return {"blue": 0, "red": 0}

    final_record = sorted(records, key=lambda record: int(record.get("turn", 0)))[-1]
    state = final_record.get("state")
    if not isinstance(state, Mapping):
        return {"blue": 0, "red": 0}

    units = state.get("units")
    if not isinstance(units, Mapping):
        return {"blue": 0, "red": 0}

    totals = {"blue": 0, "red": 0}
    for unit in units.values():
        if not isinstance(unit, Mapping):
            continue
        faction = unit.get("faction")
        strength = unit.get("strength")
        if faction in totals and isinstance(strength, (int, float)):
            totals[str(faction)] += int(strength)
    return totals


def _subtract_casualties(
    deterministic: Mapping[str, Any],
    stochastic: Mapping[str, Any],
) -> dict[str, int]:
    """Return stochastic-minus-deterministic casualty differences by unit."""

    unit_ids = sorted(set(deterministic) | set(stochastic))
    return {
        unit_id: int(stochastic.get(unit_id, 0)) - int(deterministic.get(unit_id, 0))
        for unit_id in unit_ids
    }


if __name__ == "__main__":
    raise SystemExit(main())
