"""Log-driven experiment metrics for tactical wargame runs."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from wargame.core.enums import ActionType, Faction

LogRecord: TypeAlias = dict[str, Any]
LogSource: TypeAlias = str | Path | Sequence[Mapping[str, Any]]
MetricHook: TypeAlias = Callable[[Mapping[str, Any], str], float | None]

ACTION_ESCALATION_WEIGHTS: dict[str, float] = {
    ActionType.HOLD.value: 0.0,
    ActionType.RECON.value: 0.25,
    ActionType.MOVE.value: 0.5,
    ActionType.SUPPORT_BY_FIRE.value: 0.75,
    ActionType.ATTACK.value: 1.0,
    ActionType.WITHDRAW.value: -0.5,
}


@dataclass(frozen=True, slots=True)
class InferenceTimingSummary:
    """Aggregate timing statistics recovered from turn metadata."""

    sample_count: int = 0
    mean_seconds: float = 0.0
    max_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class MockWhiteCellMetricHook:
    """Simple metric hook used when white-cell scoring is mocked."""

    doctrine_compliance: float = 0.0
    tactical_rationality: float = 0.0

    def __call__(self, record: Mapping[str, Any], metric_name: str) -> float | None:
        del record
        if metric_name == "doctrine_compliance":
            return self.doctrine_compliance
        if metric_name == "tactical_rationality":
            return self.tactical_rationality
        return None


def load_jsonl_records(path: str | Path) -> list[LogRecord]:
    """Load structured turn records from a JSONL log file."""

    records: list[LogRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"JSONL record on line {line_number} must be a JSON object."
                )
            records.append(payload)
    return records


def doctrine_compliance_rate(
    results: LogSource,
    *,
    hook: MetricHook | None = None,
) -> float:
    """Aggregate doctrine compliance values emitted into turn logs.

    The default extractor looks for white-cell metadata embedded in each turn
    record. When a project-specific scorer is not available yet, callers can
    pass a lightweight hook that returns mock values per record.
    """

    values = _collect_hook_values(
        _coerce_records(results),
        metric_name="doctrine_compliance",
        hook=hook,
    )
    return _mean(values)


def action_entropy(results: LogSource) -> float:
    """Compute Shannon entropy over logged action types."""

    action_counts = Counter(
        action_type
        for record in _coerce_records(results)
        for action_type in _iter_action_types(record)
    )
    total = sum(action_counts.values())
    if total == 0:
        return 0.0

    return -sum(
        (count / total) * math.log2(count / total)
        for count in action_counts.values()
    )


def escalation_sensitivity_index(results: LogSource) -> float:
    """Measure turn-to-turn volatility in action aggressiveness.

    Each turn receives an escalation score equal to the mean of explicit
    action-type weights. The index is the mean absolute change between
    consecutive turns, so larger values indicate more abrupt doctrinal shifts.
    """

    turn_scores: list[float] = []
    for record in _sorted_records(_coerce_records(results)):
        action_scores = [
            ACTION_ESCALATION_WEIGHTS[action_type]
            for action_type in _iter_action_types(record)
            if action_type in ACTION_ESCALATION_WEIGHTS
        ]
        if action_scores:
            turn_scores.append(_mean(action_scores))

    if len(turn_scores) < 2:
        return 0.0
    return _mean(
        abs(current - previous)
        for previous, current in zip(turn_scores, turn_scores[1:], strict=False)
    )


def escalation_index(results: LogSource) -> float:
    """Backward-compatible alias for the escalation sensitivity index."""

    return escalation_sensitivity_index(results)


def tactical_rationality_score(
    results: LogSource,
    *,
    hook: MetricHook | None = None,
) -> float:
    """Aggregate tactical rationality values emitted into turn logs."""

    values = _collect_hook_values(
        _coerce_records(results),
        metric_name="tactical_rationality",
        hook=hook,
    )
    return _mean(values)


def win_rate(
    runs: Sequence[LogSource],
    *,
    faction: Faction = Faction.BLUE,
) -> float:
    """Compute the fraction of runs won by the requested faction."""

    if not runs:
        return 0.0
    wins = sum(1 for run in runs if _winner_for_run(_coerce_records(run)) == faction)
    return wins / len(runs)


def mean_remaining_force_ratio(
    runs: Sequence[LogSource],
    *,
    faction: Faction = Faction.BLUE,
) -> float:
    """Compute the mean remaining force ratio for one faction across runs."""

    ratios: list[float] = []
    for run in runs:
        records = _coerce_records(run)
        if not records:
            continue
        initial_totals = _initial_force_totals(records)
        initial_value = float(initial_totals.get(faction.value, 0))
        if initial_value <= 0:
            continue
        final_totals = _final_force_totals(records)
        ratios.append(float(final_totals.get(faction.value, 0)) / initial_value)
    return _mean(ratios)


def tactic_transition_frequency(results: LogSource) -> float:
    """Compute how often a unit changes tactic between consecutive turns."""

    actions_by_unit: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for record in _sorted_records(_coerce_records(results)):
        turn = int(record.get("turn", 0))
        for action in _iter_actions(record):
            unit_id = action.get("unit_id")
            action_type = action.get("action_type")
            if isinstance(unit_id, str) and isinstance(action_type, str):
                actions_by_unit[unit_id].append((turn, action_type))

    transitions = 0
    opportunities = 0
    for history in actions_by_unit.values():
        ordered = sorted(history)
        for (_, previous), (_, current) in zip(ordered, ordered[1:], strict=False):
            opportunities += 1
            if previous != current:
                transitions += 1
    if opportunities == 0:
        return 0.0
    return transitions / opportunities


def json_parsing_success_rate(results: LogSource) -> float:
    """Compute the success rate of logged LLM JSON parsing attempts."""

    attempts = 0
    successes = 0
    for record in _coerce_records(results):
        for decision in _iter_decisions(record):
            metadata = decision.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if metadata.get("decision_source") != "local_llm" and "json_parse_success" not in metadata:
                continue
            attempts += 1
            if metadata.get("json_parse_success") is True:
                successes += 1
                continue
            if "json_parse_success" not in metadata and decision.get("used_fallback") is False:
                successes += 1
    if attempts == 0:
        return 0.0
    return successes / attempts


def inference_time_summary(results: LogSource) -> InferenceTimingSummary:
    """Aggregate inference-time metadata across logged agent decisions."""

    samples: list[float] = []
    for record in _coerce_records(results):
        for decision in _iter_decisions(record):
            metadata = decision.get("metadata")
            if not isinstance(metadata, dict):
                continue
            value = metadata.get("inference_time_s")
            if isinstance(value, (int, float)):
                samples.append(float(value))
    if not samples:
        return InferenceTimingSummary()
    return InferenceTimingSummary(
        sample_count=len(samples),
        mean_seconds=_mean(samples),
        max_seconds=max(samples),
    )


def _coerce_records(source: LogSource) -> list[LogRecord]:
    """Normalize either a path or an in-memory record sequence."""

    if isinstance(source, (str, Path)):
        return load_jsonl_records(source)
    return [dict(record) for record in source]


def _sorted_records(records: Sequence[LogRecord]) -> list[LogRecord]:
    """Return records ordered by their turn number."""

    return sorted(records, key=lambda record: int(record.get("turn", 0)))


def _iter_actions(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Yield normalized action dictionaries from one record."""

    raw_actions = record.get("actions")
    if not isinstance(raw_actions, list):
        return []
    return [action for action in raw_actions if isinstance(action, Mapping)]


def _iter_action_types(record: Mapping[str, Any]) -> list[str]:
    """Yield action type strings from one record."""

    action_types: list[str] = []
    for action in _iter_actions(record):
        action_type = action.get("action_type")
        if isinstance(action_type, str):
            action_types.append(action_type)
    return action_types


def _iter_decisions(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Yield blue/red/white-cell decision payloads from one turn record."""

    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return []
    return [
        decision
        for key in ("blue", "red", "white_cell")
        if isinstance((decision := metadata.get(key)), Mapping)
    ]


def _collect_hook_values(
    records: Sequence[LogRecord],
    *,
    metric_name: str,
    hook: MetricHook | None,
) -> list[float]:
    """Collect white-cell or hook-supplied scalar scores from turn records."""

    values: list[float] = []
    for record in records:
        candidate = None
        if hook is not None:
            candidate = hook(record, metric_name)
        if candidate is None:
            candidate = _extract_metric_from_record(record, metric_name)
        if isinstance(candidate, (int, float)):
            values.append(float(candidate))
    return values


def _extract_metric_from_record(record: Mapping[str, Any], metric_name: str) -> float | None:
    """Read a score from common log locations used by white-cell annotations."""

    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return None

    top_level = metadata.get(metric_name)
    if isinstance(top_level, (int, float)):
        return float(top_level)

    white_cell = metadata.get("white_cell")
    if not isinstance(white_cell, Mapping):
        return None

    white_metadata = white_cell.get("metadata")
    if not isinstance(white_metadata, Mapping):
        return None

    direct_value = white_metadata.get(metric_name)
    if isinstance(direct_value, (int, float)):
        return float(direct_value)

    scores = white_metadata.get("scores")
    if isinstance(scores, Mapping):
        nested_value = scores.get(metric_name)
        if isinstance(nested_value, (int, float)):
            return float(nested_value)
    return None


def _winner_for_run(records: Sequence[LogRecord]) -> Faction | None:
    """Determine a run winner from the final logged state or combat result."""

    if not records:
        return None

    final_record = _sorted_records(records)[-1]
    final_totals = _final_force_totals(records)
    blue_total = final_totals.get(Faction.BLUE.value, 0)
    red_total = final_totals.get(Faction.RED.value, 0)
    if blue_total > red_total:
        return Faction.BLUE
    if red_total > blue_total:
        return Faction.RED

    combat = final_record.get("combat")
    if isinstance(combat, Mapping):
        winner = combat.get("winner")
        if isinstance(winner, str):
            try:
                return Faction(winner)
            except ValueError:
                return None
    return None


def _initial_force_totals(records: Sequence[LogRecord]) -> dict[str, int]:
    """Recover initial force totals from logger context or earliest state."""

    if not records:
        return {}

    first_record = _sorted_records(records)[0]
    context = first_record.get("context")
    if isinstance(context, Mapping):
        totals = context.get("initial_force_totals")
        if isinstance(totals, Mapping):
            return {
                str(key): int(value)
                for key, value in totals.items()
                if isinstance(value, (int, float))
            }

    state = first_record.get("state")
    if isinstance(state, Mapping):
        return _force_totals_from_state(state)
    return {}


def _final_force_totals(records: Sequence[LogRecord]) -> dict[str, int]:
    """Recover final force totals from the latest logged state."""

    if not records:
        return {}

    final_record = _sorted_records(records)[-1]
    state = final_record.get("state")
    if isinstance(state, Mapping):
        return _force_totals_from_state(state)
    return {}


def _force_totals_from_state(state: Mapping[str, Any]) -> dict[str, int]:
    """Compute per-faction total strength from a serialized state snapshot."""

    totals = {Faction.BLUE.value: 0, Faction.RED.value: 0}
    units = state.get("units")
    if not isinstance(units, Mapping):
        return totals

    for unit in units.values():
        if not isinstance(unit, Mapping):
            continue
        faction = unit.get("faction")
        strength = unit.get("strength", 0)
        if faction in totals and isinstance(strength, (int, float)):
            totals[faction] += int(strength)
    return totals


def _mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean with a zero default."""

    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)
