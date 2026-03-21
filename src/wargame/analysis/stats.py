"""Summary statistics helpers for experiment outputs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from wargame.core.enums import Faction

from .metrics import (
    InferenceTimingSummary,
    LogSource,
    action_entropy,
    escalation_sensitivity_index,
    inference_time_summary,
    json_parsing_success_rate,
    mean_remaining_force_ratio,
    tactic_transition_frequency,
    win_rate,
)


@dataclass(frozen=True, slots=True)
class SummaryStatistics:
    """Run-level aggregate statistics computed from stored turn logs."""

    run_count: int = 0
    mean_turns: float = 0.0
    blue_win_rate: float = 0.0
    red_win_rate: float = 0.0
    mean_blue_remaining_force_ratio: float = 0.0
    mean_red_remaining_force_ratio: float = 0.0
    mean_action_entropy: float = 0.0
    mean_escalation_sensitivity_index: float = 0.0
    mean_tactic_transition_frequency: float = 0.0
    mean_json_parsing_success_rate: float = 0.0
    inference_timing: InferenceTimingSummary = InferenceTimingSummary()


def summarize_runs(results: Sequence[LogSource]) -> SummaryStatistics:
    """Summarize a batch of run logs into a compact statistics bundle."""

    if not results:
        return SummaryStatistics()

    mean_turns = _mean(_run_turn_counts(results))
    return SummaryStatistics(
        run_count=len(results),
        mean_turns=mean_turns,
        blue_win_rate=win_rate(results, faction=Faction.BLUE),
        red_win_rate=win_rate(results, faction=Faction.RED),
        mean_blue_remaining_force_ratio=mean_remaining_force_ratio(
            results,
            faction=Faction.BLUE,
        ),
        mean_red_remaining_force_ratio=mean_remaining_force_ratio(
            results,
            faction=Faction.RED,
        ),
        mean_action_entropy=_mean(action_entropy(run) for run in results),
        mean_escalation_sensitivity_index=_mean(
            escalation_sensitivity_index(run) for run in results
        ),
        mean_tactic_transition_frequency=_mean(
            tactic_transition_frequency(run) for run in results
        ),
        mean_json_parsing_success_rate=_mean(
            json_parsing_success_rate(run) for run in results
        ),
        inference_timing=_combine_inference_timing(
            [inference_time_summary(run) for run in results]
        ),
    )


def _run_turn_counts(results: Sequence[LogSource]) -> list[float]:
    """Estimate the number of turns in each run from its log length."""

    counts: list[float] = []
    for run in results:
        if isinstance(run, (str, bytes, Path)):
            from .metrics import load_jsonl_records

            counts.append(float(len(load_jsonl_records(run))))
        else:
            try:
                counts.append(float(len(run)))
            except TypeError:
                counts.append(0.0)
    return counts


def _combine_inference_timing(
    summaries: Sequence[InferenceTimingSummary],
) -> InferenceTimingSummary:
    """Combine per-run inference timing summaries into one batch summary."""

    sample_count = sum(summary.sample_count for summary in summaries)
    if sample_count == 0:
        return InferenceTimingSummary()

    weighted_total = sum(
        summary.mean_seconds * summary.sample_count
        for summary in summaries
    )
    return InferenceTimingSummary(
        sample_count=sample_count,
        mean_seconds=weighted_total / sample_count,
        max_seconds=max(summary.max_seconds for summary in summaries),
    )


def _mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean with a zero default."""

    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)
