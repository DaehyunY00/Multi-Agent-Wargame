"""Analysis placeholders for metrics, statistics, and plotting."""

from .metrics import (
    aggregate_casualties_by_unit,
    InferenceTimingSummary,
    MockWhiteCellMetricHook,
    action_entropy,
    combat_turn_count,
    doctrine_compliance_rate,
    escalation_index,
    escalation_sensitivity_index,
    inference_time_summary,
    json_parsing_success_rate,
    load_jsonl_records,
    mean_remaining_force_ratio,
    tactical_rationality_score,
    tactic_transition_frequency,
    win_rate,
)
from .plots import PlotBundle, build_experiment_plots
from .stats import SummaryStatistics, summarize_runs

__all__ = [
    "InferenceTimingSummary",
    "MockWhiteCellMetricHook",
    "PlotBundle",
    "SummaryStatistics",
    "aggregate_casualties_by_unit",
    "action_entropy",
    "build_experiment_plots",
    "combat_turn_count",
    "doctrine_compliance_rate",
    "escalation_index",
    "escalation_sensitivity_index",
    "inference_time_summary",
    "json_parsing_success_rate",
    "load_jsonl_records",
    "mean_remaining_force_ratio",
    "summarize_runs",
    "tactical_rationality_score",
    "tactic_transition_frequency",
    "win_rate",
]
