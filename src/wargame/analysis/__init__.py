"""Analysis placeholders for metrics, statistics, and plotting."""

from .metrics import (
    InferenceTimingSummary,
    MockWhiteCellMetricHook,
    action_entropy,
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
    "action_entropy",
    "build_experiment_plots",
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
