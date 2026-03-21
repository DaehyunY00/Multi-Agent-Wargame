"""Unit tests for scripts/run_statistical_tests.py.

All tests use synthetic data so that the suite runs without any experiment
logs on disk.  scipy is a hard dependency of the script; tests are skipped
when it is absent (to stay consistent with the optional-dependency policy).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

# Make the scripts/ directory importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

scipy = pytest.importorskip("scipy", reason="scipy is required for statistical tests")

from run_statistical_tests import (  # noqa: E402
    anova_result,
    bonferroni_pairwise_t,
    collect_dir_metrics,
    dunn_posthoc,
    kruskal_result,
    one_sample_ttest_result,
    run_rq1,
    run_rq2,
    run_rq3,
    _safe_mean,
    _safe_std,
)


# ---------------------------------------------------------------------------
# Helpers for creating synthetic JSONL fixtures
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def _turn(
    turn: int = 1,
    action_type: str = "move",
    doctrine_compliance: float = 0.8,
    tactical_rationality: float = 0.7,
) -> dict:
    """Minimal turn record understood by wargame.analysis.metrics."""
    return {
        "turn": turn,
        "actions": [
            {"unit_id": "b1", "action_type": action_type, "posture": "maneuver"}
        ],
        "metadata": {
            "doctrine_compliance": doctrine_compliance,
            "tactical_rationality": tactical_rationality,
        },
    }


# ---------------------------------------------------------------------------
# _safe_mean / _safe_std
# ---------------------------------------------------------------------------


def test_safe_mean_empty_returns_nan() -> None:
    assert math.isnan(_safe_mean([]))


def test_safe_mean_single_value() -> None:
    assert _safe_mean([3.0]) == pytest.approx(3.0)


def test_safe_std_empty_returns_nan() -> None:
    assert math.isnan(_safe_std([]))


def test_safe_std_single_returns_nan() -> None:
    assert math.isnan(_safe_std([1.0]))


def test_safe_std_constant_series_returns_zero() -> None:
    assert _safe_std([5.0, 5.0, 5.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# one_sample_ttest_result (RQ1)
# ---------------------------------------------------------------------------


def test_rq1_significant_when_dcr_clearly_above_half() -> None:
    values = [0.8, 0.9, 0.75, 0.85, 0.78, 0.92, 0.83, 0.88, 0.76, 0.90]
    result = one_sample_ttest_result(values, popmean=0.5, label="qwen")

    assert result["significant"] is True
    assert result["dcr_mean"] == pytest.approx(0.837, abs=0.01)
    assert result["t_statistic"] > 0
    assert result["p_value"] < 0.05
    assert result["cohens_d"] > 0


def test_rq1_not_significant_when_dcr_near_half() -> None:
    values = [0.48, 0.52, 0.50, 0.51, 0.49, 0.50, 0.52, 0.48, 0.51, 0.49]
    result = one_sample_ttest_result(values, popmean=0.5, label="mistral")

    assert result["significant"] is False


def test_rq1_cohens_d_is_zero_for_identical_mean_and_population() -> None:
    values = [0.5] * 10
    result = one_sample_ttest_result(values, popmean=0.5, label="llama")
    # std is 0 → Cohen's d defined as 0
    assert result["cohens_d"] == 0.0


def test_rq1_insufficient_samples_returns_error() -> None:
    result = one_sample_ttest_result([0.7], popmean=0.5, label="qwen")
    assert result["error"] == "insufficient_samples"
    assert result["significant"] is False


def test_rq1_empty_data_returns_error() -> None:
    result = one_sample_ttest_result([], popmean=0.5, label="empty")
    assert result["error"] == "insufficient_samples"
    assert result["n_runs"] == 0


def test_rq1_result_has_required_keys() -> None:
    values = [0.6, 0.7, 0.65, 0.72, 0.68]
    result = one_sample_ttest_result(values, popmean=0.5, label="qwen")
    for key in ("model", "n_runs", "dcr_mean", "dcr_std", "t_statistic", "p_value", "cohens_d", "significant"):
        assert key in result, f"missing key: {key}"


# ---------------------------------------------------------------------------
# anova_result (RQ2)
# ---------------------------------------------------------------------------


def test_rq2_anova_detects_difference_between_groups() -> None:
    group_a = [0.9, 0.85, 0.88, 0.92, 0.87]
    group_b = [0.4, 0.38, 0.42, 0.39, 0.41]
    group_c = [0.65, 0.60, 0.63, 0.67, 0.62]
    result = anova_result([group_a, group_b, group_c], ["qwen", "mistral", "llama"])

    assert result["significant"] is True
    assert result["f_statistic"] > 1.0
    assert "post_hoc" in result


def test_rq2_anova_not_significant_for_identical_groups() -> None:
    group = [0.5, 0.5, 0.5, 0.5, 0.5]
    result = anova_result([group, group, group], ["a", "b", "c"])
    assert result["significant"] is False


def test_rq2_anova_insufficient_groups_returns_error() -> None:
    result = anova_result([[0.5, 0.6]], ["only"])
    assert result["error"] == "insufficient_groups"
    assert result["significant"] is False


def test_rq2_anova_result_has_required_keys() -> None:
    groups = [[0.7, 0.8], [0.4, 0.5], [0.6, 0.65]]
    result = anova_result(groups, ["a", "b", "c"])
    for key in ("groups", "group_means", "group_ns", "f_statistic", "p_value", "significant", "post_hoc"):
        assert key in result


# ---------------------------------------------------------------------------
# bonferroni_pairwise_t
# ---------------------------------------------------------------------------


def test_bonferroni_pairwise_keys_match_group_pairs() -> None:
    groups = [[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]]
    names = ["a", "b", "c"]
    pairs = bonferroni_pairwise_t(groups, names)

    assert set(pairs.keys()) == {"a_vs_b", "a_vs_c", "b_vs_c"}


def test_bonferroni_pairwise_p_bonferroni_gte_p_value() -> None:
    groups = [[0.1, 0.2, 0.15], [0.9, 0.85, 0.92], [0.5, 0.48, 0.53]]
    names = ["a", "b", "c"]
    pairs = bonferroni_pairwise_t(groups, names)

    for pair in pairs.values():
        assert pair["p_value_bonferroni"] >= pair["p_value"]


def test_bonferroni_pairwise_skips_insufficient_groups() -> None:
    # Group "b" has only one sample — cannot compute t-test
    groups = [[0.5, 0.6, 0.7], [0.9], [0.3, 0.4, 0.35]]
    names = ["a", "b", "c"]
    pairs = bonferroni_pairwise_t(groups, names)

    # Pairs involving "b" should be absent
    assert "a_vs_b" not in pairs
    assert "b_vs_c" not in pairs
    assert "a_vs_c" in pairs


# ---------------------------------------------------------------------------
# kruskal_result (RQ3)
# ---------------------------------------------------------------------------


def test_rq3_kruskal_detects_entropy_difference() -> None:
    low_entropy = [0.1, 0.12, 0.09, 0.11, 0.10]   # e.g. rule-based
    mid_entropy = [1.0, 1.05, 0.98, 1.02, 1.01]   # e.g. LLM
    high_entropy = [2.5, 2.48, 2.52, 2.49, 2.51]  # e.g. random
    result = kruskal_result([low_entropy, mid_entropy, high_entropy], ["rule", "llm", "random"])

    assert result["significant"] is True
    assert result["h_statistic"] > 0
    assert "post_hoc" in result


def test_rq3_kruskal_not_significant_for_identical_groups() -> None:
    group = [1.0, 1.0, 1.0, 1.0]
    result = kruskal_result([group, list(group), list(group)], ["a", "b", "c"])
    assert result["significant"] is False


def test_rq3_kruskal_insufficient_groups_returns_error() -> None:
    result = kruskal_result([[1.0, 2.0]], ["only"])
    assert result["error"] == "insufficient_groups"
    assert result["significant"] is False


def test_rq3_kruskal_result_has_required_keys() -> None:
    groups = [[0.5, 0.6], [1.5, 1.6], [2.5, 2.6]]
    result = kruskal_result(groups, ["a", "b", "c"])
    for key in ("groups", "group_means", "group_ns", "h_statistic", "p_value", "significant", "post_hoc"):
        assert key in result


# ---------------------------------------------------------------------------
# dunn_posthoc
# ---------------------------------------------------------------------------


def test_dunn_posthoc_returns_all_pairs() -> None:
    groups = [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]
    names = ["a", "b", "c"]
    pairs = dunn_posthoc(groups, names)
    assert set(pairs.keys()) == {"a_vs_b", "a_vs_c", "b_vs_c"}


def test_dunn_posthoc_p_bonferroni_gte_p_raw() -> None:
    groups = [[0.1, 0.2, 0.15], [1.5, 1.6, 1.55], [3.0, 3.1, 3.05]]
    names = ["a", "b", "c"]
    for pair in dunn_posthoc(groups, names).values():
        assert pair["p_value_bonferroni"] >= pair["p_value"]


def test_dunn_posthoc_empty_data_returns_empty_dict() -> None:
    assert dunn_posthoc([], []) == {}


def test_dunn_posthoc_skips_empty_group() -> None:
    groups = [[1.0, 1.1], [], [3.0, 3.1]]
    names = ["a", "b", "c"]
    pairs = dunn_posthoc(groups, names)
    # Pairs involving the empty group must be absent
    assert "a_vs_b" not in pairs
    assert "b_vs_c" not in pairs
    assert "a_vs_c" in pairs


# ---------------------------------------------------------------------------
# collect_dir_metrics
# ---------------------------------------------------------------------------


def test_collect_dir_metrics_reads_jsonl_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "qwen"
    run_dir.mkdir()

    # Two runs of 5 turns each
    for seed in range(2):
        _write_jsonl(
            run_dir / f"seed_{seed}.jsonl",
            [_turn(t, action_type="move", doctrine_compliance=0.8, tactical_rationality=0.7) for t in range(1, 6)],
        )

    metrics = collect_dir_metrics(run_dir)

    assert metrics["label"] == "qwen"
    assert len(metrics["dcr"]) == 2
    assert len(metrics["trs"]) == 2
    assert len(metrics["esi"]) == 2
    assert len(metrics["entropy"]) == 2
    # Both runs have doctrine_compliance=0.8 in every turn
    assert metrics["dcr"][0] == pytest.approx(0.8)


def test_collect_dir_metrics_empty_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    metrics = collect_dir_metrics(run_dir)
    assert metrics["label"] == "empty"
    assert metrics["dcr"] == []
    assert metrics["entropy"] == []


def test_collect_dir_metrics_skips_empty_jsonl(tmp_path: Path) -> None:
    run_dir = tmp_path / "model"
    run_dir.mkdir()
    (run_dir / "empty.jsonl").write_text("", encoding="utf-8")
    (run_dir / "real.jsonl").write_text(json.dumps(_turn(1)), encoding="utf-8")

    metrics = collect_dir_metrics(run_dir)
    assert len(metrics["dcr"]) == 1


def test_collect_dir_metrics_recurses_into_subdirs(tmp_path: Path) -> None:
    run_dir = tmp_path / "model"
    sub = run_dir / "scenario_a"
    sub.mkdir(parents=True)
    _write_jsonl(sub / "seed_0.jsonl", [_turn(1)])
    _write_jsonl(sub / "seed_1.jsonl", [_turn(1)])

    metrics = collect_dir_metrics(run_dir)
    assert len(metrics["dcr"]) == 2


# ---------------------------------------------------------------------------
# run_rq1 / run_rq2 / run_rq3 integration with synthetic group dicts
# ---------------------------------------------------------------------------


def _fake_group(label: str, dcr: list[float], trs: list[float], esi: list[float], entropy: list[float]) -> dict:
    return {"label": label, "dcr": dcr, "trs": trs, "esi": esi, "entropy": entropy}


def test_run_rq1_returns_one_entry_per_llm_group() -> None:
    groups = [
        _fake_group("qwen", [0.8] * 5, [], [], []),
        _fake_group("mistral", [0.6] * 5, [], [], []),
    ]
    results = run_rq1(groups)
    assert len(results) == 2
    assert results[0]["model"] == "qwen"
    assert results[1]["model"] == "mistral"


def test_run_rq2_returns_trs_and_esi_keys() -> None:
    groups = [
        _fake_group("qwen", [], [0.9, 0.85, 0.88], [0.2, 0.22, 0.19], []),
        _fake_group("mistral", [], [0.4, 0.38, 0.42], [0.5, 0.52, 0.48], []),
        _fake_group("llama", [], [0.65, 0.63, 0.67], [0.35, 0.33, 0.37], []),
    ]
    result = run_rq2(groups)
    assert "trs" in result
    assert "esi" in result


def test_run_rq3_combines_llm_and_baseline_groups() -> None:
    llm = [_fake_group("qwen", [], [], [], [1.0, 1.1, 1.05])]
    baseline = [_fake_group("rule", [], [], [], [0.1, 0.12, 0.09])]
    result = run_rq3(llm + baseline)
    assert "rule" in result["groups"]
    assert "qwen" in result["groups"]
