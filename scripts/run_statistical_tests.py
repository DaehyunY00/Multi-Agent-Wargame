"""CLI script for statistical testing of wargame experiment results.

Implements Research Plan section 4.4:
  RQ1: One-sample t-test (DCR > 0.5) + Cohen's d
  RQ2: One-way ANOVA (TRS, ESI across LLM models) + Bonferroni pairwise post-hoc
  RQ3: Kruskal-Wallis (Action Entropy across all groups) + Dunn post-hoc
  α = 0.05, multiple-comparison correction: Bonferroni

Usage:
  python scripts/run_statistical_tests.py \\
    --llm-dirs runs/phase4/qwen/ runs/phase4/mistral/ runs/phase4/llama/ \\
    --baseline-dirs runs/phase4/baseline/rule/ runs/phase4/baseline/random/ \\
    --output runs/phase5/statistical_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import scipy.stats as _scipy_stats
    from scipy.stats import rankdata as _rankdata
except ImportError as exc:
    raise ImportError(
        "scipy is required for statistical tests. "
        "Install with: pip install scipy"
    ) from exc

from wargame.analysis.metrics import (  # noqa: E402
    action_entropy,
    doctrine_compliance_rate,
    escalation_sensitivity_index,
    load_jsonl_records,
    tactical_rationality_score,
)

_ALPHA = 0.05


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run all statistical tests and write a JSON results file."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm-dirs",
        nargs="+",
        type=Path,
        required=True,
        metavar="DIR",
        help="Per-model LLM run directories. Directory name is used as model label.",
    )
    parser.add_argument(
        "--baseline-dirs",
        nargs="+",
        type=Path,
        default=[],
        metavar="DIR",
        help="Baseline run directories (rule, random, script, …).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/statistical_results.json"),
    )
    parser.add_argument("--alpha", type=float, default=_ALPHA)
    args = parser.parse_args(list(argv) if argv is not None else None)

    llm_groups = [collect_dir_metrics(d) for d in args.llm_dirs]
    baseline_groups = [collect_dir_metrics(d) for d in args.baseline_dirs]

    results: dict = {
        "alpha": args.alpha,
        "rq1": run_rq1(llm_groups, alpha=args.alpha),
        "rq2": run_rq2(llm_groups, alpha=args.alpha),
        "rq3": run_rq3(llm_groups + baseline_groups, alpha=args.alpha),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------


def collect_dir_metrics(dir_path: Path) -> dict:
    """Collect per-run metric vectors from all JSONL files under *dir_path*.

    Returns a dict with keys: label, dcr, trs, esi, entropy.
    Each value (except label) is a list of floats — one entry per run.
    """
    label = dir_path.name
    dcr: list[float] = []
    trs: list[float] = []
    esi: list[float] = []
    entropy: list[float] = []

    for path in sorted(dir_path.glob("**/*.jsonl")):
        try:
            records = load_jsonl_records(path)
        except Exception:  # noqa: BLE001
            continue
        if not records:
            continue
        dcr.append(doctrine_compliance_rate(records))
        trs.append(tactical_rationality_score(records))
        esi.append(escalation_sensitivity_index(records))
        entropy.append(action_entropy(records))

    return {"label": label, "dcr": dcr, "trs": trs, "esi": esi, "entropy": entropy}


# ---------------------------------------------------------------------------
# RQ1 — One-sample t-test: DCR > 0.5 (better than chance)
# ---------------------------------------------------------------------------


def run_rq1(llm_groups: list[dict], *, alpha: float = _ALPHA) -> list[dict]:
    """One-sample t-test per LLM model: H0: μ_DCR = 0.5."""
    return [one_sample_ttest_result(g["dcr"], popmean=0.5, label=g["label"], alpha=alpha) for g in llm_groups]


def one_sample_ttest_result(
    values: list[float],
    *,
    popmean: float,
    label: str = "",
    alpha: float = _ALPHA,
) -> dict:
    """One-sample t-test (alternative='greater') with Cohen's d.

    Returns a result dict that is safe to serialize as JSON.
    """
    n = len(values)
    base: dict = {"model": label, "n_runs": n}

    if n < 2:
        return {**base, "error": "insufficient_samples", "significant": False}

    mean = _safe_mean(values)
    std = _safe_std(values)
    t_stat, p_value = _scipy_stats.ttest_1samp(values, popmean=popmean, alternative="greater")
    cohens_d = (mean - popmean) / std if std > 0 else 0.0

    return {
        **base,
        "dcr_mean": _r4(mean),
        "dcr_std": _r4(std),
        "t_statistic": _r4(float(t_stat)),
        "p_value": _r4(float(p_value)),
        "cohens_d": _r4(cohens_d),
        "significant": bool(p_value < alpha),
    }


# ---------------------------------------------------------------------------
# RQ2 — One-way ANOVA: TRS and ESI across LLM models
# ---------------------------------------------------------------------------


def run_rq2(llm_groups: list[dict], *, alpha: float = _ALPHA) -> dict:
    """One-way ANOVA on TRS and ESI across LLM models with Bonferroni post-hoc."""
    return {
        "trs": anova_result([g["trs"] for g in llm_groups], [g["label"] for g in llm_groups], alpha=alpha),
        "esi": anova_result([g["esi"] for g in llm_groups], [g["label"] for g in llm_groups], alpha=alpha),
    }


def anova_result(
    groups: list[list[float]],
    group_names: list[str],
    *,
    alpha: float = _ALPHA,
) -> dict:
    """One-way ANOVA with Bonferroni-corrected pairwise t-test post-hoc.

    Returns a JSON-safe result dict.
    """
    non_empty = [(g, n) for g, n in zip(groups, group_names) if len(g) >= 2]
    if len(non_empty) < 2:
        return {
            "error": "insufficient_groups",
            "groups": group_names,
            "significant": False,
        }

    valid_groups, valid_names = zip(*non_empty)
    f_stat, p_value = _scipy_stats.f_oneway(*valid_groups)

    return {
        "groups": list(valid_names),
        "group_means": [_r4(_safe_mean(g)) for g in valid_groups],
        "group_ns": [len(g) for g in valid_groups],
        "f_statistic": _r4(float(f_stat)),
        "p_value": _r4(float(p_value)),
        "significant": bool(p_value < alpha),
        "post_hoc_method": "bonferroni_pairwise_t",
        "post_hoc": bonferroni_pairwise_t(list(valid_groups), list(valid_names), alpha=alpha),
    }


def bonferroni_pairwise_t(
    groups: list[list[float]],
    group_names: list[str],
    *,
    alpha: float = _ALPHA,
) -> dict:
    """Pairwise independent t-tests with Bonferroni correction."""
    k = len(groups)
    n_comparisons = max(1, k * (k - 1) // 2)
    pairs: dict = {}
    for i in range(k):
        for j in range(i + 1, k):
            if len(groups[i]) < 2 or len(groups[j]) < 2:
                continue
            t_stat, p_raw = _scipy_stats.ttest_ind(groups[i], groups[j], equal_var=False)
            p_bonf = min(1.0, float(p_raw) * n_comparisons)
            key = f"{group_names[i]}_vs_{group_names[j]}"
            pairs[key] = {
                "t_statistic": _r4(float(t_stat)),
                "p_value": _r4(float(p_raw)),
                "p_value_bonferroni": _r4(p_bonf),
                "significant": bool(p_bonf < alpha),
            }
    return pairs


# ---------------------------------------------------------------------------
# RQ3 — Kruskal-Wallis: Action Entropy across all groups
# ---------------------------------------------------------------------------


def run_rq3(all_groups: list[dict], *, alpha: float = _ALPHA) -> dict:
    """Kruskal-Wallis test on Action Entropy with Dunn post-hoc."""
    return kruskal_result(
        [g["entropy"] for g in all_groups],
        [g["label"] for g in all_groups],
        alpha=alpha,
    )


def kruskal_result(
    groups: list[list[float]],
    group_names: list[str],
    *,
    alpha: float = _ALPHA,
) -> dict:
    """Kruskal-Wallis H-test with Dunn post-hoc (Bonferroni corrected).

    Returns a JSON-safe result dict.
    """
    non_empty = [(g, n) for g, n in zip(groups, group_names) if len(g) >= 1]
    if len(non_empty) < 2:
        return {
            "error": "insufficient_groups",
            "groups": group_names,
            "significant": False,
        }

    valid_groups, valid_names = zip(*non_empty)
    try:
        h_stat, p_value = _scipy_stats.kruskal(*valid_groups)
    except ValueError:
        # All values identical across groups — H is undefined; test is trivially not significant.
        return {
            "groups": list(valid_names),
            "group_means": [_r4(_safe_mean(g)) for g in valid_groups],
            "group_ns": [len(g) for g in valid_groups],
            "h_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "note": "all_values_identical",
            "post_hoc_method": "dunn_bonferroni",
            "post_hoc": {},
        }

    return {
        "groups": list(valid_names),
        "group_means": [_r4(_safe_mean(g)) for g in valid_groups],
        "group_ns": [len(g) for g in valid_groups],
        "h_statistic": _r4(float(h_stat)),
        "p_value": _r4(float(p_value)),
        "significant": bool(p_value < alpha),
        "post_hoc_method": "dunn_bonferroni",
        "post_hoc": dunn_posthoc(list(valid_groups), list(valid_names), alpha=alpha),
    }


def dunn_posthoc(
    groups: list[list[float]],
    group_names: list[str],
    *,
    alpha: float = _ALPHA,
) -> dict:
    """Dunn's post-hoc test with Bonferroni correction for multiple comparisons.

    Uses the normal approximation of the rank-sum statistic.
    """
    all_vals = [v for g in groups for v in g]
    N = len(all_vals)
    if N == 0:
        return {}

    ranks = _rankdata(all_vals)

    # Split global ranks back into per-group arrays
    group_ranks: list = []
    idx = 0
    for g in groups:
        n = len(g)
        group_ranks.append(ranks[idx : idx + n])
        idx += n

    k = len(groups)
    n_comparisons = max(1, k * (k - 1) // 2)
    pairs: dict = {}

    for i in range(k):
        for j in range(i + 1, k):
            ni, nj = len(groups[i]), len(groups[j])
            if ni == 0 or nj == 0:
                continue
            mean_rank_i = float(group_ranks[i].mean())
            mean_rank_j = float(group_ranks[j].mean())
            se = math.sqrt(N * (N + 1) / 12.0 * (1.0 / ni + 1.0 / nj))
            if se == 0:
                continue
            z = (mean_rank_i - mean_rank_j) / se
            p_raw = 2.0 * float(_scipy_stats.norm.sf(abs(z)))
            p_bonf = min(1.0, p_raw * n_comparisons)
            key = f"{group_names[i]}_vs_{group_names[j]}"
            pairs[key] = {
                "z_statistic": _r4(z),
                "p_value": _r4(p_raw),
                "p_value_bonferroni": _r4(p_bonf),
                "significant": bool(p_bonf < alpha),
            }

    return pairs


# ---------------------------------------------------------------------------
# Pure-Python helpers (no scipy dependency)
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = _safe_mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _r4(value: float) -> float:
    """Round to 4 decimal places; preserve NaN/Inf for serialisation."""
    return round(value, 4) if math.isfinite(value) else value


if __name__ == "__main__":
    raise SystemExit(main())
