"""LLM 의사결정 추적 분석 스크립트.

JSONL 로그에서 LLM Agent가 어떤 근거로 어떻게 결정했는지를
체계적으로 추출·분석하여 연구용 리포트를 생성한다.

6개 분석 모듈:
  1. Decision Trace     — 턴별 상황→추론→행동→결과 체인 추적
  2. Reasoning–Action    — reasoning 텍스트와 실제 행동의 일관성 검증
  3. Doctrine Compliance — 교리 준수 추이 및 faction별 비교
  4. Action Pattern      — 행동 분포, 전환 빈도, 에스컬레이션 패턴
  5. Fallback / Error    — 파싱 실패·폴백 분석
  6. Cross-Run Compare   — 동일 시나리오 다중 시드 간 결정 분산

Usage:
  # 단일 로그 전체 분석
  python scripts/analyze_decisions.py runs/phase2/batch_stability/seed_0.jsonl

  # 디렉토리 내 모든 JSONL 크로스런 비교
  python scripts/analyze_decisions.py runs/phase2/batch_stability/ --cross-run

  # 특정 모듈만 실행
  python scripts/analyze_decisions.py runs/phase2/batch_stability/seed_0.jsonl \
      --modules trace reasoning

  # JSON 결과 저장
  python scripts/analyze_decisions.py runs/phase2/batch_stability/seed_0.jsonl \
      --output-json runs/analysis_result.json

  # 특정 faction만 분석
  python scripts/analyze_decisions.py runs/phase2/batch_stability/seed_0.jsonl \
      --faction blue
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Try importing from wargame; fall back to inline implementations
# when the runtime Python lacks StrEnum (< 3.11).
try:
    from wargame.analysis.metrics import (
        load_jsonl_records,
        ACTION_ESCALATION_WEIGHTS,
    )
except ImportError:
    def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
        """Standalone JSONL loader (fallback for Python < 3.11)."""
        records: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records

    ACTION_ESCALATION_WEIGHTS: dict[str, float] = {
        "hold": 0.0,
        "recon": 0.25,
        "move": 0.5,
        "support_by_fire": 0.75,
        "attack": 1.0,
        "withdraw": -0.5,
    }

# ───────────────────────────────────────────────────────────────────
# Data types
# ───────────────────────────────────────────────────────────────────

LogRecord = dict[str, Any]

FACTIONS = ("blue", "red")

# Keywords mapped to expected action types for reasoning–action coherence
REASONING_ACTION_KEYWORDS: dict[str, list[str]] = {
    "attack": ["attack", "assault", "offensive", "engage", "strike", "destroy"],
    "move": ["advance", "maneuver", "flank", "reposition", "move", "push"],
    "hold": ["hold", "defend", "defensive", "secure", "consolidate", "maintain"],
    "withdraw": ["withdraw", "retreat", "fall back", "disengage", "pull back"],
    "recon": ["recon", "reconn", "observe", "scout", "surveillance", "gather intel"],
    "support_by_fire": ["support", "fire support", "suppress", "indirect fire", "overwatch"],
}


@dataclass
class DecisionTrace:
    """턴 하나의 의사결정 추적 레코드."""

    turn: int = 0
    faction: str = ""
    reasoning: str = ""
    doctrine_reference: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    used_fallback: bool = False
    json_parse_success: bool | None = None
    inference_time_s: float | None = None
    errors: list[str] = field(default_factory=list)
    decision_source: str = ""
    model_name: str = ""
    raw_output_length: int = 0
    wc_doctrine_compliance: float | None = None
    wc_tactical_rationality: float | None = None
    wc_narrative: str = ""
    combat_occurred: bool = False
    casualties: dict[str, int] = field(default_factory=dict)


@dataclass
class ReasoningActionCoherence:
    """reasoning 텍스트와 실제 행동의 일관성 분석."""

    turn: int = 0
    faction: str = ""
    reasoning: str = ""
    implied_actions: list[str] = field(default_factory=list)
    actual_actions: list[str] = field(default_factory=list)
    coherence_score: float = 0.0  # 0.0~1.0
    mismatches: list[str] = field(default_factory=list)


@dataclass
class CrossRunComparison:
    """동일 턴에서의 크로스런 결정 분산."""

    turn: int = 0
    faction: str = ""
    action_distributions: dict[str, int] = field(default_factory=dict)
    unique_reasonings: int = 0
    reasoning_samples: list[str] = field(default_factory=list)
    action_entropy: float = 0.0


# ───────────────────────────────────────────────────────────────────
# Module 1: Decision Trace — 턴별 의사결정 추적
# ───────────────────────────────────────────────────────────────────

def extract_decision_traces(
    records: list[LogRecord],
    faction: str = "blue",
) -> list[DecisionTrace]:
    """JSONL 레코드에서 faction별 의사결정 추적 체인을 추출한다."""

    traces: list[DecisionTrace] = []
    for record in sorted(records, key=lambda r: int(r.get("turn", 0))):
        turn = int(record.get("turn", 0))
        metadata = record.get("metadata", {})
        decision = metadata.get(faction, {})
        if not decision:
            continue

        dm = decision.get("metadata", {})

        # White-cell scores (faction-specific if available)
        wc = metadata.get("white_cell", {})
        wcm = wc.get("metadata", {})
        ta = wcm.get("turn_assessment", {})
        by_faction = ta.get("by_faction", {}).get(faction, {})
        wc_dc = by_faction.get("doctrine_compliance") if by_faction else wcm.get("doctrine_compliance")
        wc_tr = by_faction.get("tactical_rationality") if by_faction else wcm.get("tactical_rationality")
        wc_narrative = ta.get("narrative", "")

        # Combat info
        combat = record.get("combat", {})
        casualties = combat.get("casualties_by_unit", {}) if isinstance(combat, dict) else {}
        combat_occurred = bool(casualties)

        trace = DecisionTrace(
            turn=turn,
            faction=faction,
            reasoning=decision.get("reasoning", ""),
            doctrine_reference=decision.get("doctrine_reference", ""),
            actions=[
                {
                    "unit_id": a.get("unit_id"),
                    "action_type": a.get("action_type"),
                    "posture": a.get("posture"),
                    "target_hex": a.get("target_hex"),
                }
                for a in decision.get("actions", [])
            ],
            used_fallback=decision.get("used_fallback", False),
            json_parse_success=dm.get("json_parse_success"),
            inference_time_s=dm.get("inference_time_s"),
            errors=dm.get("errors", []),
            decision_source=dm.get("decision_source", ""),
            model_name=dm.get("model_name", ""),
            raw_output_length=len(dm.get("raw_output", "")),
            wc_doctrine_compliance=wc_dc,
            wc_tactical_rationality=wc_tr,
            wc_narrative=wc_narrative,
            combat_occurred=combat_occurred,
            casualties={k: v for k, v in casualties.items() if isinstance(v, (int, float))},
        )
        traces.append(trace)
    return traces


def format_decision_trace(trace: DecisionTrace) -> str:
    """사람이 읽을 수 있는 형태로 의사결정 추적을 포맷한다."""

    lines = [
        f"╔══ Turn {trace.turn} [{trace.faction.upper()}] {'(FALLBACK)' if trace.used_fallback else ''}",
        f"║ Source: {trace.decision_source}  Model: {trace.model_name or 'N/A'}",
        f"║ Parse: {'✓' if trace.json_parse_success else '✗'}  "
        f"Inference: {trace.inference_time_s:.2f}s" if trace.inference_time_s else "",
        f"║ Reasoning: {trace.reasoning[:200]}{'...' if len(trace.reasoning) > 200 else ''}",
        f"║ Doctrine: {trace.doctrine_reference}",
    ]

    if trace.errors:
        lines.append(f"║ Errors: {'; '.join(trace.errors)}")

    lines.append("║ Actions:")
    for a in trace.actions:
        target = f"→ ({a['target_hex']['q']},{a['target_hex']['r']})" if a.get("target_hex") else ""
        lines.append(f"║   {a['unit_id']:12s}  {a['action_type']:16s}  {a.get('posture',''):10s}  {target}")

    if trace.combat_occurred:
        cas_str = ", ".join(f"{uid}: -{loss}" for uid, loss in trace.casualties.items())
        lines.append(f"║ Combat: {cas_str}")

    if trace.wc_doctrine_compliance is not None:
        lines.append(
            f"║ WC Eval: doctrine={trace.wc_doctrine_compliance:.2f}  "
            f"rationality={trace.wc_tactical_rationality:.2f}"
        )

    lines.append("╚" + "═" * 60)
    return "\n".join(line for line in lines if line)


# ───────────────────────────────────────────────────────────────────
# Module 2: Reasoning–Action Coherence — 추론-행동 일관성 검증
# ───────────────────────────────────────────────────────────────────

def _infer_actions_from_reasoning(reasoning: str) -> list[str]:
    """reasoning 텍스트에서 암시된 행동 유형을 추출한다."""

    reasoning_lower = reasoning.lower()
    implied: list[str] = []
    for action_type, keywords in REASONING_ACTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in reasoning_lower:
                if action_type not in implied:
                    implied.append(action_type)
                break
    return implied


def analyze_reasoning_action_coherence(
    traces: list[DecisionTrace],
) -> list[ReasoningActionCoherence]:
    """각 턴에서 reasoning 텍스트와 실제 행동의 일관성을 평가한다."""

    results: list[ReasoningActionCoherence] = []
    for trace in traces:
        if trace.used_fallback:
            continue  # fallback은 모델의 자발적 결정이 아니므로 제외

        implied = _infer_actions_from_reasoning(trace.reasoning)
        actual = list({a["action_type"] for a in trace.actions if a.get("action_type")})

        if not implied and not actual:
            continue

        # 일관성 점수: implied와 actual의 교집합 비율
        if implied:
            overlap = set(implied) & set(actual)
            coherence = len(overlap) / len(set(implied))
        else:
            coherence = 0.5  # reasoning에서 키워드를 추출하지 못한 경우 중립

        mismatches = []
        for imp in implied:
            if imp not in actual:
                mismatches.append(f"reasoning implies '{imp}' but not in actions")
        for act in actual:
            if act not in implied and implied:
                mismatches.append(f"action '{act}' not implied by reasoning")

        results.append(ReasoningActionCoherence(
            turn=trace.turn,
            faction=trace.faction,
            reasoning=trace.reasoning[:300],
            implied_actions=implied,
            actual_actions=actual,
            coherence_score=coherence,
            mismatches=mismatches,
        ))
    return results


# ───────────────────────────────────────────────────────────────────
# Module 3: Doctrine Compliance Analysis — 교리 준수 분석
# ───────────────────────────────────────────────────────────────────

@dataclass
class DoctrineAnalysis:
    """교리 준수 분석 결과."""

    faction: str = ""
    turn_count: int = 0
    mean_doctrine_compliance: float = 0.0
    mean_tactical_rationality: float = 0.0
    compliance_trend: list[float] = field(default_factory=list)
    rationality_trend: list[float] = field(default_factory=list)
    doctrine_references_used: dict[str, int] = field(default_factory=dict)
    low_compliance_turns: list[int] = field(default_factory=list)


def analyze_doctrine_compliance(
    traces: list[DecisionTrace],
    low_threshold: float = 0.5,
) -> DoctrineAnalysis:
    """교리 준수 추이를 분석한다."""

    compliance_values: list[float] = []
    rationality_values: list[float] = []
    doc_refs: Counter[str] = Counter()
    low_turns: list[int] = []

    for trace in traces:
        if trace.wc_doctrine_compliance is not None:
            compliance_values.append(trace.wc_doctrine_compliance)
            if trace.wc_doctrine_compliance < low_threshold:
                low_turns.append(trace.turn)
        if trace.wc_tactical_rationality is not None:
            rationality_values.append(trace.wc_tactical_rationality)
        if trace.doctrine_reference:
            doc_refs[trace.doctrine_reference] += 1

    faction = traces[0].faction if traces else ""
    return DoctrineAnalysis(
        faction=faction,
        turn_count=len(traces),
        mean_doctrine_compliance=_safe_mean(compliance_values),
        mean_tactical_rationality=_safe_mean(rationality_values),
        compliance_trend=compliance_values,
        rationality_trend=rationality_values,
        doctrine_references_used=dict(doc_refs.most_common()),
        low_compliance_turns=low_turns,
    )


# ───────────────────────────────────────────────────────────────────
# Module 4: Action Pattern Analysis — 행동 패턴 분석
# ───────────────────────────────────────────────────────────────────

@dataclass
class ActionPatternAnalysis:
    """행동 패턴 분석 결과."""

    faction: str = ""
    action_distribution: dict[str, int] = field(default_factory=dict)
    action_distribution_pct: dict[str, float] = field(default_factory=dict)
    posture_distribution: dict[str, int] = field(default_factory=dict)
    turn_by_turn_escalation: list[float] = field(default_factory=list)
    mean_escalation: float = 0.0
    escalation_volatility: float = 0.0
    tactic_transition_rate: float = 0.0
    unit_action_history: dict[str, list[str]] = field(default_factory=dict)
    action_entropy: float = 0.0


def analyze_action_patterns(traces: list[DecisionTrace]) -> ActionPatternAnalysis:
    """행동 분포, 에스컬레이션, 전환 빈도를 분석한다."""

    action_counts: Counter[str] = Counter()
    posture_counts: Counter[str] = Counter()
    unit_history: dict[str, list[str]] = defaultdict(list)
    turn_escalation: list[float] = []

    for trace in traces:
        turn_scores: list[float] = []
        for a in trace.actions:
            at = a.get("action_type", "")
            action_counts[at] += 1
            posture = a.get("posture", "")
            if posture:
                posture_counts[posture] += 1
            uid = a.get("unit_id", "")
            if uid:
                unit_history[uid].append(at)
            if at in ACTION_ESCALATION_WEIGHTS:
                turn_scores.append(ACTION_ESCALATION_WEIGHTS[at])
        if turn_scores:
            turn_escalation.append(_safe_mean(turn_scores))

    total_actions = sum(action_counts.values())
    action_pct = {k: v / total_actions for k, v in action_counts.items()} if total_actions else {}

    # Escalation volatility
    volatility = 0.0
    if len(turn_escalation) >= 2:
        diffs = [abs(turn_escalation[i] - turn_escalation[i - 1]) for i in range(1, len(turn_escalation))]
        volatility = _safe_mean(diffs)

    # Tactic transition rate
    transitions = 0
    opportunities = 0
    for history in unit_history.values():
        for i in range(1, len(history)):
            opportunities += 1
            if history[i] != history[i - 1]:
                transitions += 1
    transition_rate = transitions / opportunities if opportunities else 0.0

    # Action entropy
    entropy = 0.0
    if total_actions > 0:
        entropy = -sum(
            (c / total_actions) * math.log2(c / total_actions)
            for c in action_counts.values()
        )

    faction = traces[0].faction if traces else ""
    return ActionPatternAnalysis(
        faction=faction,
        action_distribution=dict(action_counts.most_common()),
        action_distribution_pct={k: round(v, 4) for k, v in action_pct.items()},
        posture_distribution=dict(posture_counts.most_common()),
        turn_by_turn_escalation=turn_escalation,
        mean_escalation=_safe_mean(turn_escalation),
        escalation_volatility=volatility,
        tactic_transition_rate=transition_rate,
        unit_action_history={uid: acts for uid, acts in sorted(unit_history.items())},
        action_entropy=entropy,
    )


# ───────────────────────────────────────────────────────────────────
# Module 5: Fallback / Error Analysis — 파싱 실패·폴백 분석
# ───────────────────────────────────────────────────────────────────

@dataclass
class FallbackAnalysis:
    """파싱 실패 및 폴백 분석 결과."""

    faction: str = ""
    total_decisions: int = 0
    fallback_count: int = 0
    fallback_rate: float = 0.0
    parse_success_count: int = 0
    parse_success_rate: float = 0.0
    error_type_distribution: dict[str, int] = field(default_factory=dict)
    fallback_turns: list[int] = field(default_factory=list)
    consecutive_fallback_streaks: list[int] = field(default_factory=list)
    mean_inference_time_success: float = 0.0
    mean_inference_time_failure: float = 0.0


def analyze_fallbacks(traces: list[DecisionTrace]) -> FallbackAnalysis:
    """파싱 실패와 폴백 패턴을 분석한다."""

    total = len(traces)
    fallback_count = sum(1 for t in traces if t.used_fallback)
    parse_success_count = sum(1 for t in traces if t.json_parse_success is True)
    fallback_turns = [t.turn for t in traces if t.used_fallback]

    error_counts: Counter[str] = Counter()
    for t in traces:
        for err in t.errors:
            error_counts[err] += 1

    # Consecutive fallback streaks
    streaks: list[int] = []
    current_streak = 0
    for t in traces:
        if t.used_fallback:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)

    # Inference time by success/failure
    success_times = [t.inference_time_s for t in traces if t.json_parse_success is True and t.inference_time_s]
    failure_times = [t.inference_time_s for t in traces if t.json_parse_success is not True and t.inference_time_s]

    faction = traces[0].faction if traces else ""
    return FallbackAnalysis(
        faction=faction,
        total_decisions=total,
        fallback_count=fallback_count,
        fallback_rate=fallback_count / total if total else 0.0,
        parse_success_count=parse_success_count,
        parse_success_rate=parse_success_count / total if total else 0.0,
        error_type_distribution=dict(error_counts.most_common()),
        fallback_turns=fallback_turns,
        consecutive_fallback_streaks=streaks,
        mean_inference_time_success=_safe_mean(success_times),
        mean_inference_time_failure=_safe_mean(failure_times),
    )


# ───────────────────────────────────────────────────────────────────
# Module 6: Cross-Run Comparison — 크로스런 비교
# ───────────────────────────────────────────────────────────────────

@dataclass
class CrossRunSummary:
    """다중 런 간 의사결정 분산 요약."""

    faction: str = ""
    run_count: int = 0
    per_turn_comparisons: list[CrossRunComparison] = field(default_factory=list)
    mean_action_entropy_across_turns: float = 0.0
    mean_unique_reasonings_per_turn: float = 0.0
    overall_action_distribution: dict[str, int] = field(default_factory=dict)
    fallback_rates_per_run: list[float] = field(default_factory=list)
    parse_success_rates_per_run: list[float] = field(default_factory=list)
    doctrine_compliance_per_run: list[float] = field(default_factory=list)


def compare_across_runs(
    runs: list[list[LogRecord]],
    faction: str = "blue",
) -> CrossRunSummary:
    """동일 시나리오의 다중 런에서 의사결정 분산을 분석한다."""

    if not runs:
        return CrossRunSummary(faction=faction)

    # Collect traces per run
    all_traces: list[list[DecisionTrace]] = []
    fallback_rates: list[float] = []
    parse_rates: list[float] = []
    compliance_rates: list[float] = []

    for run_records in runs:
        traces = extract_decision_traces(run_records, faction)
        all_traces.append(traces)
        fb = analyze_fallbacks(traces)
        fallback_rates.append(fb.fallback_rate)
        parse_rates.append(fb.parse_success_rate)
        dc = analyze_doctrine_compliance(traces)
        compliance_rates.append(dc.mean_doctrine_compliance)

    # Align by turn number
    max_turn = max(
        (t.turn for traces in all_traces for t in traces),
        default=0,
    )

    per_turn: list[CrossRunComparison] = []
    overall_actions: Counter[str] = Counter()

    for turn_num in range(1, max_turn + 1):
        turn_traces = [
            t
            for traces in all_traces
            for t in traces
            if t.turn == turn_num
        ]
        if not turn_traces:
            continue

        action_dist: Counter[str] = Counter()
        reasonings: list[str] = []
        for t in turn_traces:
            for a in t.actions:
                at = a.get("action_type", "")
                action_dist[at] += 1
                overall_actions[at] += 1
            if t.reasoning and not t.used_fallback:
                reasonings.append(t.reasoning[:100])

        # Action entropy for this turn across runs
        total = sum(action_dist.values())
        entropy = 0.0
        if total > 0:
            entropy = -sum(
                (c / total) * math.log2(c / total)
                for c in action_dist.values()
            )

        unique_reasonings = len(set(reasonings))

        per_turn.append(CrossRunComparison(
            turn=turn_num,
            faction=faction,
            action_distributions=dict(action_dist),
            unique_reasonings=unique_reasonings,
            reasoning_samples=reasonings[:5],  # 최대 5개 샘플
            action_entropy=entropy,
        ))

    entropies = [c.action_entropy for c in per_turn]
    unique_rs = [float(c.unique_reasonings) for c in per_turn]

    return CrossRunSummary(
        faction=faction,
        run_count=len(runs),
        per_turn_comparisons=per_turn,
        mean_action_entropy_across_turns=_safe_mean(entropies),
        mean_unique_reasonings_per_turn=_safe_mean(unique_rs),
        overall_action_distribution=dict(overall_actions.most_common()),
        fallback_rates_per_run=fallback_rates,
        parse_success_rates_per_run=parse_rates,
        doctrine_compliance_per_run=compliance_rates,
    )


# ───────────────────────────────────────────────────────────────────
# Report generation
# ───────────────────────────────────────────────────────────────────

def generate_text_report(
    traces: list[DecisionTrace],
    coherence: list[ReasoningActionCoherence],
    doctrine: DoctrineAnalysis,
    patterns: ActionPatternAnalysis,
    fallbacks: FallbackAnalysis,
    cross_run: CrossRunSummary | None = None,
) -> str:
    """사람이 읽을 수 있는 분석 리포트를 생성한다."""

    lines: list[str] = []
    sep = "=" * 70

    # Header
    lines.append(sep)
    lines.append("  LLM Decision Analysis Report")
    lines.append(f"  Faction: {traces[0].faction.upper() if traces else 'N/A'}")
    lines.append(f"  Turns analyzed: {len(traces)}")
    source = traces[0].decision_source if traces else "N/A"
    model = traces[0].model_name if traces else "N/A"
    lines.append(f"  Decision source: {source}  Model: {model}")
    lines.append(sep)

    # ── 1. Decision Trace Summary ──
    lines.append("\n[1] DECISION TRACE SUMMARY")
    lines.append("-" * 40)
    for trace in traces:
        lines.append(format_decision_trace(trace))

    # ── 2. Reasoning–Action Coherence ──
    lines.append(f"\n[2] REASONING–ACTION COHERENCE")
    lines.append("-" * 40)
    if coherence:
        scores = [c.coherence_score for c in coherence]
        lines.append(f"Analyzed turns (non-fallback): {len(coherence)}")
        lines.append(f"Mean coherence score: {_safe_mean(scores):.3f}")
        low_coherence = [c for c in coherence if c.coherence_score < 0.5]
        lines.append(f"Low coherence turns (<0.5): {len(low_coherence)}")
        for c in low_coherence:
            lines.append(f"  Turn {c.turn}: score={c.coherence_score:.2f}")
            lines.append(f"    Implied: {c.implied_actions}")
            lines.append(f"    Actual:  {c.actual_actions}")
            for m in c.mismatches:
                lines.append(f"    ! {m}")
    else:
        lines.append("No non-fallback decisions to analyze.")

    # ── 3. Doctrine Compliance ──
    lines.append(f"\n[3] DOCTRINE COMPLIANCE")
    lines.append("-" * 40)
    lines.append(f"Mean doctrine compliance: {doctrine.mean_doctrine_compliance:.3f}")
    lines.append(f"Mean tactical rationality: {doctrine.mean_tactical_rationality:.3f}")
    if doctrine.doctrine_references_used:
        lines.append("Doctrine references used:")
        for ref, count in doctrine.doctrine_references_used.items():
            lines.append(f"  {ref}: {count} times")
    if doctrine.low_compliance_turns:
        lines.append(f"Low compliance turns (<0.5): {doctrine.low_compliance_turns}")
    if doctrine.compliance_trend:
        lines.append(f"Compliance trend: {' → '.join(f'{v:.2f}' for v in doctrine.compliance_trend)}")

    # ── 4. Action Patterns ──
    lines.append(f"\n[4] ACTION PATTERNS")
    lines.append("-" * 40)
    lines.append(f"Action entropy: {patterns.action_entropy:.3f}")
    lines.append(f"Mean escalation level: {patterns.mean_escalation:.3f}")
    lines.append(f"Escalation volatility: {patterns.escalation_volatility:.3f}")
    lines.append(f"Tactic transition rate: {patterns.tactic_transition_rate:.3f}")
    lines.append("Action distribution:")
    for action, count in patterns.action_distribution.items():
        pct = patterns.action_distribution_pct.get(action, 0)
        lines.append(f"  {action:20s}  {count:4d}  ({pct:.1%})")
    lines.append("Posture distribution:")
    for posture, count in patterns.posture_distribution.items():
        lines.append(f"  {posture:20s}  {count:4d}")
    if patterns.turn_by_turn_escalation:
        lines.append(f"Escalation trend: {' → '.join(f'{v:.2f}' for v in patterns.turn_by_turn_escalation)}")

    # ── 5. Fallback / Error ──
    lines.append(f"\n[5] FALLBACK / ERROR ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"Total decisions: {fallbacks.total_decisions}")
    lines.append(f"Fallback count: {fallbacks.fallback_count} ({fallbacks.fallback_rate:.1%})")
    lines.append(f"Parse success: {fallbacks.parse_success_count} ({fallbacks.parse_success_rate:.1%})")
    if fallbacks.consecutive_fallback_streaks:
        lines.append(f"Consecutive fallback streaks: {fallbacks.consecutive_fallback_streaks}")
        lines.append(f"  Max streak: {max(fallbacks.consecutive_fallback_streaks)}")
    if fallbacks.error_type_distribution:
        lines.append("Error types:")
        for err, count in fallbacks.error_type_distribution.items():
            lines.append(f"  [{count}x] {err}")
    if fallbacks.mean_inference_time_success > 0:
        lines.append(
            f"Inference time — success: {fallbacks.mean_inference_time_success:.2f}s, "
            f"failure: {fallbacks.mean_inference_time_failure:.2f}s"
        )

    # ── 6. Cross-Run Comparison ──
    if cross_run and cross_run.run_count > 1:
        lines.append(f"\n[6] CROSS-RUN COMPARISON ({cross_run.run_count} runs)")
        lines.append("-" * 40)
        lines.append(f"Mean action entropy across turns: {cross_run.mean_action_entropy_across_turns:.3f}")
        lines.append(f"Mean unique reasonings per turn: {cross_run.mean_unique_reasonings_per_turn:.1f}")
        lines.append(f"Overall action distribution: {cross_run.overall_action_distribution}")
        lines.append(f"Fallback rates per run: {[f'{r:.2f}' for r in cross_run.fallback_rates_per_run]}")
        lines.append(f"Parse success rates per run: {[f'{r:.2f}' for r in cross_run.parse_success_rates_per_run]}")
        lines.append(f"Doctrine compliance per run: {[f'{r:.2f}' for r in cross_run.doctrine_compliance_per_run]}")

        # 턴별 행동 분산이 높은 턴 표시
        high_entropy_turns = [
            c for c in cross_run.per_turn_comparisons
            if c.action_entropy > 1.0
        ]
        if high_entropy_turns:
            lines.append(f"\nHigh-variance turns (entropy > 1.0):")
            for c in high_entropy_turns:
                lines.append(
                    f"  Turn {c.turn}: entropy={c.action_entropy:.2f}, "
                    f"unique_reasonings={c.unique_reasonings}, "
                    f"actions={c.action_distributions}"
                )
                for sample in c.reasoning_samples[:2]:
                    lines.append(f"    sample: \"{sample}\"")

    lines.append(f"\n{sep}")
    lines.append("  End of Report")
    lines.append(sep)
    return "\n".join(lines)


def build_json_result(
    traces: list[DecisionTrace],
    coherence: list[ReasoningActionCoherence],
    doctrine: DoctrineAnalysis,
    patterns: ActionPatternAnalysis,
    fallbacks: FallbackAnalysis,
    cross_run: CrossRunSummary | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """분석 결과를 JSON으로 직렬화 가능한 dict로 반환한다."""

    result: dict[str, Any] = {}
    if context:
        result["context"] = context

    result["decision_traces"] = [asdict(t) for t in traces]
    result["reasoning_action_coherence"] = {
        "analyzed_count": len(coherence),
        "mean_coherence_score": _safe_mean([c.coherence_score for c in coherence]),
        "low_coherence_turns": [
            asdict(c) for c in coherence if c.coherence_score < 0.5
        ],
        "all": [asdict(c) for c in coherence],
    }
    result["doctrine_compliance"] = asdict(doctrine)
    result["action_patterns"] = asdict(patterns)
    result["fallback_analysis"] = asdict(fallbacks)
    if cross_run:
        result["cross_run_comparison"] = asdict(cross_run)

    return result


# ───────────────────────────────────────────────────────────────────
# Utilities
# ───────────────────────────────────────────────────────────────────

def _safe_mean(values: list[float] | Sequence[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _collect_jsonl_paths(path: Path) -> list[Path]:
    """파일 또는 디렉토리에서 JSONL 경로를 수집한다."""
    if path.is_file() and path.suffix == ".jsonl":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.jsonl"))
    return []


def _extract_context(records: list[LogRecord]) -> dict[str, Any]:
    """첫 레코드에서 실험 컨텍스트를 추출한다."""
    if records:
        return records[0].get("context", {})
    return {}


# ───────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────

ALL_MODULES = ("trace", "reasoning", "doctrine", "patterns", "fallbacks", "cross-run")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM 의사결정 추적 분석 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="JSONL 로그 파일 또는 디렉토리",
    )
    parser.add_argument(
        "--faction",
        default="blue",
        choices=["blue", "red"],
        help="분석 대상 faction (default: blue)",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=list(ALL_MODULES),
        choices=ALL_MODULES,
        help="실행할 분석 모듈 (default: 전체)",
    )
    parser.add_argument(
        "--cross-run",
        action="store_true",
        help="디렉토리 내 모든 JSONL을 크로스런 비교",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="JSON 결과 저장 경로",
    )
    parser.add_argument(
        "--output-text",
        type=Path,
        default=None,
        help="텍스트 리포트 저장 경로",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="stdout 출력 억제 (파일 저장만 수행)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Collect all JSONL files
    all_paths: list[Path] = []
    for p in args.paths:
        all_paths.extend(_collect_jsonl_paths(Path(p)))

    if not all_paths:
        print("Error: No JSONL files found.", file=sys.stderr)
        return 1

    # Load records
    all_runs: list[list[LogRecord]] = []
    for path in all_paths:
        try:
            records = load_jsonl_records(path)
            if records:
                all_runs.append(records)
        except Exception as exc:
            print(f"Warning: Failed to load {path}: {exc}", file=sys.stderr)

    if not all_runs:
        print("Error: No valid records loaded.", file=sys.stderr)
        return 1

    # Use first run for single-run analysis; all runs for cross-run
    primary_records = all_runs[0]
    context = _extract_context(primary_records)

    # Module 1: Decision Traces
    traces = extract_decision_traces(primary_records, args.faction)

    # Module 2: Reasoning–Action Coherence
    coherence: list[ReasoningActionCoherence] = []
    if "reasoning" in args.modules:
        coherence = analyze_reasoning_action_coherence(traces)

    # Module 3: Doctrine Compliance
    doctrine = DoctrineAnalysis(faction=args.faction)
    if "doctrine" in args.modules:
        doctrine = analyze_doctrine_compliance(traces)

    # Module 4: Action Patterns
    patterns = ActionPatternAnalysis(faction=args.faction)
    if "patterns" in args.modules:
        patterns = analyze_action_patterns(traces)

    # Module 5: Fallbacks
    fallbacks = FallbackAnalysis(faction=args.faction)
    if "fallbacks" in args.modules:
        fallbacks = analyze_fallbacks(traces)

    # Module 6: Cross-Run Comparison
    cross_run: CrossRunSummary | None = None
    if args.cross_run and len(all_runs) > 1 and "cross-run" in args.modules:
        cross_run = compare_across_runs(all_runs, args.faction)

    # Generate reports
    text_report = generate_text_report(traces, coherence, doctrine, patterns, fallbacks, cross_run)
    json_result = build_json_result(traces, coherence, doctrine, patterns, fallbacks, cross_run, context)

    if not args.quiet:
        print(text_report)

    if args.output_text:
        args.output_text.parent.mkdir(parents=True, exist_ok=True)
        args.output_text.write_text(text_report, encoding="utf-8")
        print(f"\nText report saved: {args.output_text}", file=sys.stderr)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(json_result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"JSON result saved: {args.output_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
