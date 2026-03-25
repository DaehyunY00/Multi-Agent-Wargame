"""White-cell evaluation agents for turn-by-turn tactical assessment."""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from wargame.core.enums import ActionType, Faction

from .base import AgentDecision, BaseAgent
from .local_llm import BackendResponse, LocalLLMAgent, ModelOutputError, extract_json_object
from .parser import ActionParser
from .prompts import PromptRole
from .white_cell_parser import (
    ParsedWhiteCellEvaluation,
    WhiteCellParseError,
    WhiteCellParser,
)

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WhiteCellAgent(LocalLLMAgent):
    """White-role LLM agent that uses a dedicated evaluation parser.

    Unlike Blue/Red agents, WhiteCellAgent does not use ``ActionParser``; it
    uses ``WhiteCellParser`` to parse evaluation JSON with keys such as
    ``tactical_soundness``, ``doctrine_compliance``, and ``narrative``.  The
    system prompt is also served with ``WHITE_CELL_OUTPUT_CONTRACT`` rather
    than the action-agent ``OUTPUT_CONTRACT``.
    """

    name: str = "white_cell"
    faction: Faction = Faction.WHITE
    role: PromptRole = PromptRole.WHITE
    # parser is inherited but unused; provide a default so callers need not pass one
    parser: ActionParser = field(default_factory=ActionParser)
    wc_parser: WhiteCellParser = field(default_factory=WhiteCellParser)

    def decide(
        self,
        state_text: str,
        *,
        valid_unit_ids: Collection[str] = (),
    ) -> AgentDecision:
        """Evaluate a turn summary JSON using the dedicated white-cell parser.

        ``valid_unit_ids`` is intentionally ignored: the White Cell adjudicates
        rather than commands, so unit-ID validation is not applicable.
        """
        del valid_unit_ids

        from .local_llm import ChatMessage  # noqa: PLC0415 — avoids circular import

        prompt_spec = self.prompt_registry.get(self.role)
        messages = (
            ChatMessage(role="system", content=prompt_spec.full_system_prompt()),
            ChatMessage(role="user", content=state_text),
        )
        rendered_prompt = self.chat_template_adapter.render(messages)
        response = self.backend.generate(rendered_prompt, self.config)

        try:
            json_payload = extract_json_object(response.content)
            evaluation = self.wc_parser.parse(json_payload)
        except (ModelOutputError, WhiteCellParseError) as exc:
            _logger.debug(
                "White-cell parse failed (%s: %s) — raw output: %r",
                type(exc).__name__,
                exc,
                response.content,
            )
            if not self.fallback_on_error:
                raise
            evaluation = self.wc_parser.build_fallback_evaluation(error=exc)

        return self._build_wc_decision(evaluation=evaluation, response=response)

    def _build_wc_decision(
        self,
        *,
        evaluation: ParsedWhiteCellEvaluation,
        response: BackendResponse,
    ) -> AgentDecision:
        """Convert a parsed white-cell evaluation into the stable AgentDecision format.

        Metadata is normalised to be compatible with the heuristic white cell:
        - ``scores`` dict mirrors ``HeuristicWhiteCellAgent``'s structure
        - Top-level ``doctrine_compliance`` / ``tactical_rationality`` shortcuts
          are promoted so ``_extract_metric_from_record`` can find them at any
          of the paths it checks.
        """
        scores: dict[str, float] = {}
        if evaluation.tactical_soundness is not None:
            scores["tactical_soundness"] = float(evaluation.tactical_soundness)
        if evaluation.doctrine_compliance is not None:
            scores["doctrine_compliance"] = evaluation.doctrine_compliance
        if evaluation.tactical_rationality is not None:
            scores["tactical_rationality"] = evaluation.tactical_rationality

        metadata: dict[str, Any] = {
            "decision_source": "white_cell_llm",
            "scores": scores,
            "used_fallback": evaluation.used_fallback,
            "errors": list(evaluation.errors),
            "raw_output": response.content,
        }
        # Promote top-level shortcuts for metric-extractor compatibility
        # (matches the structure produced by HeuristicWhiteCellAgent)
        if evaluation.doctrine_compliance is not None:
            metadata["doctrine_compliance"] = evaluation.doctrine_compliance
        if evaluation.tactical_rationality is not None:
            metadata["tactical_rationality"] = evaluation.tactical_rationality
        metadata.update(response.metadata)

        return AgentDecision(
            faction=self.faction,
            reasoning=evaluation.narrative,
            doctrine_reference="white_cell/llm/v1",
            actions=[],  # White Cell evaluates; it does not command units
            used_fallback=evaluation.used_fallback,
            metadata=metadata,
        )


@dataclass(slots=True)
class HeuristicWhiteCellAgent(BaseAgent):
    """Deterministic white-cell evaluator used for repeatable experiments."""

    name: str = "white_cell_heuristic"
    faction: Faction = Faction.WHITE

    def decide(
        self,
        state_text: str,
        *,
        valid_unit_ids: Collection[str] = (),
    ) -> AgentDecision:
        """Score a structured turn summary encoded as JSON."""

        del valid_unit_ids
        payload = _load_turn_summary(state_text)
        evaluation = evaluate_turn_summary(payload)
        scores = evaluation["scores"]
        return AgentDecision(
            faction=self.faction,
            reasoning=str(evaluation["narrative"]),
            doctrine_reference="white_cell/heuristic/v1",
            metadata={
                "scores": scores,
                "doctrine_compliance": scores["doctrine_compliance"],
                "tactical_rationality": scores["tactical_rationality"],
                "turn_assessment": evaluation,
                "decision_source": "white_cell_heuristic",
            },
        )


def evaluate_turn_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one structured turn summary with transparent heuristics."""

    faction_scores: dict[str, dict[str, float]] = {}
    narratives: list[str] = []

    for faction_key in ("blue", "red"):
        view = _mapping(summary.get(f"{faction_key}_view"))
        decision = _mapping(summary.get(f"{faction_key}_decision"))
        combat = _mapping(summary.get("combat"))
        action_evaluations = _evaluate_actions_for_faction(
            faction=faction_key,
            view=view,
            decision=decision,
            combat=combat,
        )
        doctrine_score = _mean(item["doctrine"] for item in action_evaluations) if action_evaluations else 0.0
        rationality_score = _mean(item["rationality"] for item in action_evaluations) if action_evaluations else 0.0
        faction_scores[faction_key] = {
            "doctrine_compliance": round(doctrine_score, 4),
            "tactical_rationality": round(rationality_score, 4),
        }
        narratives.append(
            f"{faction_key} doctrine={doctrine_score:.2f}, rationality={rationality_score:.2f}"
        )

    overall_doctrine = _mean(score["doctrine_compliance"] for score in faction_scores.values())
    overall_rationality = _mean(score["tactical_rationality"] for score in faction_scores.values())
    return {
        "scores": {
            "doctrine_compliance": round(overall_doctrine, 4),
            "tactical_rationality": round(overall_rationality, 4),
        },
        "by_faction": faction_scores,
        "narrative": "; ".join(narratives),
    }


def _evaluate_actions_for_faction(
    *,
    faction: str,
    view: Mapping[str, Any],
    decision: Mapping[str, Any],
    combat: Mapping[str, Any],
) -> list[dict[str, float]]:
    """Evaluate per-action doctrine and rationality against visible context."""

    enemy_positions = [
        _position_from_mapping(observation.get("last_known_position"))
        for observation in _mapping(view.get("enemy_observations")).values()
        if isinstance(observation, Mapping)
    ]
    enemy_positions = [position for position in enemy_positions if position is not None]
    casualties = {
        str(unit_id): int(loss)
        for unit_id, loss in _mapping(combat.get("casualties_by_unit")).items()
        if isinstance(loss, (int, float))
    }

    evaluated: list[dict[str, float]] = []
    for action in decision.get("actions", []):
        if not isinstance(action, Mapping):
            continue
        action_type = str(action.get("action_type", ""))
        posture = str(action.get("posture", ""))
        target = _position_from_mapping(action.get("target_hex"))
        unit_id = str(action.get("unit_id", ""))
        proximity = min((_hex_distance(target, position) for position in enemy_positions if target is not None), default=99)
        combat_loss = casualties.get(unit_id, 0)

        doctrine = 0.5
        rationality = 3.0
        if action_type == ActionType.ATTACK.value:
            doctrine = 1.0 if proximity <= 1 else 0.2
            rationality = 4.5 if proximity <= 1 else 1.5
        elif action_type == ActionType.SUPPORT_BY_FIRE.value:
            doctrine = 1.0 if proximity <= 1 else 0.4
            rationality = 4.0 if posture == "support" else 3.0
        elif action_type in {ActionType.MOVE.value, ActionType.RECON.value}:
            doctrine = 0.8
            rationality = 3.5 if target is not None else 2.0
        elif action_type == ActionType.WITHDRAW.value:
            doctrine = 0.9 if combat_loss > 0 or proximity <= 2 else 0.6
            rationality = 4.0 if combat_loss > 0 else 3.0
        elif action_type == ActionType.HOLD.value:
            doctrine = 0.9 if posture == "defend" else 0.6
            rationality = 4.0 if not enemy_positions or proximity > 1 else 3.0

        evaluated.append(
            {
                "doctrine": doctrine,
                "rationality": rationality,
            }
        )

    if not evaluated:
        evaluated.append({"doctrine": 0.0, "rationality": 1.0})
    return evaluated


def _load_turn_summary(payload: str) -> Mapping[str, Any]:
    """Parse the white-cell turn summary JSON."""

    data = json.loads(payload)
    if not isinstance(data, Mapping):
        raise ValueError("White-cell summary must be a JSON object.")
    return data


def _mapping(value: Any) -> Mapping[str, Any]:
    """Normalize a nested mapping-like object."""

    return value if isinstance(value, Mapping) else {}


def _position_from_mapping(value: Any) -> tuple[int, int] | None:
    """Parse a serialized position payload."""

    if not isinstance(value, Mapping):
        return None
    q = value.get("q")
    r = value.get("r")
    if not isinstance(q, int) or not isinstance(r, int):
        return None
    return q, r


def _hex_distance(start: tuple[int, int] | None, end: tuple[int, int] | None) -> int:
    """Compute axial hex distance between two serialized positions."""

    if start is None or end is None:
        return 99
    dq = end[0] - start[0]
    dr = end[1] - start[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _mean(values) -> float:
    """Return a zero-safe arithmetic mean."""

    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)
