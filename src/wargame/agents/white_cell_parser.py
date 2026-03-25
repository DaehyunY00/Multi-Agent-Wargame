"""Dedicated parser for White Cell LLM evaluation output.

The White Cell uses a different output schema than action agents (Blue/Red).
This module provides the parser, result type, and errors for that schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


class WhiteCellParseError(ValueError):
    """Raised when a white-cell model output cannot be parsed into an evaluation."""


@dataclass(frozen=True, slots=True)
class ParsedWhiteCellEvaluation:
    """Normalized result from a successfully parsed white-cell LLM response.

    Fields mirror the output contract defined in ``WHITE_CELL_OUTPUT_CONTRACT``.
    ``tactical_rationality`` is optional because the LLM prompt does not ask for
    it; it is included here for forward-compatibility with richer LLM prompts and
    to share the same result type with the heuristic evaluator.
    """

    tactical_soundness: int | None
    doctrine_compliance: float | None
    tactical_rationality: float | None
    narrative: str
    actions: list[Any]
    used_fallback: bool = False
    errors: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class WhiteCellParser:
    """Parse and validate structured evaluation payloads from a white-cell LLM.

    Unlike ``ActionParser``, this parser accepts white-cell keys
    (``tactical_soundness``, ``doctrine_compliance``, ``narrative``) and
    treats the ``actions`` array as advisory-only (may be empty).
    """

    def parse(self, payload: str) -> ParsedWhiteCellEvaluation:
        """Parse a raw JSON string into a normalized white-cell evaluation.

        Raises ``WhiteCellParseError`` when the payload is malformed or missing
        required fields.
        """
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise WhiteCellParseError(
                f"Malformed JSON in white-cell output: {exc.msg}."
            ) from exc

        if not isinstance(document, dict):
            raise WhiteCellParseError("White-cell payload must be a JSON object.")

        # narrative is the only truly required string field
        narrative = document.get("narrative", "")
        if not isinstance(narrative, str):
            raise WhiteCellParseError("'narrative' must be a string.")

        tactical_soundness = self._parse_tactical_soundness(
            document.get("tactical_soundness")
        )
        doctrine_compliance = self._parse_bounded_float(
            document, "doctrine_compliance", lo=0.0, hi=1.0
        )
        # tactical_rationality is optional and not bounded 0–1 (heuristic uses 1–5)
        tactical_rationality = self._parse_unbounded_float(
            document, "tactical_rationality"
        )
        actions: list[Any] = list(document.get("actions") or [])

        return ParsedWhiteCellEvaluation(
            tactical_soundness=tactical_soundness,
            doctrine_compliance=doctrine_compliance,
            tactical_rationality=tactical_rationality,
            narrative=narrative,
            actions=actions,
        )

    def build_fallback_evaluation(
        self,
        *,
        error: Exception | str,
    ) -> ParsedWhiteCellEvaluation:
        """Build a safe fallback evaluation after a caller catches parse failure."""
        return ParsedWhiteCellEvaluation(
            tactical_soundness=None,
            doctrine_compliance=None,
            tactical_rationality=None,
            narrative="White-cell evaluation failed due to parse error.",
            actions=[],
            used_fallback=True,
            errors=(str(error),),
        )

    @staticmethod
    def _parse_tactical_soundness(raw: Any) -> int | None:
        """Validate and coerce ``tactical_soundness`` to an integer in [1, 5]."""
        if raw is None:
            return None
        if isinstance(raw, float) and raw.is_integer():
            raw = int(raw)
        if not isinstance(raw, int):
            raise WhiteCellParseError(
                f"'tactical_soundness' must be an integer 1–5, got {raw!r}."
            )
        if raw not in range(1, 6):
            raise WhiteCellParseError(
                f"'tactical_soundness' must be between 1 and 5, got {raw!r}."
            )
        return raw

    @staticmethod
    def _parse_bounded_float(
        document: dict[str, Any],
        key: str,
        *,
        lo: float,
        hi: float,
    ) -> float | None:
        """Parse an optional float field and validate it within [lo, hi]."""
        raw = document.get(key)
        if raw is None:
            return None
        if not isinstance(raw, (int, float)):
            raise WhiteCellParseError(
                f"'{key}' must be a number, got {raw!r}."
            )
        value = float(raw)
        if not (lo <= value <= hi):
            raise WhiteCellParseError(
                f"'{key}' must be between {lo} and {hi}, got {value!r}."
            )
        return value

    @staticmethod
    def _parse_unbounded_float(
        document: dict[str, Any],
        key: str,
    ) -> float | None:
        """Parse an optional float field without range constraint."""
        raw = document.get(key)
        if raw is None:
            return None
        if not isinstance(raw, (int, float)):
            raise WhiteCellParseError(
                f"'{key}' must be a number, got {raw!r}."
            )
        return float(raw)
