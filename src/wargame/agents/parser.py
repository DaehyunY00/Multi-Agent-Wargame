"""Strict parsing and validation of model-produced tactical actions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Any

from wargame.core.enums import ActionType, Posture
from wargame.core.hexgrid import HexGrid
from wargame.core.models import ActionCommand, Position


class ActionParseError(ValueError):
    """Raised when an agent payload cannot be parsed into actions."""


STRICT_ACTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["reasoning", "actions", "doctrine_reference"],
    "properties": {
        "reasoning": {"type": "string"},
        "doctrine_reference": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["unit_id", "action_type", "posture"],
                "properties": {
                    "unit_id": {"type": "string"},
                    "action_type": {"type": "string"},
                    "posture": {"type": "string"},
                    "target_hex": {
                        "type": ["object", "null"],
                        "required": ["q", "r"],
                        "properties": {
                            "q": {"type": "integer"},
                            "r": {"type": "integer"},
                        },
                    },
                },
            },
        },
    },
}


@dataclass(frozen=True, slots=True)
class ParsedActionPlan:
    """Normalized action plan returned from a validated model payload."""

    reasoning: str
    doctrine_reference: str
    actions: list[ActionCommand]
    used_fallback: bool = False
    errors: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class ActionParser:
    """Parse and validate structured action payloads from an LLM."""

    grid: HexGrid | None = None

    def parse(
        self,
        payload: str,
        *,
        valid_unit_ids: set[str] | frozenset[str],
    ) -> ParsedActionPlan:
        """Parse a raw JSON payload into a normalized action plan."""

        document = self._load_json(payload)
        self._validate_document_shape(document)

        reasoning = self._require_string(document, "reasoning")
        doctrine_reference = self._require_string(document, "doctrine_reference")
        raw_actions = document["actions"]
        if not isinstance(raw_actions, list):
            raise ActionParseError("'actions' must be a list.")

        actions = [
            self._parse_action(item, valid_unit_ids=valid_unit_ids)
            for item in raw_actions
        ]
        self.validate_actions(actions, valid_unit_ids=valid_unit_ids)
        return ParsedActionPlan(
            reasoning=reasoning,
            doctrine_reference=doctrine_reference,
            actions=actions,
        )

    def parse_actions(
        self,
        payload: str,
        *,
        valid_unit_ids: set[str] | frozenset[str],
    ) -> list[ActionCommand]:
        """Backward-compatible helper that returns only normalized commands."""

        return self.parse(payload, valid_unit_ids=valid_unit_ids).actions

    def validate_actions(
        self,
        actions: list[ActionCommand],
        *,
        valid_unit_ids: set[str] | frozenset[str],
    ) -> None:
        """Validate normalized actions against known engine-facing constraints."""

        allowed_unit_ids = set(valid_unit_ids)
        for action in actions:
            if action.unit_id not in allowed_unit_ids:
                raise ActionParseError(f"Unknown unit_id: {action.unit_id!r}.")
            if action.target_hex is not None and self.grid is not None:
                if not self.grid.is_within_bounds(action.target_hex):
                    raise ActionParseError(
                        f"Target hex {action.target_hex} is out of bounds."
                    )
            if action.action_type is ActionType.HOLD and action.target_hex is not None:
                raise ActionParseError("Hold actions must not specify a target_hex.")
            if action.action_type is not ActionType.HOLD and action.target_hex is None:
                raise ActionParseError(
                    f"Action {action.action_type.value!r} requires a target_hex."
                )

    def build_fallback_plan(
        self,
        *,
        unit_ids: set[str] | frozenset[str],
        error: Exception | str,
    ) -> ParsedActionPlan:
        """Build a safe fallback plan after a caller catches parse failure."""

        error_text = str(error)
        actions = [
            ActionCommand(
                unit_id=unit_id,
                action_type=ActionType.HOLD,
                posture=Posture.DEFEND,
                metadata={"fallback": True, "fallback_reason": error_text},
            )
            for unit_id in sorted(unit_ids)
        ]
        return ParsedActionPlan(
            reasoning="Fallback defensive hold due to invalid model output.",
            doctrine_reference="fallback/default_defense",
            actions=actions,
            used_fallback=True,
            errors=(error_text,),
        )

    @staticmethod
    def _load_json(payload: str) -> dict[str, Any]:
        """Load a JSON object from the raw payload string."""

        try:
            document = json.loads(payload)
        except JSONDecodeError as exc:
            raise ActionParseError(f"Malformed JSON payload: {exc.msg}.") from exc
        if not isinstance(document, dict):
            raise ActionParseError("Top-level payload must be a JSON object.")
        return document

    @staticmethod
    def _validate_document_shape(document: dict[str, Any]) -> None:
        """Check the minimum top-level schema requirements."""

        missing_keys = [
            key for key in ("reasoning", "actions", "doctrine_reference") if key not in document
        ]
        if missing_keys:
            raise ActionParseError(
                f"Missing required keys: {', '.join(sorted(missing_keys))}."
            )

    @staticmethod
    def _require_string(document: dict[str, Any], key: str) -> str:
        """Require that a top-level field exists and is a string."""

        value = document.get(key)
        if not isinstance(value, str):
            raise ActionParseError(f"{key!r} must be a string.")
        return value

    def _parse_action(
        self,
        item: Any,
        *,
        valid_unit_ids: set[str] | frozenset[str],
    ) -> ActionCommand:
        """Parse and normalize one action item."""

        if not isinstance(item, dict):
            raise ActionParseError("Each action entry must be an object.")

        unit_id = item.get("unit_id")
        if not isinstance(unit_id, str):
            raise ActionParseError("Each action must include a string unit_id.")
        if unit_id not in set(valid_unit_ids):
            raise ActionParseError(f"Unknown unit_id: {unit_id!r}.")

        action_type = self._parse_action_type(item.get("action_type"))
        posture = self._parse_posture(item.get("posture"))
        target_hex = self._parse_target_hex(item.get("target_hex"))

        return ActionCommand(
            unit_id=unit_id,
            action_type=action_type,
            target_hex=target_hex,
            posture=posture,
        )

    @staticmethod
    def _parse_action_type(raw_value: Any) -> ActionType:
        """Parse an action type string into the enum."""

        if not isinstance(raw_value, str):
            raise ActionParseError("action_type must be a string.")
        try:
            return ActionType(raw_value)
        except ValueError as exc:
            raise ActionParseError(f"Unknown action_type: {raw_value!r}.") from exc

    @staticmethod
    def _parse_posture(raw_value: Any) -> Posture:
        """Parse a posture string into the enum."""

        if not isinstance(raw_value, str):
            raise ActionParseError("posture must be a string.")
        try:
            return Posture(raw_value)
        except ValueError as exc:
            raise ActionParseError(f"Unknown posture: {raw_value!r}.") from exc

    def _parse_target_hex(self, raw_value: Any) -> Position | None:
        """Parse and validate a target hex object."""

        if raw_value is None:
            return None
        if not isinstance(raw_value, dict):
            raise ActionParseError("target_hex must be an object with q and r.")

        q = raw_value.get("q")
        r = raw_value.get("r")
        if not isinstance(q, int) or not isinstance(r, int):
            raise ActionParseError("target_hex.q and target_hex.r must be integers.")

        position = Position(q=q, r=r)
        if self.grid is not None and not self.grid.is_within_bounds(position):
            raise ActionParseError(f"Target hex {position} is out of bounds.")
        return position
