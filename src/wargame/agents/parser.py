"""Strict parsing and validation of model-produced tactical actions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Any, ClassVar

from wargame.core.enums import ActionType, Posture

_logger = logging.getLogger(__name__)
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

        parsed = [
            self._parse_action(item, valid_unit_ids=valid_unit_ids)
            for item in raw_actions
        ]
        # Filter out None entries (skipped enemy-unit actions)
        actions = [a for a in parsed if a is not None]
        actions = self.validate_actions(actions, valid_unit_ids=valid_unit_ids)
        actions = self._fill_missing_units(actions, valid_unit_ids=valid_unit_ids)
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
    ) -> list[ActionCommand]:
        """Validate normalized actions against known engine-facing constraints.

        Actions that are non-HOLD but missing a target_hex are silently demoted
        to HOLD/DEFEND rather than rejecting the entire plan.  All other
        constraint violations (unknown unit, out-of-bounds hex) still raise
        ``ActionParseError`` so the caller can build a full fallback plan.
        """

        allowed_unit_ids = set(valid_unit_ids)
        result: list[ActionCommand] = []
        for action in actions:
            if action.unit_id not in allowed_unit_ids:
                raise ActionParseError(f"Unknown unit_id: {action.unit_id!r}.")
            if action.target_hex is not None and self.grid is not None:
                if not self.grid.is_within_bounds(action.target_hex):
                    raise ActionParseError(
                        f"Target hex {action.target_hex} is out of bounds."
                    )
            if action.action_type is ActionType.HOLD and action.target_hex is not None:
                _logger.warning(
                    "HOLD action for unit %r has a spurious target_hex — clearing it.",
                    action.unit_id,
                )
                action.target_hex = None
            if action.action_type is not ActionType.HOLD and action.target_hex is None:
                original = action.action_type.value
                _logger.warning(
                    "Action %r for unit %r has no target_hex — demoting to HOLD/DEFEND.",
                    original,
                    action.unit_id,
                )
                action.action_type = ActionType.HOLD
                action.posture = Posture.DEFEND
                action.metadata = {
                    **action.metadata,
                    "auto_demoted": True,
                    "original_action": original,
                }
            result.append(action)
        return result

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
    ) -> ActionCommand | None:
        """Parse and normalize one action item."""

        if not isinstance(item, dict):
            raise ActionParseError("Each action entry must be an object.")

        unit_id = item.get("unit_id")
        if not isinstance(unit_id, str):
            raise ActionParseError("Each action must include a string unit_id.")
        if unit_id not in set(valid_unit_ids):
            unit_id = self._soft_match_unit_id(unit_id, valid_unit_ids)
            if unit_id is None:
                return None  # signal to caller: skip this action

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
    def _soft_match_unit_id(
        unit_id: str,
        valid_unit_ids: set[str] | frozenset[str],
    ) -> str | None:
        """Attempt soft recovery for a mismatched unit_id.

        Returns the corrected unit_id, or ``None`` to signal the action
        should be silently dropped (enemy unit_id case).
        """
        # Detect opposing-faction IDs: if all valid IDs share a common prefix
        # (e.g. "blue-") and the unit_id starts with a different faction prefix,
        # silently drop this action.
        faction_prefixes = {"blue-", "red-"}
        unit_prefix = None
        for prefix in faction_prefixes:
            if unit_id.startswith(prefix):
                unit_prefix = prefix
                break
        if unit_prefix is not None:
            # Check if ANY valid unit shares the same faction prefix
            if not any(vid.startswith(unit_prefix) for vid in valid_unit_ids):
                _logger.warning(
                    "Dropping action for enemy unit_id %r — not in valid set.",
                    unit_id,
                )
                return None

        # Partial/suffix match: e.g. "assault" matching "blue-assault"
        suffix_matches = [
            vid for vid in valid_unit_ids if vid.endswith(f"-{unit_id}")
        ]
        if len(suffix_matches) == 1:
            _logger.warning(
                "Soft-matched unit_id %r → %r (suffix match).",
                unit_id,
                suffix_matches[0],
            )
            return suffix_matches[0]

        # No recovery possible
        raise ActionParseError(f"Unknown unit_id: {unit_id!r}.")

    @staticmethod
    def _fill_missing_units(
        actions: list[ActionCommand],
        *,
        valid_unit_ids: set[str] | frozenset[str],
    ) -> list[ActionCommand]:
        """Ensure every valid unit_id has exactly one action.

        Units missing from the parsed output receive a HOLD/DEFEND fallback.
        """
        covered = {a.unit_id for a in actions}
        missing = set(valid_unit_ids) - covered
        for uid in sorted(missing):
            _logger.warning(
                "Unit %r missing from model output — inserting HOLD/DEFEND fallback.",
                uid,
            )
            actions.append(
                ActionCommand(
                    unit_id=uid,
                    action_type=ActionType.HOLD,
                    posture=Posture.DEFEND,
                    metadata={"fallback": True, "fallback_reason": "missing from model output"},
                )
            )
        return actions

    # Common LLM confusions for action_type values.
    _ACTION_TYPE_ALIASES: ClassVar[dict[str, str]] = {
        "maneuver": "move",
        "defend": "hold",
        "fire": "support_by_fire",
    }

    @classmethod
    def _parse_action_type(cls, raw_value: Any) -> ActionType:
        """Parse an action type string into the enum.

        Common LLM synonyms are soft-mapped with a warning rather than
        raising an error.
        """

        if not isinstance(raw_value, str):
            raise ActionParseError("action_type must be a string.")
        canonical = cls._ACTION_TYPE_ALIASES.get(raw_value)
        if canonical is not None:
            _logger.warning(
                "action_type %r is not a valid value — mapping to %r.",
                raw_value,
                canonical,
            )
            raw_value = canonical
        try:
            return ActionType(raw_value)
        except ValueError as exc:
            raise ActionParseError(f"Unknown action_type: {raw_value!r}.") from exc

    # Common LLM confusions for posture values.
    _POSTURE_ALIASES: ClassVar[dict[str, str]] = {
        "observe": "maneuver",
        "offense": "attack",
        "recon": "maneuver",           # LLM uses action_type name as posture
        "support_by_fire": "support",  # LLM uses action_type name as posture
    }

    @classmethod
    def _parse_posture(cls, raw_value: Any) -> Posture:
        """Parse a posture string into the enum.

        Common LLM synonyms are soft-mapped with a warning rather than
        raising an error.
        """

        if not isinstance(raw_value, str):
            raise ActionParseError("posture must be a string.")
        canonical = cls._POSTURE_ALIASES.get(raw_value)
        if canonical is not None:
            _logger.warning(
                "posture %r is not a valid value — mapping to %r.",
                raw_value,
                canonical,
            )
            raw_value = canonical
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
