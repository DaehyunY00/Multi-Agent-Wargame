"""Tests for strict action parsing and safe fallback behavior."""

import pytest

from wargame.agents import ActionParser
from wargame.agents.parser import ActionParseError
from wargame.core import ActionType, HexGrid, Posture, Position


def test_valid_action_payload_is_normalized_into_commands() -> None:
    """A valid LLM payload should become typed engine commands."""

    parser = ActionParser(grid=HexGrid(width=5, height=5))
    payload = """
    {
      "reasoning": "Advance to contact while maintaining support.",
      "doctrine_reference": "FM 3-90 maneuver",
      "actions": [
        {
          "unit_id": "blue-1",
          "action_type": "move",
          "target_hex": {"q": 1, "r": 2},
          "posture": "maneuver"
        }
      ]
    }
    """

    plan = parser.parse(payload, valid_unit_ids={"blue-1"})

    assert plan.reasoning.startswith("Advance to contact")
    assert plan.doctrine_reference == "FM 3-90 maneuver"
    assert len(plan.actions) == 1
    assert plan.actions[0].unit_id == "blue-1"
    assert plan.actions[0].action_type is ActionType.MOVE
    assert plan.actions[0].posture is Posture.MANEUVER
    assert plan.actions[0].target_hex == Position(1, 2)


def test_invalid_hex_is_rejected() -> None:
    """Out-of-bounds target hexes should fail validation."""

    parser = ActionParser(grid=HexGrid(width=3, height=3))
    payload = """
    {
      "reasoning": "Push the lead element forward.",
      "doctrine_reference": "unit-test",
      "actions": [
        {
          "unit_id": "blue-1",
          "action_type": "move",
          "target_hex": {"q": 9, "r": 9},
          "posture": "maneuver"
        }
      ]
    }
    """

    with pytest.raises(ActionParseError, match="out of bounds"):
        parser.parse(payload, valid_unit_ids={"blue-1"})


def test_unknown_unit_is_rejected() -> None:
    """Actions must reference a known unit id."""

    parser = ActionParser(grid=HexGrid(width=5, height=5))
    payload = """
    {
      "reasoning": "Commit a unit that does not exist.",
      "doctrine_reference": "unit-test",
      "actions": [
        {
          "unit_id": "ghost-1",
          "action_type": "move",
          "target_hex": {"q": 1, "r": 1},
          "posture": "maneuver"
        }
      ]
    }
    """

    with pytest.raises(ActionParseError, match="Unknown unit_id"):
        parser.parse(payload, valid_unit_ids={"blue-1"})


def test_malformed_json_can_be_mapped_to_safe_fallback_plan() -> None:
    """Callers should be able to recover from malformed JSON with a fallback."""

    parser = ActionParser(grid=HexGrid(width=5, height=5))
    payload = '{"reasoning": "broken", "actions": ['

    with pytest.raises(ActionParseError) as exc_info:
        parser.parse(payload, valid_unit_ids={"blue-1", "blue-2"})

    fallback = parser.build_fallback_plan(
        unit_ids={"blue-1", "blue-2"},
        error=exc_info.value,
    )

    assert fallback.used_fallback is True
    assert len(fallback.actions) == 2
    assert all(action.action_type is ActionType.HOLD for action in fallback.actions)
    assert all(action.posture is Posture.DEFEND for action in fallback.actions)
