"""Tests for the local LLM agent wrapper and prompt registry."""

import pytest

from wargame.agents import (
    ActionParser,
    BlueAgent,
    LocalLLMConfig,
    MockLocalLLMBackend,
    ModelOutputError,
    PromptRegistry,
    PromptRole,
    extract_json_object,
)
from wargame.core import ActionType, HexGrid, Posture, Position


def test_prompt_registry_exposes_blue_red_and_white_roles() -> None:
    """The default prompt registry should define all planned tactical roles."""

    registry = PromptRegistry.default()

    assert registry.get(PromptRole.BLUE).role is PromptRole.BLUE
    assert registry.get(PromptRole.RED).role is PromptRole.RED
    assert registry.get(PromptRole.WHITE).role is PromptRole.WHITE
    assert "Respond with a single JSON object only." in registry.get(PromptRole.BLUE).full_system_prompt()


def test_local_llm_agent_uses_fake_backend_and_parses_actions() -> None:
    """The wrapper should be fully testable without loading a real model."""

    backend = MockLocalLLMBackend(
        responses=[
            """
            Model preamble
            {
              "reasoning": "Advance the lead unit.",
              "doctrine_reference": "FM 3-90 maneuver",
              "actions": [
                {
                  "unit_id": "blue-1",
                  "action_type": "move",
                  "target_hex": {"q": 1, "r": 1},
                  "posture": "maneuver"
                }
              ]
            }
            """
        ]
    )
    agent = BlueAgent(
        backend=backend,
        config=LocalLLMConfig(model_name="mock-qwen"),
        parser=ActionParser(grid=HexGrid(width=5, height=5)),
    )

    decision = agent.decide("Enemy spotted near the center.", valid_unit_ids={"blue-1"})

    assert decision.faction.value == "blue"
    assert decision.reasoning == "Advance the lead unit."
    assert decision.doctrine_reference == "FM 3-90 maneuver"
    assert decision.used_fallback is False
    assert len(decision.actions) == 1
    assert decision.actions[0].action_type is ActionType.MOVE
    assert decision.actions[0].posture is Posture.MANEUVER
    assert decision.actions[0].target_hex == Position(1, 1)
    assert "Enemy spotted near the center." in backend.prompts[0]
    assert "BLUE FORCE" in backend.prompts[0].upper()


def test_agent_fallback_exposes_errors_in_metadata() -> None:
    """Fallback behavior should stay explicit instead of silently swallowing errors."""

    backend = MockLocalLLMBackend(responses=["not json at all"])
    agent = BlueAgent(
        backend=backend,
        config=LocalLLMConfig(model_name="mock-qwen"),
        parser=ActionParser(grid=HexGrid(width=5, height=5)),
    )

    decision = agent.decide("Hold the line.", valid_unit_ids={"blue-1", "blue-2"})

    assert decision.used_fallback is True
    assert [action.unit_id for action in decision.actions] == ["blue-1", "blue-2"]
    assert all(action.action_type is ActionType.HOLD for action in decision.actions)
    assert decision.metadata["used_fallback"] is True
    assert decision.metadata["errors"]


def test_agent_can_raise_when_fallback_is_disabled() -> None:
    """Callers should be able to opt into strict failure instead of fallback."""

    backend = MockLocalLLMBackend(responses=["not json at all"])
    agent = BlueAgent(
        backend=backend,
        config=LocalLLMConfig(model_name="mock-qwen"),
        parser=ActionParser(grid=HexGrid(width=5, height=5)),
        fallback_on_error=False,
    )

    with pytest.raises(ModelOutputError, match="No JSON object found"):
        agent.decide("Hold the line.", valid_unit_ids={"blue-1"})


def test_extract_json_object_recovers_balanced_object_from_wrapped_output() -> None:
    """The JSON extraction utility should recover a balanced object when present."""

    payload = 'prefix ```json {"reasoning":"ok","doctrine_reference":"ref","actions":[]} ``` suffix'

    extracted = extract_json_object(payload)

    assert extracted.startswith("{")
    assert extracted.endswith("}")
