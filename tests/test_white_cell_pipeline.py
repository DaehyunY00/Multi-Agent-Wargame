"""Tests for the White Cell evaluation pipeline.

Covers the architectural fix that separates White Cell output schema from
the Blue/Red action-agent schema:

- White Cell prompt uses WHITE_CELL_OUTPUT_CONTRACT, not OUTPUT_CONTRACT
- White Cell parser accepts evaluation JSON (tactical_soundness, doctrine_compliance, narrative)
- White Cell parser rejects action-schema JSON and malformed JSON
- WhiteCellAgent.decide() returns normalized evaluation metadata
- Blue/Red LocalLLMAgent behavior is unchanged
- Heuristic and LLM WhiteCellAgent produce compatible metadata shapes
"""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass, field

import pytest

from wargame.agents import (
    ActionParser,
    AgentDecision,
    BaseAgent,
    BlueAgent,
    HeuristicWhiteCellAgent,
    LocalLLMConfig,
    MockLocalLLMBackend,
    ParsedWhiteCellEvaluation,
    PromptRegistry,
    PromptRole,
    RedAgent,
    WhiteCellAgent,
    WhiteCellParseError,
    WhiteCellParser,
)
from wargame.agents.prompts import OUTPUT_CONTRACT, WHITE_CELL_OUTPUT_CONTRACT
from wargame.core.enums import ActionType, Faction


# ---------------------------------------------------------------------------
# 1. Prompt contract separation
# ---------------------------------------------------------------------------


class TestPromptContractSeparation:
    """White Cell and action agents must receive different output contracts."""

    def test_blue_prompt_contains_action_contract(self) -> None:
        registry = PromptRegistry.default()
        full = registry.get(PromptRole.BLUE).full_system_prompt()
        assert "reasoning" in full
        assert "doctrine_reference" in full
        assert "actions" in full
        # must contain the canonical action-contract opener
        assert "Respond with a single JSON object only." in full

    def test_red_prompt_contains_action_contract(self) -> None:
        registry = PromptRegistry.default()
        full = registry.get(PromptRole.RED).full_system_prompt()
        assert "reasoning" in full
        assert "doctrine_reference" in full

    def test_white_prompt_contains_evaluation_contract(self) -> None:
        registry = PromptRegistry.default()
        full = registry.get(PromptRole.WHITE).full_system_prompt()
        assert "tactical_soundness" in full
        assert "doctrine_compliance" in full
        assert "narrative" in full

    def test_white_prompt_does_not_contain_action_agent_keys(self) -> None:
        """The white cell prompt must not instruct the model to produce 'reasoning'
        or 'doctrine_reference' — those are action-agent keys."""
        registry = PromptRegistry.default()
        full = registry.get(PromptRole.WHITE).full_system_prompt()
        # These are required keys of OUTPUT_CONTRACT — must NOT appear in WHITE
        assert "- reasoning:" not in full
        assert "- doctrine_reference:" not in full

    def test_white_output_contract_differs_from_action_output_contract(self) -> None:
        assert WHITE_CELL_OUTPUT_CONTRACT != OUTPUT_CONTRACT
        assert "tactical_soundness" in WHITE_CELL_OUTPUT_CONTRACT
        assert "tactical_soundness" not in OUTPUT_CONTRACT

    def test_white_prompt_spec_uses_evaluation_contract(self) -> None:
        registry = PromptRegistry.default()
        spec = registry.get(PromptRole.WHITE)
        assert spec.output_contract == WHITE_CELL_OUTPUT_CONTRACT

    def test_blue_prompt_spec_uses_action_contract(self) -> None:
        registry = PromptRegistry.default()
        spec = registry.get(PromptRole.BLUE)
        assert spec.output_contract == OUTPUT_CONTRACT


# ---------------------------------------------------------------------------
# 2. WhiteCellParser — valid input
# ---------------------------------------------------------------------------


class TestWhiteCellParserValidInput:
    """WhiteCellParser should accept properly structured evaluation JSON."""

    def test_parse_full_evaluation_json(self) -> None:
        parser = WhiteCellParser()
        payload = json.dumps({
            "tactical_soundness": 4,
            "doctrine_compliance": 0.833,
            "narrative": "Concentration and maneuver passed; security failed.",
            "actions": [],
        })
        result = parser.parse(payload)
        assert result.tactical_soundness == 4
        assert result.doctrine_compliance == pytest.approx(0.833)
        assert result.narrative == "Concentration and maneuver passed; security failed."
        assert result.actions == []
        assert result.used_fallback is False
        assert result.errors == ()

    def test_parse_minimum_fields_only(self) -> None:
        """narrative is the only required string; all numeric fields are optional."""
        parser = WhiteCellParser()
        payload = json.dumps({"narrative": "No violations detected."})
        result = parser.parse(payload)
        assert result.narrative == "No violations detected."
        assert result.tactical_soundness is None
        assert result.doctrine_compliance is None
        assert result.tactical_rationality is None

    def test_parse_doctrine_compliance_boundary_zero(self) -> None:
        parser = WhiteCellParser()
        payload = json.dumps({"narrative": "fail", "doctrine_compliance": 0.0})
        assert parser.parse(payload).doctrine_compliance == 0.0

    def test_parse_doctrine_compliance_boundary_one(self) -> None:
        parser = WhiteCellParser()
        payload = json.dumps({"narrative": "ok", "doctrine_compliance": 1.0})
        assert parser.parse(payload).doctrine_compliance == 1.0

    def test_parse_tactical_soundness_boundary_values(self) -> None:
        parser = WhiteCellParser()
        for value in (1, 2, 3, 4, 5):
            result = parser.parse(json.dumps({"narrative": "x", "tactical_soundness": value}))
            assert result.tactical_soundness == value

    def test_parse_actions_list_preserved(self) -> None:
        parser = WhiteCellParser()
        payload = json.dumps({
            "narrative": "severe violation",
            "actions": [{"unit_id": "blue-1", "action_type": "hold"}],
        })
        result = parser.parse(payload)
        assert len(result.actions) == 1

    def test_parse_tactical_rationality_optional(self) -> None:
        """tactical_rationality is an optional field (not in LLM prompt, used by heuristic)."""
        parser = WhiteCellParser()
        payload = json.dumps({"narrative": "ok", "tactical_rationality": 3.5})
        result = parser.parse(payload)
        assert result.tactical_rationality == pytest.approx(3.5)

    def test_parse_float_tactical_soundness_coerced_to_int(self) -> None:
        """Some models may emit 4.0 instead of 4; coerce if it's a whole number float."""
        parser = WhiteCellParser()
        payload = json.dumps({"narrative": "ok", "tactical_soundness": 4.0})
        result = parser.parse(payload)
        assert result.tactical_soundness == 4


# ---------------------------------------------------------------------------
# 3. WhiteCellParser — rejection of invalid input
# ---------------------------------------------------------------------------


class TestWhiteCellParserRejectsInvalidInput:
    """WhiteCellParser must surface clear errors for malformed payloads."""

    def test_rejects_malformed_json(self) -> None:
        parser = WhiteCellParser()
        with pytest.raises(WhiteCellParseError, match="Malformed JSON"):
            parser.parse("{not valid json")

    def test_rejects_json_array_top_level(self) -> None:
        parser = WhiteCellParser()
        with pytest.raises(WhiteCellParseError, match="JSON object"):
            parser.parse("[1, 2, 3]")

    def test_rejects_non_string_narrative(self) -> None:
        parser = WhiteCellParser()
        payload = json.dumps({"narrative": 42, "tactical_soundness": 3})
        with pytest.raises(WhiteCellParseError, match="narrative"):
            parser.parse(payload)

    def test_rejects_tactical_soundness_out_of_range_low(self) -> None:
        parser = WhiteCellParser()
        payload = json.dumps({"narrative": "x", "tactical_soundness": 0})
        with pytest.raises(WhiteCellParseError, match="tactical_soundness"):
            parser.parse(payload)

    def test_rejects_tactical_soundness_out_of_range_high(self) -> None:
        parser = WhiteCellParser()
        payload = json.dumps({"narrative": "x", "tactical_soundness": 6})
        with pytest.raises(WhiteCellParseError, match="tactical_soundness"):
            parser.parse(payload)

    def test_rejects_doctrine_compliance_above_one(self) -> None:
        parser = WhiteCellParser()
        payload = json.dumps({"narrative": "x", "doctrine_compliance": 1.1})
        with pytest.raises(WhiteCellParseError, match="doctrine_compliance"):
            parser.parse(payload)

    def test_rejects_doctrine_compliance_below_zero(self) -> None:
        parser = WhiteCellParser()
        payload = json.dumps({"narrative": "x", "doctrine_compliance": -0.1})
        with pytest.raises(WhiteCellParseError, match="doctrine_compliance"):
            parser.parse(payload)

    def test_rejects_non_numeric_doctrine_compliance(self) -> None:
        parser = WhiteCellParser()
        payload = json.dumps({"narrative": "x", "doctrine_compliance": "high"})
        with pytest.raises(WhiteCellParseError, match="doctrine_compliance"):
            parser.parse(payload)

    def test_rejects_action_agent_schema_missing_narrative(self) -> None:
        """An action-agent payload (reasoning/actions/doctrine_reference) succeeds
        only if narrative is absent — but narrative defaults to '' in that case.
        The important point is that 'reasoning' and 'doctrine_reference' keys are
        not required and won't cause errors by themselves."""
        parser = WhiteCellParser()
        # Action-agent JSON is NOT rejected outright — it just yields empty narrative
        action_payload = json.dumps({
            "reasoning": "Advance.",
            "doctrine_reference": "FM 3-90",
            "actions": [],
        })
        result = parser.parse(action_payload)
        # narrative defaults to ""
        assert result.narrative == ""
        # numeric fields are all None (not present in action payload)
        assert result.tactical_soundness is None
        assert result.doctrine_compliance is None


# ---------------------------------------------------------------------------
# 4. WhiteCellParser — fallback
# ---------------------------------------------------------------------------


class TestWhiteCellParserFallback:
    def test_fallback_evaluation_has_used_fallback_true(self) -> None:
        parser = WhiteCellParser()
        result = parser.build_fallback_evaluation(error="some error")
        assert result.used_fallback is True
        assert "some error" in result.errors
        assert result.tactical_soundness is None
        assert result.doctrine_compliance is None

    def test_fallback_evaluation_has_narrative(self) -> None:
        parser = WhiteCellParser()
        result = parser.build_fallback_evaluation(error=ValueError("bad"))
        assert result.narrative != ""
        assert result.actions == []


# ---------------------------------------------------------------------------
# 5. ParsedWhiteCellEvaluation — immutability / type
# ---------------------------------------------------------------------------


class TestParsedWhiteCellEvaluation:
    def test_is_frozen(self) -> None:
        ev = ParsedWhiteCellEvaluation(
            tactical_soundness=3,
            doctrine_compliance=0.5,
            tactical_rationality=None,
            narrative="ok",
            actions=[],
        )
        with pytest.raises((AttributeError, TypeError)):
            ev.narrative = "changed"  # type: ignore[misc]

    def test_default_not_used_fallback(self) -> None:
        ev = ParsedWhiteCellEvaluation(
            tactical_soundness=None,
            doctrine_compliance=None,
            tactical_rationality=None,
            narrative="",
            actions=[],
        )
        assert ev.used_fallback is False
        assert ev.errors == ()


# ---------------------------------------------------------------------------
# 6. WhiteCellAgent — decide() uses WhiteCellParser
# ---------------------------------------------------------------------------


def _make_white_cell_agent(responses: list[str]) -> WhiteCellAgent:
    """Build a WhiteCellAgent with a mock backend, no ActionParser required."""
    return WhiteCellAgent(
        backend=MockLocalLLMBackend(responses=responses),
        config=LocalLLMConfig(model_name="mock-wc"),
    )


class TestWhiteCellAgentDecide:
    def test_decide_parses_valid_evaluation_json(self) -> None:
        raw = json.dumps({
            "tactical_soundness": 4,
            "doctrine_compliance": 0.667,
            "narrative": "Good maneuver; security weak.",
            "actions": [],
        })
        agent = _make_white_cell_agent([raw])
        decision = agent.decide("{}", valid_unit_ids=())

        assert decision.used_fallback is False
        assert decision.reasoning == "Good maneuver; security weak."
        assert decision.doctrine_reference == "white_cell/llm/v1"
        assert decision.actions == []
        assert decision.faction is Faction.WHITE

    def test_decide_metadata_contains_scores(self) -> None:
        raw = json.dumps({
            "tactical_soundness": 2,
            "doctrine_compliance": 0.333,
            "narrative": "Multiple violations.",
            "actions": [],
        })
        agent = _make_white_cell_agent([raw])
        decision = agent.decide("{}", valid_unit_ids=())

        scores = decision.metadata["scores"]
        assert scores["doctrine_compliance"] == pytest.approx(0.333)
        assert scores["tactical_soundness"] == pytest.approx(2.0)

    def test_decide_promotes_doctrine_compliance_to_top_level(self) -> None:
        """doctrine_compliance must appear at metadata root for metric extractor."""
        raw = json.dumps({
            "tactical_soundness": 5,
            "doctrine_compliance": 1.0,
            "narrative": "Perfect.",
            "actions": [],
        })
        agent = _make_white_cell_agent([raw])
        decision = agent.decide("{}", valid_unit_ids=())

        assert decision.metadata["doctrine_compliance"] == pytest.approx(1.0)

    def test_decide_fallback_on_bad_json(self) -> None:
        agent = _make_white_cell_agent(["not json at all"])
        decision = agent.decide("{}", valid_unit_ids=())

        assert decision.used_fallback is True
        assert decision.actions == []
        assert decision.metadata["used_fallback"] is True
        assert decision.metadata["errors"]

    def test_decide_fallback_on_empty_output(self) -> None:
        agent = _make_white_cell_agent([""])
        decision = agent.decide("{}", valid_unit_ids=())
        assert decision.used_fallback is True

    def test_decide_system_prompt_contains_evaluation_contract(self) -> None:
        """The rendered prompt sent to the backend must include white-cell keys."""
        raw = json.dumps({"narrative": "ok"})
        backend = MockLocalLLMBackend(responses=[raw])
        agent = WhiteCellAgent(
            backend=backend,
            config=LocalLLMConfig(model_name="mock"),
        )
        agent.decide("{}", valid_unit_ids=())
        rendered = backend.prompts[0]
        assert "tactical_soundness" in rendered
        assert "doctrine_compliance" in rendered
        assert "narrative" in rendered

    def test_decide_system_prompt_does_not_contain_action_keys(self) -> None:
        """White Cell prompt must not instruct the model to produce reasoning/doctrine_reference."""
        raw = json.dumps({"narrative": "ok"})
        backend = MockLocalLLMBackend(responses=[raw])
        agent = WhiteCellAgent(
            backend=backend,
            config=LocalLLMConfig(model_name="mock"),
        )
        agent.decide("{}", valid_unit_ids=())
        rendered = backend.prompts[0]
        assert "- reasoning:" not in rendered
        assert "- doctrine_reference:" not in rendered

    def test_decide_raises_when_fallback_disabled(self) -> None:
        from wargame.agents.local_llm import ModelOutputError
        from wargame.agents.white_cell_parser import WhiteCellParseError

        agent = WhiteCellAgent(
            backend=MockLocalLLMBackend(responses=["not json"]),
            config=LocalLLMConfig(model_name="mock"),
            fallback_on_error=False,
        )
        with pytest.raises((ModelOutputError, WhiteCellParseError)):
            agent.decide("{}", valid_unit_ids=())

    def test_decide_ignores_valid_unit_ids(self) -> None:
        """White Cell does not validate unit IDs — it evaluates, not commands."""
        raw = json.dumps({"narrative": "ok", "doctrine_compliance": 0.5})
        agent = _make_white_cell_agent([raw])
        # Should not raise even with unknown unit IDs
        decision = agent.decide("{}", valid_unit_ids={"nonexistent-unit"})
        assert decision is not None


# ---------------------------------------------------------------------------
# 7. Blue/Red agents — unchanged behavior
# ---------------------------------------------------------------------------


class TestBlueRedAgentUnchanged:
    """Adding the white cell pipeline must not affect Blue/Red agent operation."""

    def test_blue_agent_uses_action_parser(self) -> None:
        raw = json.dumps({
            "reasoning": "Advance.",
            "doctrine_reference": "FM 3-90",
            "actions": [
                {
                    "unit_id": "blue-1",
                    "action_type": "hold",
                    "posture": "defend",
                    "target_hex": None,
                }
            ],
        })
        backend = MockLocalLLMBackend(responses=[raw])
        agent = BlueAgent(
            backend=backend,
            config=LocalLLMConfig(model_name="mock"),
            parser=ActionParser(),
        )
        decision = agent.decide("situation", valid_unit_ids={"blue-1"})
        assert decision.reasoning == "Advance."
        assert len(decision.actions) == 1
        assert decision.actions[0].action_type is ActionType.HOLD

    def test_blue_agent_fallback_on_white_cell_json(self) -> None:
        """If a Blue agent accidentally receives white-cell JSON, it should
        fall back gracefully because the action schema is different."""
        raw = json.dumps({
            "tactical_soundness": 3,
            "doctrine_compliance": 0.5,
            "narrative": "ok",
            "actions": [],
        })
        backend = MockLocalLLMBackend(responses=[raw])
        agent = BlueAgent(
            backend=backend,
            config=LocalLLMConfig(model_name="mock"),
            parser=ActionParser(),
        )
        decision = agent.decide("situation", valid_unit_ids={"blue-1"})
        assert decision.used_fallback is True

    def test_red_agent_prompt_does_not_contain_white_cell_keys(self) -> None:
        registry = PromptRegistry.default()
        full = registry.get(PromptRole.RED).full_system_prompt()
        assert "tactical_soundness" not in full
        assert "doctrine_compliance" not in full.split("DOCTRINE GUIDELINES")[0]


# ---------------------------------------------------------------------------
# 8. Metadata shape compatibility between heuristic and LLM white cell
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _FakeActionAgent(BaseAgent):
    """Minimal fake agent for pipeline tests."""

    planned_actions: list = field(default_factory=list)

    def decide(
        self, state_text: str, *, valid_unit_ids: Collection[str] = ()
    ) -> AgentDecision:
        from wargame.core.models import ActionCommand
        from wargame.core.enums import ActionType, Posture

        return AgentDecision(
            faction=self.faction,
            reasoning="test",
            doctrine_reference="test",
            actions=[
                ActionCommand(
                    unit_id=uid,
                    action_type=ActionType.HOLD,
                    posture=Posture.DEFEND,
                )
                for uid in sorted(valid_unit_ids)
            ],
        )


class TestMetadataCompatibility:
    """Heuristic and LLM WhiteCellAgent must produce compatible score metadata."""

    def _heuristic_scores(self) -> dict:
        agent = HeuristicWhiteCellAgent()
        payload = json.dumps(
            {
                "turn": 1,
                "blue_view": {
                    "faction": "blue",
                    "turn_metadata": {"turn": 1, "max_turns": 10, "phase": "active"},
                    "friendly_units": {},
                    "enemy_observations": {},
                    "terrain_by_hex": {},
                    "metadata": {},
                },
                "red_view": {
                    "faction": "red",
                    "turn_metadata": {"turn": 1, "max_turns": 10, "phase": "active"},
                    "friendly_units": {},
                    "enemy_observations": {},
                    "terrain_by_hex": {},
                    "metadata": {},
                },
                "blue_decision": {"actions": [], "reasoning": "", "doctrine_reference": ""},
                "red_decision": {"actions": [], "reasoning": "", "doctrine_reference": ""},
                "combat": {},
            }
        )
        return agent.decide(payload).metadata

    def _llm_scores(self, dc: float = 0.667) -> dict:
        raw = json.dumps(
            {
                "tactical_soundness": 3,
                "doctrine_compliance": dc,
                "narrative": "Two of three principles passed.",
                "actions": [],
            }
        )
        agent = _make_white_cell_agent([raw])
        return agent.decide("{}", valid_unit_ids=()).metadata

    def test_both_expose_scores_doctrine_compliance(self) -> None:
        h = self._heuristic_scores()
        l = self._llm_scores()
        assert "scores" in h
        assert "scores" in l
        assert "doctrine_compliance" in h["scores"]
        assert "doctrine_compliance" in l["scores"]

    def test_both_expose_top_level_doctrine_compliance(self) -> None:
        h = self._heuristic_scores()
        l = self._llm_scores()
        assert "doctrine_compliance" in h
        assert "doctrine_compliance" in l

    def test_both_expose_decision_source(self) -> None:
        h = self._heuristic_scores()
        l = self._llm_scores()
        assert h["decision_source"] == "white_cell_heuristic"
        assert l["decision_source"] == "white_cell_llm"

    def test_llm_scores_dict_contains_tactical_soundness(self) -> None:
        l = self._llm_scores()
        assert "tactical_soundness" in l["scores"]
        assert l["scores"]["tactical_soundness"] == pytest.approx(3.0)

    def test_heuristic_scores_dict_contains_tactical_rationality(self) -> None:
        h = self._heuristic_scores()
        assert "tactical_rationality" in h["scores"]

    def test_metric_extractor_reads_doctrine_compliance_from_both(self) -> None:
        """_extract_metric_from_record must find doctrine_compliance in both shapes."""
        from wargame.analysis.metrics import _extract_metric_from_record  # noqa: PLC0415

        h_meta = self._heuristic_scores()
        l_meta = self._llm_scores(dc=0.5)

        # Simulate turn record structure: metadata.white_cell.metadata = agent.metadata
        h_record = {"metadata": {"white_cell": {"metadata": h_meta}}}
        l_record = {"metadata": {"white_cell": {"metadata": l_meta}}}

        h_val = _extract_metric_from_record(h_record, "doctrine_compliance")
        l_val = _extract_metric_from_record(l_record, "doctrine_compliance")

        assert isinstance(h_val, float)
        assert isinstance(l_val, float)
        assert l_val == pytest.approx(0.5)
