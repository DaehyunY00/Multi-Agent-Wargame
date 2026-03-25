"""Local LLM backend abstractions and a testable agent wrapper."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Any

from wargame.core.enums import ActionType, Faction, Posture
from wargame.core.models import ActionCommand

from .base import AgentDecision, BaseAgent
from .parser import ActionParseError, ActionParser, ParsedActionPlan
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptRegistry, PromptRole


@dataclass(frozen=True, slots=True)
class LocalLLMConfig:
    """Configuration shared across local LLM backends."""

    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1024
    context_window: int = 2048


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Generic chat message used before model-specific template rendering."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class BackendResponse:
    """Backend generation result returned by a local inference adapter."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ChatTemplateAdapter(ABC):
    """Render generic chat messages into a backend-ready prompt string."""

    @abstractmethod
    def render(self, messages: Sequence[ChatMessage]) -> str:
        """Render generic messages using a model-family-specific chat template."""


@dataclass(frozen=True, slots=True)
class DefaultChatTemplateAdapter(ChatTemplateAdapter):
    """Minimal deterministic template adapter used for tests and simple backends."""

    def render(self, messages: Sequence[ChatMessage]) -> str:
        sections = [f"{message.role.upper()}:\n{message.content}" for message in messages]
        return "\n\n".join(sections)


class LocalLLMBackend(ABC):
    """Abstract adapter interface for local text generation backends."""

    @abstractmethod
    def generate(self, prompt: str, config: LocalLLMConfig) -> BackendResponse:
        """Generate model output from a rendered prompt string."""


_logger = logging.getLogger(__name__)


class ModelOutputError(ValueError):
    """Raised when model output cannot be converted into a JSON action payload."""


@dataclass(slots=True)
class MockLocalLLMBackend(LocalLLMBackend):
    """Deterministic fake backend for unit tests."""

    responses: deque[str] = field(default_factory=deque)
    prompts: list[str] = field(default_factory=list)

    def __init__(self, responses: Sequence[str] | None = None) -> None:
        self.responses = deque(responses or [])
        self.prompts = []

    def generate(self, prompt: str, config: LocalLLMConfig) -> BackendResponse:
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("MockLocalLLMBackend has no queued responses.")
        return BackendResponse(content=self.responses.popleft())


@dataclass(slots=True, kw_only=True)
class LocalLLMAgent(BaseAgent):
    """Agent wrapper that combines prompts, backend generation, and parsing."""

    backend: LocalLLMBackend
    config: LocalLLMConfig
    parser: ActionParser
    role: PromptRole
    prompt_registry: PromptRegistry = field(default_factory=PromptRegistry.default)
    chat_template_adapter: ChatTemplateAdapter = field(
        default_factory=DefaultChatTemplateAdapter
    )
    fallback_on_error: bool = True

    def decide(
        self,
        state_text: str,
        *,
        valid_unit_ids: Collection[str] = (),
    ) -> AgentDecision:
        prompt_spec = self.prompt_registry.get(self.role)
        messages = (
            ChatMessage(role="system", content=prompt_spec.full_system_prompt()),
            ChatMessage(role="user", content=state_text),
        )
        rendered_prompt = self.chat_template_adapter.render(messages)
        response = self.backend.generate(rendered_prompt, self.config)

        try:
            json_payload = extract_json_object(response.content)
            plan = self.parser.parse(json_payload, valid_unit_ids=set(valid_unit_ids))
        except (ModelOutputError, ActionParseError) as exc:
            _logger.debug(
                "Parsing failed (%s: %s) — raw output: %r",
                type(exc).__name__,
                exc,
                response.content,
            )
            if not self.fallback_on_error:
                raise
            plan = self.parser.build_fallback_plan(
                unit_ids=set(valid_unit_ids),
                error=exc,
            )
            return self._build_decision(plan=plan, response=response, rendered_prompt=rendered_prompt)

        # Ensure every friendly unit has exactly one action.
        plan = self._ensure_all_units_covered(plan, valid_unit_ids=set(valid_unit_ids))

        return self._build_decision(plan=plan, response=response, rendered_prompt=rendered_prompt)

    @staticmethod
    def _ensure_all_units_covered(
        plan: ParsedActionPlan,
        *,
        valid_unit_ids: set[str],
    ) -> ParsedActionPlan:
        """Append HOLD/DEFEND fallback for any friendly unit missing from the plan."""
        covered = {a.unit_id for a in plan.actions}
        missing = valid_unit_ids - covered
        if not missing:
            return plan
        extra = [
            ActionCommand(
                unit_id=uid,
                action_type=ActionType.HOLD,
                posture=Posture.DEFEND,
                metadata={"fallback": True, "fallback_reason": "missing from model output"},
            )
            for uid in sorted(missing)
        ]
        for uid in sorted(missing):
            _logger.warning(
                "Unit %r missing from parsed plan — adding HOLD/DEFEND fallback.",
                uid,
            )
        return ParsedActionPlan(
            reasoning=plan.reasoning,
            doctrine_reference=plan.doctrine_reference,
            actions=list(plan.actions) + extra,
            used_fallback=plan.used_fallback,
            errors=plan.errors,
        )

    def _build_decision(
        self,
        *,
        plan: ParsedActionPlan,
        response: BackendResponse,
        rendered_prompt: str,
    ) -> AgentDecision:
        """Convert a parsed action plan into the stable agent decision format."""

        metadata = {
            "decision_source": "local_llm",
            "role": self.role.value,
            "model_name": self.config.model_name,
            "used_fallback": plan.used_fallback,
            "json_parse_success": not plan.used_fallback,
            "errors": list(plan.errors),
            "raw_output": response.content,
            "rendered_prompt": rendered_prompt,
        }
        metadata.update(response.metadata)
        return AgentDecision(
            faction=self.faction,
            reasoning=plan.reasoning,
            doctrine_reference=plan.doctrine_reference,
            actions=plan.actions,
            used_fallback=plan.used_fallback,
            metadata=metadata,
        )


def extract_json_object(raw_output: str) -> str:
    """Extract the best-matching JSON plan object from model output.

    Scans for all top-level balanced JSON objects, then returns the first one
    that contains all three expected plan keys (``reasoning``, ``actions``,
    ``doctrine_reference``).  If none contains all three, returns the first
    valid JSON object found (original behaviour), so the downstream validator
    can produce a precise error message about which keys are missing.

    This handles the common Mistral failure mode where the model emits a short
    per-unit action dict *before* the full plan object, causing a first-wins
    extractor to pick up the wrong object.
    """

    _PLAN_KEYS = frozenset({"reasoning", "actions", "doctrine_reference"})
    candidates = _find_all_top_level_json_objects(raw_output)
    if not candidates:
        raise ModelOutputError("No JSON object found in model output.")

    # Pass 1: prefer a candidate that already satisfies the full plan schema.
    for raw in candidates:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and _PLAN_KEYS.issubset(parsed.keys()):
            return raw

    # Pass 2: return the first parseable object and let the validator report
    # exactly which required keys are missing.
    for raw in candidates:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelOutputError(
                f"Extracted JSON object is invalid: {exc.msg}."
            ) from exc
        if not isinstance(parsed, dict):
            raise ModelOutputError("Extracted JSON payload must be an object.")
        return raw

    raise ModelOutputError("Unterminated JSON object in model output.")


def _find_all_top_level_json_objects(text: str) -> list[str]:
    """Return every top-level balanced ``{…}`` object found in *text*.

    Each call restarts the depth counter from zero after a complete object is
    found, so nested objects inside a returned candidate are *not* returned
    separately.
    """

    results: list[str] = []
    pos = 0
    length = len(text)
    while pos < length:
        start = text.find("{", pos)
        if start < 0:
            break
        depth = 0
        in_string = False
        escaped = False
        end = -1
        for i in range(start, length):
            ch = text[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end >= 0:
            results.append(text[start : end + 1])
            pos = end + 1
        else:
            # Unterminated object — no further complete objects possible.
            break
    return results
