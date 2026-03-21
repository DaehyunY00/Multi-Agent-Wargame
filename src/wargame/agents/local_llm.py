"""Local LLM backend abstractions and a testable agent wrapper."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Any

from wargame.core.enums import Faction

from .base import AgentDecision, BaseAgent
from .parser import ActionParseError, ActionParser, ParsedActionPlan
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptRegistry, PromptRole


@dataclass(frozen=True, slots=True)
class LocalLLMConfig:
    """Configuration shared across local LLM backends."""

    model_name: str
    temperature: float = 0.7
    max_tokens: int = 512
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
            if not self.fallback_on_error:
                raise
            plan = self.parser.build_fallback_plan(
                unit_ids=set(valid_unit_ids),
                error=exc,
            )
            return self._build_decision(plan=plan, response=response, rendered_prompt=rendered_prompt)

        return self._build_decision(plan=plan, response=response, rendered_prompt=rendered_prompt)

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
    """Extract the first balanced JSON object from model output.

    This allows callers to recover JSON from fenced code blocks or prose-wrapped
    model output while still surfacing clear errors when no valid object exists.
    """

    start_index = raw_output.find("{")
    if start_index < 0:
        raise ModelOutputError("No JSON object found in model output.")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start_index, len(raw_output)):
        character = raw_output[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                candidate = raw_output[start_index : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ModelOutputError(
                        f"Extracted JSON object is invalid: {exc.msg}."
                    ) from exc
                if not isinstance(parsed, dict):
                    raise ModelOutputError("Extracted JSON payload must be an object.")
                return candidate

    raise ModelOutputError("Unterminated JSON object in model output.")
