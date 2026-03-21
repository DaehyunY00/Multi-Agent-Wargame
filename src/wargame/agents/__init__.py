"""Agent interfaces, wrappers, and baseline agent placeholders."""

from .base import AgentDecision, BaseAgent, StructuredStateAgent
from .blue_agent import BlueAgent
from .local_llm import (
    BackendResponse,
    ChatMessage,
    ChatTemplateAdapter,
    DefaultChatTemplateAdapter,
    LocalLLMAgent,
    LocalLLMBackend,
    LocalLLMConfig,
    MockLocalLLMBackend,
    ModelOutputError,
    extract_json_object,
)
from .parser import ActionParser, ParsedActionPlan, STRICT_ACTION_SCHEMA
from .prompts import DEFAULT_PROMPT_REGISTRY, PromptRegistry, PromptRole
from .random_agent import RandomAgent
from .red_agent import RedAgent
from .rule_agent import RuleAgent, RuleBasedAgent
from .script_agent import ScriptAgent, ScriptBehavior
from .state_to_text import StateRenderer
from .white_cell import HeuristicWhiteCellAgent, WhiteCellAgent

__all__ = [
    "ActionParser",
    "AgentDecision",
    "BackendResponse",
    "BaseAgent",
    "BlueAgent",
    "ChatMessage",
    "ChatTemplateAdapter",
    "DEFAULT_PROMPT_REGISTRY",
    "DefaultChatTemplateAdapter",
    "LocalLLMAgent",
    "LocalLLMBackend",
    "LocalLLMConfig",
    "MockLocalLLMBackend",
    "ModelOutputError",
    "ParsedActionPlan",
    "PromptRegistry",
    "PromptRole",
    "RandomAgent",
    "RedAgent",
    "RuleBasedAgent",
    "RuleAgent",
    "ScriptAgent",
    "ScriptBehavior",
    "STRICT_ACTION_SCHEMA",
    "StateRenderer",
    "StructuredStateAgent",
    "HeuristicWhiteCellAgent",
    "WhiteCellAgent",
    "extract_json_object",
]
