"""Red Force local-LLM agent specialization."""

from __future__ import annotations

from dataclasses import dataclass

from wargame.core.enums import Faction

from .local_llm import LocalLLMAgent
from .prompts import PromptRole


@dataclass(slots=True)
class RedAgent(LocalLLMAgent):
    """Red-role specialization of the generic local LLM agent wrapper."""

    name: str = "red_agent"
    faction: Faction = Faction.RED
    role: PromptRole = PromptRole.RED
