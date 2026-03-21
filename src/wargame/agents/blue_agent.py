"""Blue Force local-LLM agent specialization."""

from __future__ import annotations

from dataclasses import dataclass

from wargame.core.enums import Faction

from .local_llm import LocalLLMAgent
from .prompts import PromptRole


@dataclass(slots=True)
class BlueAgent(LocalLLMAgent):
    """Blue-role specialization of the generic local LLM agent wrapper."""

    name: str = "blue_agent"
    faction: Faction = Faction.BLUE
    role: PromptRole = PromptRole.BLUE
