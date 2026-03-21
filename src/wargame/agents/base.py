"""Abstract agent interfaces for tactical decision-making components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from wargame.core.enums import Faction
from wargame.core.models import ActionCommand


@dataclass(slots=True)
class AgentDecision:
    """Structured decision bundle returned by an agent."""

    faction: Faction | None = None
    reasoning: str = ""
    doctrine_reference: str = ""
    actions: list[ActionCommand] = field(default_factory=list)
    used_fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BaseAgent(ABC):
    """Abstract interface shared by all tactical agents."""

    name: str
    faction: Faction | None = None

    @abstractmethod
    def decide(
        self,
        state_text: str,
        *,
        valid_unit_ids: Collection[str] = (),
    ) -> AgentDecision:
        """Produce a structured decision from a textual state description."""


@runtime_checkable
class StructuredStateAgent(Protocol):
    """Protocol for agents that can consume structured observed state directly."""

    def decide_view(
        self,
        state_view: object,
        *,
        valid_unit_ids: Collection[str] = (),
    ) -> AgentDecision:
        """Produce a decision from a structured faction-specific state view."""
