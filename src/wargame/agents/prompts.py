"""Prompt registry and role-specific prompt templates for local LLM agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PromptRole(StrEnum):
    """Canonical prompt roles supported by the current agent layer."""

    BLUE = "blue"
    RED = "red"
    WHITE = "white"


OUTPUT_CONTRACT = """
Respond with a single JSON object only.
Required top-level keys:
- reasoning: string
- doctrine_reference: string
- actions: array of action objects

Each action object must contain:
- unit_id: string
- action_type: one of hold, move, attack, support_by_fire, recon, withdraw
- posture: one of attack, defend, support, maneuver, withdraw
- target_hex: object with integer q and r, or null only for hold
""".strip()

BLUE_SYSTEM_PROMPT = """
You are the Blue Force commander in a tactical hex-grid wargame.
Prioritize mission progress while minimizing unnecessary friendly losses.
Use terrain, posture, and observed enemy positions to choose practical actions.
""".strip()

RED_SYSTEM_PROMPT = """
You are the Red Force commander defending against a Blue Force operation.
Favor defensive positioning, counterattack opportunities, and terrain advantage.
Use only the information visible in the report.
""".strip()

WHITE_CELL_SYSTEM_PROMPT = """
You are the White Cell controller for a tactical wargame.
When asked to produce actions, remain conservative, rule-aware, and explicit.
Prefer safe defensive defaults if the report is incomplete or ambiguous.
""".strip()


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """Prompt template metadata for one agent role."""

    role: PromptRole
    system_prompt: str
    output_contract: str = OUTPUT_CONTRACT

    def full_system_prompt(self) -> str:
        """Return the system prompt combined with the output contract."""

        return f"{self.system_prompt}\n\n{self.output_contract}"


@dataclass(frozen=True, slots=True)
class PromptRegistry:
    """Immutable registry of role-specific prompt templates."""

    prompts: dict[PromptRole, PromptSpec]

    def get(self, role: PromptRole) -> PromptSpec:
        """Return the prompt specification for the requested role."""

        try:
            return self.prompts[role]
        except KeyError as exc:
            raise KeyError(f"No prompt registered for role {role.value!r}.") from exc

    @classmethod
    def default(cls) -> "PromptRegistry":
        """Build the default prompt registry for blue, red, and white roles."""

        return cls(
            prompts={
                PromptRole.BLUE: PromptSpec(
                    role=PromptRole.BLUE,
                    system_prompt=BLUE_SYSTEM_PROMPT,
                ),
                PromptRole.RED: PromptSpec(
                    role=PromptRole.RED,
                    system_prompt=RED_SYSTEM_PROMPT,
                ),
                PromptRole.WHITE: PromptSpec(
                    role=PromptRole.WHITE,
                    system_prompt=WHITE_CELL_SYSTEM_PROMPT,
                ),
            }
        )


DEFAULT_PROMPT_REGISTRY = PromptRegistry.default()


def build_agent_prompt(system_prompt: str, situation_report: str) -> str:
    """Build a simple prompt body from a system prompt and state text."""

    return (
        f"SYSTEM:\n{system_prompt}\n\n"
        f"USER:\nSituation report follows.\n{situation_report}\n"
    )
