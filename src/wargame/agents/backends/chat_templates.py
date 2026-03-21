"""Chat template adapters for model-family-specific prompt formatting.

Each adapter converts a sequence of generic ``ChatMessage`` objects into the
prompt string expected by a particular model family.  ``AutoChatTemplateAdapter``
detects the family automatically from the model path string.
"""

from __future__ import annotations

from collections.abc import Sequence

from wargame.agents.local_llm import ChatMessage, ChatTemplateAdapter, DefaultChatTemplateAdapter


class QwenChatTemplateAdapter(ChatTemplateAdapter):
    """ChatML template used by Qwen2 / Qwen2.5 models.

    Format::

        <|im_start|>system
        ...<|im_end|>
        <|im_start|>user
        ...<|im_end|>
        <|im_start|>assistant
    """

    def render(self, messages: Sequence[ChatMessage]) -> str:
        parts = [f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>" for msg in messages]
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


class MistralChatTemplateAdapter(ChatTemplateAdapter):
    """[INST] template used by Mistral-7B-Instruct models.

    System and user messages are merged into a single ``[INST] … [/INST]``
    block because Mistral v1/v2 does not have a dedicated system turn.
    """

    def render(self, messages: Sequence[ChatMessage]) -> str:
        combined = "\n".join(msg.content for msg in messages)
        return f"[INST] {combined} [/INST]"


class LlamaChatTemplateAdapter(ChatTemplateAdapter):
    """Header-based template used by Llama-3 / Llama-3.1 models.

    Format::

        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        ...<|eot_id|><|start_header_id|>user<|end_header_id|>

        ...<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """

    _BOS = "<|begin_of_text|>"
    _EOT = "<|eot_id|>"
    _HEADER_START = "<|start_header_id|>"
    _HEADER_END = "<|end_header_id|>"

    def render(self, messages: Sequence[ChatMessage]) -> str:
        parts: list[str] = [self._BOS]
        for msg in messages:
            parts.append(
                f"{self._HEADER_START}{msg.role}{self._HEADER_END}\n\n"
                f"{msg.content}{self._EOT}"
            )
        parts.append(f"{self._HEADER_START}assistant{self._HEADER_END}\n\n")
        return "".join(parts)


_FAMILY_MAP: dict[str, type[ChatTemplateAdapter]] = {
    "qwen": QwenChatTemplateAdapter,
    "mistral": MistralChatTemplateAdapter,
    "llama": LlamaChatTemplateAdapter,
}


class AutoChatTemplateAdapter(ChatTemplateAdapter):
    """Detect the model family from *model_path* and delegate rendering.

    Detection is case-insensitive substring matching against the keys of
    ``_FAMILY_MAP``.  Falls back to ``DefaultChatTemplateAdapter`` when no
    known family is found.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        lower = model_path.lower()
        for key, adapter_cls in _FAMILY_MAP.items():
            if key in lower:
                self._delegate: ChatTemplateAdapter = adapter_cls()
                return
        self._delegate = DefaultChatTemplateAdapter()

    def render(self, messages: Sequence[ChatMessage]) -> str:
        return self._delegate.render(messages)
