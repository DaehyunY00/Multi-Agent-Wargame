"""Local LLM backend implementations.

Both ``MLXLocalLLMBackend`` (Mac Apple Silicon) and ``VLLMLocalLLMBackend``
(Linux/CUDA) are always importable.  Their respective heavy dependencies
(mlx-lm, vllm) are loaded lazily on first ``generate()`` call so that
importing this package never triggers a model load or a hard import error.
"""

from __future__ import annotations

from wargame.agents.backends.chat_templates import (
    AutoChatTemplateAdapter,
    LlamaChatTemplateAdapter,
    MistralChatTemplateAdapter,
    QwenChatTemplateAdapter,
)
from wargame.agents.backends.mlx_backend import MLXLocalLLMBackend
from wargame.agents.backends.vllm_backend import VLLMLocalLLMBackend
from wargame.agents.local_llm import MockLocalLLMBackend

__all__ = [
    "AutoChatTemplateAdapter",
    "LlamaChatTemplateAdapter",
    "MistralChatTemplateAdapter",
    "MLXLocalLLMBackend",
    "MockLocalLLMBackend",
    "QwenChatTemplateAdapter",
    "VLLMLocalLLMBackend",
]
