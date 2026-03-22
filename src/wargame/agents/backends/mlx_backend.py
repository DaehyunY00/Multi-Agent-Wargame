"""MLX-based local LLM backend for Apple Silicon (Mac M-series).

mlx and mlx-lm are optional dependencies.  The class is importable in any
environment; ImportError is raised on first ``generate()`` call when the
libraries are absent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from wargame.agents.local_llm import BackendResponse, LocalLLMBackend, LocalLLMConfig


@dataclass
class MLXLocalLLMBackend(LocalLLMBackend):
    """Local inference backend using mlx-lm on Apple Silicon.

    Model loading is deferred to the first ``generate()`` call so that
    importing this module never triggers a heavy model load.
    """

    model_path: str
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        """Load model and tokenizer via mlx_lm (lazy, called once)."""
        try:
            import mlx_lm  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for MLXLocalLLMBackend. "
                "Install it with: pip install mlx-lm"
            ) from exc

        self._model, self._tokenizer = mlx_lm.load(self.model_path)

    def generate(self, prompt: str, config: LocalLLMConfig) -> BackendResponse:
        """Run text generation via mlx_lm, measuring wall-clock inference time."""
        if self._model is None:
            self._load()

        try:
            import mlx_lm  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for MLXLocalLLMBackend. "
                "Install it with: pip install mlx-lm"
            ) from exc

        from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415

        sampler = make_sampler(temp=config.temperature)
        start = time.perf_counter()
        output: str = mlx_lm.generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        elapsed = time.perf_counter() - start

        return BackendResponse(
            content=output,
            metadata={
                "inference_time_s": round(elapsed, 4),
                "model_path": self.model_path,
                "backend": "mlx",
            },
        )
