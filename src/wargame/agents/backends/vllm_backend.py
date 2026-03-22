"""vLLM-based local LLM backend for Linux/CUDA environments (e.g. Google Colab).

vLLM, torch, and transformers are optional dependencies. The class is always
importable, loads the heavy runtime lazily, and returns empty output if
generation fails so callers can fall back safely.

Colab usage notes
-----------------
* Pass ``gpu_memory_utilization`` (0.0–1.0) to leave headroom for the
  Colab runtime itself.
* Call ``unload()`` between experiments to free VRAM before loading a
  different model.
* ``model_path`` accepts a Google Drive mount path such as
  ``/content/drive/MyDrive/models/Qwen2.5-7B``.
* Use ``quantization="bitsandbytes"`` for 4-bit loading when VRAM is tight.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from typing import overload

from wargame.agents.local_llm import BackendResponse, LocalLLMBackend, LocalLLMConfig

# Sentinel so _require_vllm() raises lazily, not at import time.
_VLLM_MISSING = object()


@dataclass
class VLLMLocalLLMBackend(LocalLLMBackend):
    """Inference backend using vLLM on Linux/CUDA (including Google Colab).

    Model loading is deferred to the first ``generate()`` call.

    The backend supports both the repository's ``LocalLLMBackend`` contract
    ``generate(prompt, config) -> BackendResponse`` and a lightweight
    compatibility form ``generate(prompt, max_tokens) -> str`` for simple
    scripted integrations.

    Parameters
    ----------
    model_path:
        HuggingFace repo id, local path, or Google Drive mount path.
    model_id:
        Alias for ``model_path`` kept for older call sites.
    quantization:
        Quantization scheme forwarded to ``vllm.LLM``.  Use
        ``"bitsandbytes"`` for 4-bit INT4 loading.  ``None`` disables
        quantization (default).
    gpu_memory_utilization:
        Fraction of GPU memory vLLM may use (0.0–1.0).  Defaults to 0.85
        to leave headroom for the Colab runtime.
    dtype:
        PyTorch dtype string passed to ``vllm.LLM`` (e.g. ``"float16"``).
        ``"auto"`` lets vLLM decide based on the model config.
    """

    model_path: str | None = None
    model_id: str | None = None
    quantization: str | None = None
    gpu_memory_utilization: float = 0.85
    dtype: str = "auto"

    _llm: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize legacy and current model identifier constructor args."""

        resolved_model = self.model_path or self.model_id
        if resolved_model is None:
            raise ValueError("VLLMLocalLLMBackend requires model_path or model_id.")
        if self.model_path is not None and self.model_id is not None and self.model_path != self.model_id:
            raise ValueError("model_path and model_id must match when both are provided.")

        self.model_path = resolved_model
        self.model_id = resolved_model

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Instantiate ``vllm.LLM`` lazily."""
        vllm = _require_vllm()

        kwargs: dict[str, Any] = {
            "model": self.model_path,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
        }
        if self.quantization is not None:
            kwargs["quantization"] = self.quantization

        self._llm = vllm.LLM(**kwargs)

    # ------------------------------------------------------------------
    # LocalLLMBackend interface
    # ------------------------------------------------------------------

    @overload
    def generate(self, prompt: str, config: LocalLLMConfig) -> BackendResponse: ...

    @overload
    def generate(self, prompt: str, config: int) -> str: ...

    def generate(self, prompt: str, config: LocalLLMConfig | int) -> BackendResponse | str:
        """Run text generation via vLLM.

        When ``config`` is a :class:`LocalLLMConfig`, the method follows the
        repository's backend contract and returns :class:`BackendResponse`.
        When ``config`` is an ``int``, it is treated as ``max_tokens`` and the
        method returns the generated text directly for compatibility with
        simpler scripted integrations.

        The structured response path includes the usual metadata keys:
        ``"inference_time_s"``, ``"model_path"``, and ``"vllm"`` backend tags.
        """

        if isinstance(config, int):
            response = self._generate_response(
                prompt,
                LocalLLMConfig(
                    model_name=self.model_path,
                    max_tokens=config,
                ),
            )
            return response.content
        return self._generate_response(prompt, config)

    def _generate_response(
        self,
        prompt: str,
        config: LocalLLMConfig,
    ) -> BackendResponse:
        """Run text generation and package a safe backend response."""

        start = time.perf_counter()
        if self._llm is None:
            try:
                self._load()
            except Exception as exc:
                return self._failure_response(start, exc)

        try:
            vllm = _require_vllm()
            sampling_params = vllm.SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            outputs = self._llm.generate([prompt], sampling_params)
            elapsed = time.perf_counter() - start

            text: str = outputs[0].outputs[0].text if outputs else ""

            return BackendResponse(
                content=text,
                metadata={
                    "inference_time_s": round(elapsed, 4),
                    "model_path": self.model_path,
                    "backend": "vllm",
                    "quantization": self.quantization,
                    "gpu_memory_utilization": self.gpu_memory_utilization,
                    "failed": False,
                },
            )
        except Exception as exc:
            return self._failure_response(start, exc)

    def _failure_response(
        self,
        start: float,
        error: Exception,
    ) -> BackendResponse:
        """Return an empty response when model setup or generation fails."""

        return BackendResponse(
            content="",
            metadata={
                "inference_time_s": round(time.perf_counter() - start, 4),
                "model_path": self.model_path,
                "backend": "vllm",
                "quantization": self.quantization,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "failed": True,
                "error": f"{type(error).__name__}: {error}",
            },
        )

    # ------------------------------------------------------------------
    # Memory management (Colab / multi-experiment sessions)
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release the loaded model from GPU memory.

        Call this between experiments in long-running Colab sessions to
        avoid out-of-memory errors when switching model checkpoints.
        """
        if self._llm is None:
            return
        try:
            # vLLM does not expose a public unload API; delete the object and
            # let Python/torch release the CUDA tensors via garbage collection.
            del self._llm
        finally:
            self._llm = None
            _try_empty_cuda_cache()

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` if a model is currently resident in GPU memory."""
        return self._llm is not None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_vllm():
    """Return the ``vllm`` module; raise a clear ImportError when absent."""
    try:
        import vllm  # noqa: PLC0415

        return vllm
    except ImportError as exc:
        raise ImportError(
            "vllm is required for VLLMLocalLLMBackend. "
            "Install it with: pip install vllm  "
            "(Linux + CUDA only; not supported on macOS)"
        ) from exc


def _try_empty_cuda_cache() -> None:
    """Attempt to empty the PyTorch CUDA cache; silently ignore if torch is absent."""
    try:
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
