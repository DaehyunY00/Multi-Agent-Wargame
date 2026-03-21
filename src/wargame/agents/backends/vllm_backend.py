"""vLLM-based local LLM backend for Linux/CUDA environments (e.g. Google Colab).

vLLM, torch, and transformers are optional dependencies.  The class is
always importable; ImportError is raised on the first ``generate()`` call
when the libraries are absent.

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

from wargame.agents.local_llm import BackendResponse, LocalLLMBackend, LocalLLMConfig

# Sentinel so _require_vllm() raises lazily, not at import time.
_VLLM_MISSING = object()


@dataclass
class VLLMLocalLLMBackend(LocalLLMBackend):
    """Inference backend using vLLM on Linux/CUDA (including Google Colab).

    Model loading is deferred to the first ``generate()`` call.

    Parameters
    ----------
    model_path:
        HuggingFace repo id, local path, or Google Drive mount path.
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

    model_path: str
    quantization: str | None = None
    gpu_memory_utilization: float = 0.85
    dtype: str = "auto"

    _llm: Any = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Instantiate ``vllm.LLM``; raises ``ImportError`` when vLLM is absent."""
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

    def generate(self, prompt: str, config: LocalLLMConfig) -> BackendResponse:
        """Run text generation via vLLM, measuring wall-clock inference time."""
        if self._llm is None:
            self._load()

        vllm = _require_vllm()
        sampling_params = vllm.SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        start = time.perf_counter()
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
