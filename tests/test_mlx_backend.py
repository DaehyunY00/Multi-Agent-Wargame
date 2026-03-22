"""Tests for MLXLocalLLMBackend, VLLMLocalLLMBackend, and chat template adapters.

All tests run without mlx / mlx-lm / vllm installed so that the CI environment
(non-Apple-Silicon / non-CUDA) can exercise the full suite.  Backend-specific
behaviour is verified via monkeypatching sys.modules.
"""

from __future__ import annotations

import sys

import pytest

from wargame.agents.backends import (
    AutoChatTemplateAdapter,
    LlamaChatTemplateAdapter,
    MistralChatTemplateAdapter,
    MLXLocalLLMBackend,
    MockLocalLLMBackend,
    QwenChatTemplateAdapter,
    VLLMLocalLLMBackend,
)
from wargame.agents.local_llm import (
    BackendResponse,
    ChatMessage,
    DefaultChatTemplateAdapter,
    LocalLLMBackend,
    LocalLLMConfig,
)

# ---------------------------------------------------------------------------
# Interface compatibility
# ---------------------------------------------------------------------------


def test_mlx_backend_is_importable_without_mlx() -> None:
    """MLXLocalLLMBackend must be importable even when mlx-lm is not installed."""
    from wargame.agents.backends.mlx_backend import MLXLocalLLMBackend as _  # noqa: F401


def test_mlx_backend_is_subclass_of_local_llm_backend() -> None:
    backend = MLXLocalLLMBackend(model_path="Qwen/Qwen2.5-7B-Instruct-4bit")
    assert isinstance(backend, LocalLLMBackend)


def test_mock_backend_is_subclass_of_local_llm_backend() -> None:
    backend = MockLocalLLMBackend(responses=["hello"])
    assert isinstance(backend, LocalLLMBackend)


def test_mock_backend_generate_returns_backend_response() -> None:
    backend = MockLocalLLMBackend(responses=["hello"])
    config = LocalLLMConfig(model_name="mock")
    response = backend.generate("prompt", config)
    assert isinstance(response, BackendResponse)
    assert response.content == "hello"


# ---------------------------------------------------------------------------
# MLX ImportError behaviour (mlx-lm absent)
# ---------------------------------------------------------------------------


def test_mlx_backend_raises_import_error_on_generate_when_mlx_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate() must raise ImportError with a helpful message when mlx-lm is absent."""
    monkeypatch.setitem(sys.modules, "mlx_lm", None)

    backend = MLXLocalLLMBackend(model_path="test-model")
    # Ensure lazy state is reset so _load() is triggered
    backend._model = None
    config = LocalLLMConfig(model_name="test")

    with pytest.raises(ImportError, match="mlx-lm"):
        backend.generate("hello", config)


def test_mlx_backend_metadata_contract() -> None:
    """BackendResponse from MLX must include inference_time_s, model_path, backend keys."""
    # We can verify the contract by inspecting the implementation without running inference.
    import inspect

    src = inspect.getsource(MLXLocalLLMBackend.generate)
    assert "inference_time_s" in src
    assert "model_path" in src
    assert '"mlx"' in src


# ---------------------------------------------------------------------------
# Qwen chat template
# ---------------------------------------------------------------------------


def test_qwen_template_wraps_messages_in_chatml_tokens() -> None:
    messages = [
        ChatMessage(role="system", content="You are a tactician."),
        ChatMessage(role="user", content="Move north."),
    ]
    result = QwenChatTemplateAdapter().render(messages)

    assert "<|im_start|>system\nYou are a tactician.<|im_end|>" in result
    assert "<|im_start|>user\nMove north.<|im_end|>" in result
    assert result.endswith("<|im_start|>assistant\n")


def test_qwen_template_single_message() -> None:
    messages = [ChatMessage(role="user", content="Attack.")]
    result = QwenChatTemplateAdapter().render(messages)
    assert result.startswith("<|im_start|>user")
    assert result.endswith("<|im_start|>assistant\n")


# ---------------------------------------------------------------------------
# Mistral chat template
# ---------------------------------------------------------------------------


def test_mistral_template_wraps_all_content_in_inst_block() -> None:
    messages = [
        ChatMessage(role="system", content="You are a tactician."),
        ChatMessage(role="user", content="Move north."),
    ]
    result = MistralChatTemplateAdapter().render(messages)

    assert result.startswith("[INST]")
    assert result.endswith("[/INST]")
    assert "You are a tactician." in result
    assert "Move north." in result


def test_mistral_template_single_user_message() -> None:
    messages = [ChatMessage(role="user", content="Retreat.")]
    result = MistralChatTemplateAdapter().render(messages)
    assert result == "[INST] Retreat. [/INST]"


# ---------------------------------------------------------------------------
# Llama chat template
# ---------------------------------------------------------------------------


def test_llama_template_uses_header_tokens() -> None:
    messages = [
        ChatMessage(role="system", content="You are a tactician."),
        ChatMessage(role="user", content="Move north."),
    ]
    result = LlamaChatTemplateAdapter().render(messages)

    assert result.startswith("<|begin_of_text|>")
    assert "<|start_header_id|>system<|end_header_id|>" in result
    assert "You are a tactician." in result
    assert "<|start_header_id|>user<|end_header_id|>" in result
    assert "Move north." in result
    assert "<|eot_id|>" in result
    assert result.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


# ---------------------------------------------------------------------------
# AutoChatTemplateAdapter detection
# ---------------------------------------------------------------------------


def test_auto_adapter_selects_qwen_for_qwen_path() -> None:
    adapter = AutoChatTemplateAdapter("Qwen/Qwen2.5-7B-Instruct-4bit")
    assert isinstance(adapter._delegate, QwenChatTemplateAdapter)


def test_auto_adapter_selects_mistral_for_mistral_path() -> None:
    adapter = AutoChatTemplateAdapter("mistralai/Mistral-7B-Instruct-v0.3")
    assert isinstance(adapter._delegate, MistralChatTemplateAdapter)


def test_auto_adapter_selects_llama_for_llama_path() -> None:
    adapter = AutoChatTemplateAdapter("meta-llama/Llama-3.1-8B-Instruct")
    assert isinstance(adapter._delegate, LlamaChatTemplateAdapter)


def test_auto_adapter_falls_back_to_default_for_unknown_path() -> None:
    adapter = AutoChatTemplateAdapter("some-unknown-model-xyz")
    assert isinstance(adapter._delegate, DefaultChatTemplateAdapter)


def test_auto_adapter_detection_is_case_insensitive() -> None:
    adapter = AutoChatTemplateAdapter("QWEN2.5-7B")
    assert isinstance(adapter._delegate, QwenChatTemplateAdapter)


def test_auto_adapter_render_matches_delegate_output() -> None:
    messages = [
        ChatMessage(role="system", content="Sys"),
        ChatMessage(role="user", content="User"),
    ]
    auto = AutoChatTemplateAdapter("Qwen/Qwen2.5-7B")
    direct = QwenChatTemplateAdapter()
    assert auto.render(messages) == direct.render(messages)


# ---------------------------------------------------------------------------
# VLLMLocalLLMBackend — interface and error-handling
# ---------------------------------------------------------------------------


def test_vllm_backend_is_importable_without_vllm() -> None:
    """VLLMLocalLLMBackend must be importable even when vllm is not installed."""
    from wargame.agents.backends.vllm_backend import VLLMLocalLLMBackend as _  # noqa: F401


def test_vllm_backend_is_subclass_of_local_llm_backend() -> None:
    backend = VLLMLocalLLMBackend(model_path="meta-llama/Llama-3.1-8B-Instruct")
    assert isinstance(backend, LocalLLMBackend)


def test_vllm_backend_accepts_model_id_alias() -> None:
    backend = VLLMLocalLLMBackend(model_id="meta-llama/Llama-3.1-8B-Instruct")
    assert backend.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert backend.model_path == "meta-llama/Llama-3.1-8B-Instruct"


def test_vllm_backend_default_parameters() -> None:
    backend = VLLMLocalLLMBackend(model_path="some/model")
    assert backend.quantization is None
    assert backend.gpu_memory_utilization == pytest.approx(0.85)
    assert backend.dtype == "auto"
    assert backend.is_loaded is False


def test_vllm_backend_bitsandbytes_quantization_stored() -> None:
    backend = VLLMLocalLLMBackend(model_path="some/model", quantization="bitsandbytes")
    assert backend.quantization == "bitsandbytes"


def test_vllm_backend_returns_empty_output_on_generate_when_vllm_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate() should fail safely with empty output when vllm is absent."""
    monkeypatch.setitem(sys.modules, "vllm", None)

    backend = VLLMLocalLLMBackend(model_path="test-model")
    backend._llm = None  # ensure lazy load triggers
    config = LocalLLMConfig(model_name="test")

    response = backend.generate("hello", config)

    assert isinstance(response, BackendResponse)
    assert response.content == ""
    assert response.metadata["backend"] == "vllm"
    assert response.metadata["failed"] is True
    assert "error" in response.metadata


def test_vllm_backend_unload_is_safe_when_not_loaded() -> None:
    """unload() on an unloaded backend must not raise."""
    backend = VLLMLocalLLMBackend(model_path="some/model")
    backend.unload()  # should be a no-op
    assert backend.is_loaded is False


def test_vllm_backend_unload_clears_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """unload() sets _llm to None and flips is_loaded to False."""
    backend = VLLMLocalLLMBackend(model_path="some/model")
    # Simulate a loaded state with a sentinel object
    backend._llm = object()
    assert backend.is_loaded is True

    # Suppress torch.cuda.empty_cache if torch isn't available
    monkeypatch.setitem(sys.modules, "torch", None)
    backend.unload()

    assert backend.is_loaded is False
    assert backend._llm is None


def test_vllm_backend_google_drive_path_is_accepted() -> None:
    """model_path accepts arbitrary filesystem paths including Google Drive mounts."""
    gdrive_path = "/content/drive/MyDrive/models/Qwen2.5-7B"
    backend = VLLMLocalLLMBackend(model_path=gdrive_path)
    assert backend.model_path == gdrive_path


def test_vllm_backend_metadata_contract() -> None:
    """BackendResponse metadata must include the required keys."""
    import inspect  # noqa: PLC0415

    src = inspect.getsource(VLLMLocalLLMBackend.generate)
    assert "inference_time_s" in src
    assert "model_path" in src
    assert '"vllm"' in src


def test_vllm_backend_generate_with_mock_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """generate() returns a BackendResponse when vllm is mocked end-to-end."""
    import types  # noqa: PLC0415

    # Build a minimal fake vllm module
    fake_vllm = types.SimpleNamespace()

    class FakeSamplingParams:
        def __init__(self, **_): ...

    class FakeOutput:
        text = "HOLD all units."

    class FakeRequestOutput:
        outputs = [FakeOutput()]

    class FakeLLM:
        def __init__(self, **_): ...

        def generate(self, prompts, params):
            return [FakeRequestOutput()]

    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams

    # Patch both the module cache and the backend's own import
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    backend = VLLMLocalLLMBackend(model_path="fake/model")
    backend._llm = None  # force lazy load path
    config = LocalLLMConfig(model_name="fake", temperature=0.5, max_tokens=64)

    response = backend.generate("Describe the situation.", config)

    assert isinstance(response, BackendResponse)
    assert response.content == "HOLD all units."
    assert response.metadata["backend"] == "vllm"
    assert response.metadata["model_path"] == "fake/model"
    assert "inference_time_s" in response.metadata
    assert response.metadata["failed"] is False


def test_vllm_backend_generate_returns_text_for_max_tokens_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compatibility signature should return plain text for simple callers."""
    import types  # noqa: PLC0415

    fake_vllm = types.SimpleNamespace()

    class FakeSamplingParams:
        def __init__(self, **_): ...

    class FakeOutput:
        text = "attack-right-flank"

    class FakeRequestOutput:
        outputs = [FakeOutput()]

    class FakeLLM:
        def __init__(self, **_): ...

        def generate(self, prompts, params):
            return [FakeRequestOutput()]

    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    backend = VLLMLocalLLMBackend(model_id="fake/model")

    response = backend.generate("Describe the situation.", 32)

    assert response == "attack-right-flank"
