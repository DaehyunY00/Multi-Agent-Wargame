"""Tests for local-LLM experiment configuration and CLI documentation alignment."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


def test_pyproject_declares_python_311_and_llm_mlx_extra() -> None:
    """Packaging metadata should match the codebase's actual local-LLM requirements."""

    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)

    project = data["project"]
    optional_dependencies = project["optional-dependencies"]

    assert project["requires-python"] == ">=3.11"
    assert "llm-mlx" in optional_dependencies
    assert any("mlx-lm" in dependency for dependency in optional_dependencies["llm-mlx"])


def test_single_game_help_mentions_local_llm_and_fog_presets() -> None:
    """The single-game CLI help should document local_llm specs and experiment presets."""

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_single_game.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    help_text = " ".join(result.stdout.split())

    assert "local_llm:<model_id_or_path>" in help_text
    assert "Recommended presets: baseline=8, local_llm=5." in help_text
    assert "Recommended presets: baseline=3, local_llm=2." in help_text
    assert "--backend" in help_text
    assert "use --backend vllm for Mistral/Llama on Colab" in help_text


def test_batch_help_mentions_local_llm_matchups_and_backend_hint() -> None:
    """The batch CLI help should document local_llm matchup syntax and backend usage."""

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_batch.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    help_text = " ".join(result.stdout.split())

    assert "local_llm:" in help_text
    assert "Qwen2.5-7B-Instruct-4bit,rule" in help_text
    assert "Recommended presets: baseline=8, local_llm=5." in help_text
    assert "Recommended presets: baseline=3, local_llm=2." in help_text
    assert "use --backend vllm for Mistral/Llama on Colab" in help_text
