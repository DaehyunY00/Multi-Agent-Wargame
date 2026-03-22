"""Focused tests for the repeated local-LLM stability CLI."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def test_collect_local_llm_stats_and_summary_aggregation(tmp_path: Path) -> None:
    """The stability helper should count only blue/red local-LLM decisions."""

    module = _load_stability_module()
    log_path = tmp_path / "llm.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "turn": 1,
                "metadata": {
                    "blue": {
                        "used_fallback": False,
                        "metadata": {
                            "decision_source": "local_llm",
                            "json_parse_success": True,
                        },
                    },
                    "red": {
                        "used_fallback": True,
                        "metadata": {
                            "decision_source": "local_llm",
                            "json_parse_success": False,
                        },
                    },
                    "white_cell": {
                        "used_fallback": True,
                        "metadata": {
                            "decision_source": "local_llm",
                            "json_parse_success": False,
                        },
                    },
                },
            },
            {
                "turn": 2,
                "metadata": {
                    "blue": {
                        "used_fallback": False,
                        "metadata": {"decision_source": "structured_state"},
                    },
                    "red": {
                        "used_fallback": False,
                        "metadata": {
                            "decision_source": "local_llm",
                            "json_parse_success": True,
                        },
                    },
                },
            },
        ],
    )

    stats = module.collect_local_llm_stats(log_path)
    assert stats.decision_count == 3
    assert stats.fallback_count == 1
    assert stats.parse_success_count == 2

    summary = module.build_stability_summary(
        [
            module.StabilityRunRecord(
                seed=0,
                completed=True,
                abnormal_termination=False,
                output_path=str(log_path),
                fallback_count=stats.fallback_count,
                parse_success_count=stats.parse_success_count,
                llm_decision_count=stats.decision_count,
            ),
            module.StabilityRunRecord(
                seed=1,
                completed=False,
                abnormal_termination=True,
                output_path=None,
                fallback_count=0,
                parse_success_count=0,
                llm_decision_count=0,
                error="mock failure",
            ),
        ]
    )

    assert summary["total_runs"] == 2
    assert summary["completed_runs"] == 1
    assert summary["abnormal_termination_count"] == 1
    assert summary["fallback_count"] == 1
    assert summary["fallback_rate"] == 1 / 3
    assert summary["parse_success_rate"] == 2 / 3
    assert summary["per_run_output_paths"] == [str(log_path)]


def test_check_llm_stability_cli_writes_summary_files(tmp_path: Path) -> None:
    """The repeated-run CLI should orchestrate runs and save summaries."""

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "stability"
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "check_llm_stability.py"),
            "--scenario",
            "s1_open_encounter",
            "--blue-agent",
            "rule",
            "--red-agent",
            "rule",
            "--seed-count",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    payload = json.loads(result.stdout)
    summary_path = output_dir / "stability_summary.json"
    text_path = output_dir / "stability_summary.txt"
    assert summary_path.exists()
    assert text_path.exists()
    assert payload["total_runs"] == 2
    assert payload["completed_runs"] == 2
    assert payload["abnormal_termination_count"] == 0
    assert len(payload["per_run_output_paths"]) == 2

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["per_run_output_paths"] == payload["per_run_output_paths"]
    assert "Local LLM Stability Summary" in text_path.read_text(encoding="utf-8")


def _load_stability_module():
    """Import the standalone stability script as a Python module."""

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "check_llm_stability.py"
    spec = importlib.util.spec_from_file_location("check_llm_stability", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Persist JSONL records for log-driven stability tests."""

    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
