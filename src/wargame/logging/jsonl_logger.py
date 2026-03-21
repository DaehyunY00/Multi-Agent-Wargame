"""JSON Lines logging for experiment and turn records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from wargame.core.models import GameState, TurnResult


@dataclass(slots=True)
class JsonlLogger:
    """Write structured turn-by-turn records to a JSONL file."""

    path: Path
    context: dict[str, Any] = field(default_factory=dict)

    def set_context(self, **context: Any) -> None:
        """Replace the current logger context for subsequent records."""

        self.context = dict(context)

    def clear(self) -> None:
        """Remove the current log file if it already exists."""

        if self.path.exists():
            self.path.unlink()

    def log_turn(self, result: TurnResult, state: GameState | None = None) -> None:
        """Record a single turn result and optional state snapshot."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = _serialize(result)
        if self.context:
            record["context"] = _serialize(self.context)
        if state is not None:
            record["state"] = _serialize(state)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def flush(self) -> None:
        """Flush any buffered records to disk.

        The current implementation writes each record immediately, so this is a
        no-op kept for interface stability.
        """

        return None


def _serialize(value: Any) -> Any:
    """Serialize nested dataclasses, enums, and keyed maps into JSON-friendly data."""

    if is_dataclass(value):
        return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {_serialize_key(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _serialize_key(key: Any) -> str:
    """Serialize dictionary keys that are not naturally JSON strings."""

    if isinstance(key, str):
        return key
    if hasattr(key, "q") and hasattr(key, "r"):
        return f"{key.q},{key.r}"
    if isinstance(key, Enum):
        return key.value
    return str(key)
