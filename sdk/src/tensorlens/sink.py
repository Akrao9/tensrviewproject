"""Where captured records go. The capture path only ever calls `write` from a worker thread."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any, Protocol


class Sink(Protocol):
    def write(self, records: list[dict[str, Any]]) -> None: ...

    def close(self) -> None: ...


class MemorySink:
    """Keeps records in a list. For tests and notebooks."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def write(self, records: list[dict[str, Any]]) -> None:
        with self._lock:
            self.records.extend(records)

    def close(self) -> None:
        pass


def _finite(value: Any) -> Any:
    """JSON has no NaN or infinity, and an overflow is exactly what must not be lost."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


class JsonlSink:
    """One JSON object per captured step. Non-finite numbers are written as the strings
    "nan", "inf" and "-inf" so the file stays standard JSON."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self._file.write(json.dumps(_finite(record), separators=(",", ":"), allow_nan=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
