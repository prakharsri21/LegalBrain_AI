"""Thread-safe in-memory TTL cache for generated answers."""

import hashlib
import json
import time
from collections import OrderedDict
from copy import copy
from threading import RLock
from typing import Any, Callable


def _copy_value(value: Any) -> Any:
    """Copy cached values to avoid exposing the stored object."""
    if isinstance(value, dict):
        return value.copy()
    return copy(value)


class ResponseCache:
    def __init__(
        self,
        max_entries: int = 128,
        ttl_seconds: int = 300,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be at least 1")

        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock or time.monotonic
        self._entries: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Return the cached value, or None when absent or expired."""
        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                self._misses += 1
                return None

            expires_at, value = entry

            if self._clock() >= expires_at:
                del self._entries[key]
                self._misses += 1
                return None

            self._entries.move_to_end(key)
            self._hits += 1
            return _copy_value(value)

    def set(self, key: str, value: Any) -> None:
        """Cache non-None values and evict least-recently-used entries."""
        if value is None:
            return

        with self._lock:
            self._entries[key] = (
                self._clock() + self.ttl_seconds,
                _copy_value(value),
            )
            self._entries.move_to_end(key)

            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Remove all entries and reset hit/miss counters."""
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, int]:
        """Return cache statistics after removing expired entries."""
        with self._lock:
            now = self._clock()

            expired_keys = [
                key
                for key, (expires_at, _) in self._entries.items()
                if now >= expires_at
            ]

            for key in expired_keys:
                del self._entries[key]

            return {
                "hits": self._hits,
                "misses": self._misses,
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
            }


def build_cache_key(
    *,
    question: str,
    max_tokens: int,
    temperature: float,
    prompt_version: str,
    model: str | None = None,
    model_name: str | None = None,
    adapter_path: str | None = None,
    adapter: str | None = None,
    context: str | None = None,
) -> str:
    """Build a stable key supporting both old and current argument names."""
    payload = {
        "model": model_name or model or "unknown",
        "adapter": adapter_path or adapter or "base",
        "question": question.strip(),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "prompt_version": prompt_version,
    }

    # Backward compatibility; the current app does not provide context.
    if context is not None:
        payload["context"] = context

    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
