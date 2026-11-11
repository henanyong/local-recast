"""Cooperative cancellation for the workbench Stop button and CLI."""

from __future__ import annotations

import threading


class CancelledError(RuntimeError):
    """Raised when the user stops the current task."""


_lock = threading.Lock()
_active = threading.Event()


def bind_job(event: threading.Event | None = None) -> threading.Event:
    """Attach a fresh cancel event as the active job (avoids a leftover stop flag)."""
    ev = event or threading.Event()
    global _active
    with _lock:
        _active = ev
    return ev


def request_stop() -> None:
    with _lock:
        ev = _active
    ev.set()


def clear_stop() -> None:
    with _lock:
        ev = _active
    ev.clear()


def stopping() -> bool:
    with _lock:
        ev = _active
    return ev.is_set()


def check_cancel() -> None:
    if stopping():
        raise CancelledError("已停止当前任务")
