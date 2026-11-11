"""Task / drama state machine (architecture chapter 9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from recast.core.constants import (
    MAX_STEP_RETRIES,
    ST_DONE,
    ST_FAILED,
    ST_NEED_ASSETS,
    ST_PENDING,
    ST_REVIEW,
    ST_RUNNING,
)

Event = Literal[
    "materials_missing",
    "materials_ok",
    "start",
    "stop",
    "step_ok",
    "step_fail",
    "unrecoverable",
    "all_done",
    "recheck",
]


class InvalidTransition(ValueError):
    pass


def format_missing(names: Iterable[str]) -> str:
    """Human-readable missing-file sentence. Never a vague「资料不全」."""
    items = [str(n).strip() for n in names if str(n).strip()]
    if not items:
        return ""
    return "缺少 " + "、".join(items)


@dataclass
class Transition:
    state: str
    progress: str
    retry_count: int = 0
    log: str = ""


def classify_error(exc: BaseException) -> Literal["stop", "fatal", "retry"]:
    from recast.runtime import CancelledError

    if isinstance(exc, CancelledError):
        return "stop"
    errno = getattr(exc, "errno", None)
    # POSIX ENOSPC=28, Windows ERROR_DISK_FULL=112
    if errno in {28, 112}:
        return "fatal"
    text = str(exc)
    lowered = text.lower()
    fatal_markers = (
        "no space left",
        "磁盘空间不足",
        "磁盘已满",
        "not enough space",
        "找不到模型",
        "缺少模型",
        "model not found",
        "找不到 ffmpeg",
        "找不到 ffprobe",
    )
    if any(m in text or m in lowered for m in fatal_markers):
        return "fatal"
    return "retry"


def apply(
    state: str,
    event: Event,
    *,
    missing: Iterable[str] | None = None,
    reason: str = "",
    retry_count: int = 0,
    step: str = "",
) -> Transition:
    """Pure transition. Callers persist the result."""
    missing_list = [str(n).strip() for n in (missing or []) if str(n).strip()]
    missing_msg = format_missing(missing_list)

    if event in {"materials_missing", "recheck"} and missing_list:
        return Transition(ST_NEED_ASSETS, missing_msg, retry_count=0, log=missing_msg)

    if event in {"materials_ok", "recheck"} and not missing_list:
        if state == ST_NEED_ASSETS:
            return Transition(ST_PENDING, "资料已补齐，等待执行", retry_count=0)
        if state in {ST_PENDING, ST_REVIEW, ST_FAILED, ST_DONE}:
            return Transition(state if state != ST_NEED_ASSETS else ST_PENDING, reason or "资料检查通过", retry_count=retry_count)
        return Transition(state, reason or "资料检查通过", retry_count=retry_count)

    if event == "start":
        if state == ST_NEED_ASSETS and missing_list:
            return Transition(ST_NEED_ASSETS, missing_msg, log="资料仍缺，不能进入处理中")
        return Transition(ST_RUNNING, reason or "开始处理", retry_count=0)

    if event == "stop":
        # Stop current task → 待执行, never 失败
        return Transition(ST_PENDING, "已停止当前任务，可继续", retry_count=0, log="用户停止，状态回到待执行")

    if event == "step_ok":
        label = f"步骤完成：{step}" if step else "步骤完成"
        return Transition(ST_RUNNING, label, retry_count=0)

    if event == "all_done":
        return Transition(ST_DONE, reason or "全部完成", retry_count=0)

    if event == "unrecoverable":
        text = reason or "发生不可恢复错误"
        return Transition(ST_FAILED, text, retry_count=retry_count, log=text)

    if event == "step_fail":
        nxt = retry_count + 1
        if nxt <= MAX_STEP_RETRIES:
            msg = f"{step or '步骤'}失败，准备第 {nxt} 次重试" + (f"：{reason}" if reason else "")
            return Transition(ST_RUNNING, msg, retry_count=nxt, log=msg)
        # After two retries, park the episode for human review; do not fail the whole drama here.
        msg = f"{step or '步骤'}连续失败 {MAX_STEP_RETRIES} 次，已标记为待人工检查"
        if reason:
            msg += f"：{reason}"
        return Transition(ST_REVIEW, msg, retry_count=nxt, log=msg)

    return Transition(state, reason or state, retry_count=retry_count)
