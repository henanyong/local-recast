"""审核进度纯计算（R6 拆分，P3 第 3 波）。

主窗口 ``_review_stats`` 与人物库 ``_review_progress_stats`` 原来
各抄一份计数逻辑（已确认 / 待审核 / 聚类 / 低分），分母规则还一
度漂移过（N17/N22）：这里是唯一来源，两边只留取数（service 调用、
主窗口的缓存）与刷 UI（进度条可见性、标签文案、主窗口队列清空后
的自动提示跳转），计数本身不再各写一份。

分母规则（N17，勿改）：``total = 已确认 + 待审核全部项``——聚类
与低分样本都算进分母；只剩低分样本时也不会报 100%。
"""

from __future__ import annotations

from typing import Any


def review_progress_stats(
    characters: list[dict[str, Any]] | None,
    queue_items: list[dict[str, Any]] | None,
) -> dict[str, int]:
    """审核进度数：已确认 / 待审核 / 聚类 / 低分 / 总数 / 百分比。

    纯函数：只数传入的两份列表，不读 service、不碰 UI、不缓存。
    ``characters`` 是人物行（``confirmed`` 真值即已确认），
    ``queue_items`` 是审核队列项（``kind == "cluster"`` 为未确认聚类，
    ``kind == "sample"`` 为低分样本）。``total`` 为 0 时 ``percent``
    为 0（调用方据此隐藏进度条）。
    """
    characters = characters or []
    queue_items = queue_items or []
    confirmed = sum(1 for c in characters if c.get("confirmed"))
    pending = len(queue_items)
    clusters = sum(1 for item in queue_items if item.get("kind") == "cluster")
    low = sum(1 for item in queue_items if item.get("kind") == "sample")
    total = confirmed + pending
    return {
        "confirmed": confirmed,
        "pending": pending,
        "clusters": clusters,
        "low": low,
        "total": total,
        "percent": int(confirmed * 100 / total) if total > 0 else 0,
    }
