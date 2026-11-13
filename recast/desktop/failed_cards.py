"""失败任务卡片的建 UI（R6 拆分，P4）。

主窗口左栏「失败任务」组的建组与卡片重建原来都在
``main_window.py`` 里：这里是唯一来源，主窗口只留取数
（``MainWindow._failed_entries``，见 ``recast.core.failed_tasks``）
与刷新触发（``_refresh_failed_tasks`` 只做委托），卡片长什么样、
按钮文案、提示、失败集合不变时不重建（400ms 刷新别抢焦点）
都在这里。

文案与对象名与抽取前一字不差：卡片 ``failedCard``、文本
``failedText``、重试按钮 ``failedRetry``（文案“从失败步续跑”），
空态标签由调用方传入。
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def build_failed_group() -> tuple[QWidget, QLabel, QVBoxLayout]:
    """建「失败任务」组：空态标签 + 卡片纵向布局。

    返回 ``(group, empty_label, cards_layout)``，调用方挂到左栏并
    自己持有 ``failed_cards`` 列表与 ``_failed_sig``。
    """
    failed_group = QGroupBox("失败任务")
    fcol = QVBoxLayout(failed_group)
    empty_label = QLabel("没有失败的任务。")
    empty_label.setStyleSheet("color:#8b90a0;")
    fcol.addWidget(empty_label)
    cards_layout = QVBoxLayout()
    fcol.addLayout(cards_layout)
    return failed_group, empty_label, cards_layout


def build_failed_card(
    entry: dict[str, str], on_retry: Callable[[str], None]
) -> QFrame:
    """一张失败卡片：失败阶段 + 原因 + [从失败步续跑]。

    点击重试时调 ``on_retry(drama_id)``（主窗口传 ``_retry_failed``）。
    """
    card = QFrame()
    card.setObjectName("failedCard")
    card.setStyleSheet("QFrame#failedCard { border: 1px solid #5a3a3a; }")
    vbox = QVBoxLayout(card)
    label = QLabel(f"「{entry['name']}」失败在：{entry['stage']}\n{entry['reason']}")
    label.setWordWrap(True)
    label.setObjectName("failedText")
    vbox.addWidget(label)
    retry = QPushButton("从失败步续跑")
    retry.setObjectName("failedRetry")
    retry.setToolTip(
        "只重跑失败步及之后：已完成的步骤按检查点跳过，不从头重跑"
        "（同一任务续跑，不新建任务；历史面板的「一键复跑（新建任务）」是新建任务全量重跑）。"
    )
    retry.setProperty("drama_id", entry["drama_id"])
    retry.clicked.connect(
        lambda _checked=False, did=entry["drama_id"]: on_retry(did)
    )
    vbox.addWidget(retry)
    return card


def refresh_failed_cards(
    *,
    layout: QVBoxLayout,
    empty_label: QLabel,
    entries: list[dict[str, str]],
    prev_sig: Any,
    on_retry: Callable[[str], None],
) -> tuple[tuple, list[QFrame] | None]:
    """按失败集合重建卡片；集合没变就不重建（返回 ``(prev_sig, None)``）。

    调用方在返回的卡片列表不是 None 时更新自己的 ``_failed_sig``
    与 ``failed_cards``；是 None 时什么都不动。
    """
    sig = tuple((e["drama_id"], e["stage"], e["reason"]) for e in entries)
    if sig == prev_sig:
        return prev_sig, None
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
    cards: list[QFrame] = []
    empty_label.setVisible(not entries)
    for entry in entries:
        card = build_failed_card(entry, on_retry)
        layout.addWidget(card)
        cards.append(card)
    return sig, cards
