"""导入预检卡的建 UI（R6 拆分，P4）。

主窗口中栏「导入预检」组的建组与行状态点的刷新原来都在
``main_window.py`` 里：建组与按行刷点是纯 UI，判断（通过 / 警告 /
阻止）与文案在 ``recast.core.precheck``，这里只按传入的点与提示
写表。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QLabel, QTableWidget, QVBoxLayout, QWidget


def build_precheck_panel() -> tuple[QWidget, QLabel]:
    """建「导入预检」组（默认隐藏，选中一部剧才显示）。

    返回 ``(group, label)``：``label`` 初始文案“先选一部剧。”。
    """
    precheck_group = QGroupBox("导入预检")
    pcl = QVBoxLayout(precheck_group)
    label = QLabel("先选一部剧。")
    label.setWordWrap(True)
    pcl.addWidget(label)
    precheck_group.setVisible(False)
    return precheck_group, label


def paint_precheck_dot(
    table: QTableWidget, drama_id: str, dot: str, tip: str
) -> None:
    """刚到一个预检结果就刷新该行的状态点，不等 400ms 的整表重绘。

    按行号找该剧（第 1 列的 UserRole 存 drama_id），只改第 4 列
    （状态列）的点前缀与提示，状态本身的文字不动。
    """
    for row in range(table.rowCount()):
        item = table.item(row, 1)
        if item is None or item.data(Qt.ItemDataRole.UserRole) != drama_id:
            continue
        st = table.item(row, 4)
        if st is None:
            return
        raw = (st.text() or "").lstrip("✓⚠✗ ")
        st.setText(f"{dot}{raw}")
        st.setToolTip(tip)
        return
