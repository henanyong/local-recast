"""逐镜头增强对话框（R6 拆分，P3 第 3 波）：从 main_window.py 搬出。

搬之前它以 ``MainWindow._open_shot_enhance`` /
``MainWindow._toggle_shot_enhance`` 的形式住在主窗口里（约 80 行），
只依赖 service + 当前剧目 ID，和主窗口其余状态没有纠缠，所以整块
搬出来、主窗口只留选剧、空态提示和打开（薄调用）。

行为与文案原样保留，特别是 N29 的两条路文案：

* 提示行：\u201c两条路都支持：导入路逐镜头增强，剧本路在拼接后按标记时段增强。\u201d
* 主窗口 ``_open_shot_enhance`` 的 docstring 里同一句话也保留，
  ``tests/test_p3_main_window.py::test_shot_enhance_docstring_synced``
  在盯着它。

主窗口刻意还留着三处空态 ``QMessageBox.information``（没选剧 / 没
分集任务 / 没镜头记录）：那几处要走 ``main_window.QMessageBox``，
现有测试是按这个命名空间 monkeypatch 的，所以不搬进这里。
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from recast.desktop.dialog_utils import cap_dialog_to_screen, scroll_wrap

#: 对话框顶部的提示行（N29 文案，一字不改）。
HINT_TEXT = (
    "勾选即对该镜头开换后增强（hqdn3d + unsharp），和终检窗口同一份开关。\n"
    "两条路都支持：导入路逐镜头增强，剧本路在拼接后按标记时段增强。"
)


def collect_shot_enhance_rows(service: Any, drama_id: str) -> list[tuple[dict, dict]]:
    """当前剧目全部（分集任务，镜头）行：只要有 shot_code 的镜头。

    空 shot_code 的镜头行跳过——和搬之前主窗口里的循环同一规则。
    """
    rows: list[tuple[dict, dict]] = []
    for task in service.store.tasks_for_drama(drama_id):
        for shot in service.store.shots_for_task(task["id"]):
            code = str(shot.get("shot_code") or "").strip()
            if code:
                rows.append((task, shot))
    return rows


class ShotEnhanceDialog(QDialog):
    """逐镜头增强对话框：列出当前剧目的全部镜头，逐行开关。

    开关走 ``service.set_shot_enhance``，记在分集工作目录的
    shots_enhance.json 里，和终检窗口同一份 flags。只读预览之外的
    唯一写操作就是开关本身。``boxes`` 按 ``(task_id, shot_code)``
    索引每行复选框（主窗口为兼容保留同名属性时直接取这里）。
    """

    def __init__(
        self,
        service: Any,
        drama_id: str,
        rows: list[tuple[dict, dict]],
        *,
        drama_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.service = service
        self.setWindowTitle(f"逐镜头增强 — {drama_name or drama_id}")
        # P5/B3: the checkbox list is dynamic — a drama with many shots
        # used to push the [关闭] button off a 1280x720 screen. The rows
        # now live in a scroll area; the button bar is pinned at the
        # bottom and the dialog is clamped to the screen.
        cap_dialog_to_screen(self, 480, 480)
        content = QWidget()
        layout = QVBoxLayout(content)
        hint = QLabel(HINT_TEXT)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.boxes: dict[tuple[str, str], QCheckBox] = {}
        for task, shot in rows:
            task_id = task["id"]
            code = str(shot.get("shot_code"))
            flags = self.service.shot_enhance_flags(task_id)
            box = QCheckBox(f"{code}（{task.get('episode_name') or task_id}）")
            box.setChecked(bool(flags.get(code)))
            box.toggled.connect(
                lambda checked, tid=task_id, c=code, b=box: self._toggle(tid, c, checked, b)
            )
            self.boxes[(task_id, code)] = box
            layout.addWidget(box)
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color:#e08a8a;")
        layout.addWidget(self.error_label)
        buttons = QWidget()
        buttons_layout = QVBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        close = QPushButton("关闭")
        close.clicked.connect(self.accept)
        buttons_layout.addWidget(close)
        self.close_btn = close
        root = QVBoxLayout(self)
        root.addWidget(scroll_wrap(content, buttons))

    def _toggle(self, task_id: str, code: str, checked: bool, box: QCheckBox) -> None:
        """对话框里的一行开关：写 service.set_shot_enhance，失败回滚勾选。"""
        try:
            self.service.set_shot_enhance(task_id, code, checked)
        except Exception as exc:
            box.blockSignals(True)
            box.setChecked(not checked)
            box.blockSignals(False)
            self.error_label.setText(f"镜头 {code} 开关没存上：{exc}")
        else:
            self.error_label.setText("")
