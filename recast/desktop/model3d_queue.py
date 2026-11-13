"""3D 生成队列对话框（PM §四 5.2/N16）。

TripoSG 生成一张图要几分钟，8GB 显存下一次只能跑一个。以前点完
「生成 3D 模型」只能干等：不知道排到哪、失败了也没有重试。这个
对话框把待生成的人物排成队列，一次只跑一个，前一个完成/失败后才
跑下一个；失败的行显示失败原因，点 [重试] 重新入队。

串行机制：对话框内自带串行 worker（一个 daemon 线程 + deque），
没有复用 ``recast.core.scheduler.Scheduler``。Scheduler 的 Job 是
分集/任务绑定的（drama_id + task_id，dispatch 走整集处理流水线），
拿它跑单人物 3D 生成要造假分集/假任务行，抽象不对，还会把 3D
生成和分集队列的并发/重试语义搅在一起。8GB 下串行是 TripoSG
后端自己的要求（generate3d.py 模块 docstring），队列层只保证
“同一时刻最多一个 generate_fn 在跑”。

生成本身走已有接口，不改后端：默认 ``generate_fn`` 就是
``service.generate_model_3d(character, image=image_ref)``，它内部
调 ``generate_model_glb``，生成结果写进人物文件夹 ``model/`` 并在
meta.json 注册、装配身体指向它（注册流程见
recast/assets3d/generate3d.py）。``generate_fn`` 可注入，测试用假
后端替掉真实 TripoSG 调用，绝不真跑几分钟的生成。

P1 教训：耗时操作在后台线程，绝不在 paint/定时器刷新里 exec 模态
框——本对话框全程没有任何 QMessageBox，失败只写进失败行。

N31：生成完成/失败不再静默——worker 跑完一行就发 ``item_finished``
信号（人物，状态，失败原因），人物库接到后在状态栏给一行轻提示，
完成时还自动刷新人物库（见 library_window._on_model3d_item_finished）。
对话框关掉也不静默：``close()`` 只是隐藏（人物库缓存了对话框实例，
见 library_window._model3d_queue_dialog_obj），daemon worker 关窗后
继续跑、只在进程退出时停；关窗时若队列未空，``on_close`` 回调会让
人物库在状态栏留一行“3D 生成进行中（n 个排队），完成后在人物库刷新
可见”。“关窗后续跑”是按代码结构（缓存实例 + daemon 线程）写的，
Windows 上真跑一次的实测确认仍在验证清单里，见最终报告。

N36：worker 线程与 GUI 线程共用一把 meta.json 写锁
（``recast.core.write_lock.META_WRITE_LOCK``）：worker 绕整个
``generate_fn`` 调用持锁，GUI 侧所有 meta 写经
``characters.save_meta`` 持同一把锁；``save_meta`` 本身在锁内做
顶层合并 + 原子替换，防“读-改-写”丢字段与半截文件。
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# N36：与 GUI 线程共用同一把 meta.json 写锁（recast.core.write_lock），
# 不要另起一把。worker 绕 generate_fn 持锁，save_meta 里也持同一把。
from recast.core.write_lock import META_WRITE_LOCK, meta_write_lock  # noqa: E402

# 队列状态文案（PM 5.2：每人物一行，排队/生成中/失败/完成）。
ST_QUEUED = "排队"
ST_RUNNING = "生成中"
ST_FAILED = "失败"
ST_DONE = "完成"

# 生成中行显示的阶段文案：TripoSG 一张图要几分钟，告诉用户别关窗口。
STAGE_WAITING = "等待生成（8GB 显存下一次只跑一个）"
STAGE_RUNNING = "TripoSG 生成中…一张图要几分钟，请勿关闭"


@dataclass
class Model3dQueueItem:
    """队列里的一行：一个人物的一次 3D 生成。"""

    character: str
    image_ref: str | Path | None = None
    status: str = ST_QUEUED
    stage: str = STAGE_WAITING
    error: str = ""
    result: dict[str, Any] | None = None


class _QueueSignals(QObject):
    """Worker 线程 → GUI 线程的唯一通道：只发信号，不碰任何控件。"""

    item_changed = Signal(str)  # character key whose row changed
    queue_changed = Signal()
    # N31：一行跑完（完成/失败）时发：人物，状态（"完成"/"失败"），失败原因。
    item_finished = Signal(str, str, str)


GenerateFn = Callable[[str, Any], dict[str, Any]]


class Model3dQueueDialog(QDialog):
    """3D 生成队列：每人物一行，串行跑，失败可重试。

    ``service`` 是 RecastService；默认生成函数就是
    ``service.generate_model_3d``（已有注册流程，不改后端）。
    ``generate_fn`` 可注入 ``(character, image_ref) -> dict`` 的假后端，
    供测试或第二波接线时替换。构造后用 ``enqueue(character, image_ref)``
    入队；``refresh()`` 按当前队列状态重建表格。

    N31：``on_item_finished(character, status, error)`` 在一行跑完时
    （GUI 线程）回调一次，人物库用它写状态栏轻提示 + 完成时刷新；
    ``on_close(pending_count)`` 在对话框关掉时回调，人物库用它在队
    列未空时留一行“进行中”提示。两个回调都不弹模态框。
    关窗行为：本对话框被人物库缓存，``close()`` 只隐藏不销毁，
    daemon worker 关窗后继续跑、只在进程退出时停（Windows 实测待确认）。
    """

    def __init__(
        self,
        service: Any | None,
        parent: QWidget | None = None,
        *,
        generate_fn: GenerateFn | None = None,
        on_item_finished: Callable[[str, str, str], None] | None = None,
        on_close: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.service = service
        if generate_fn is not None:
            self._generate_fn = generate_fn
        elif service is not None:
            self._generate_fn = lambda character, image_ref: service.generate_model_3d(
                character, image=image_ref
            )
        else:
            raise ValueError("Model3dQueueDialog 需要 service 或 generate_fn，二者至少给一个。")
        self._on_item_finished = on_item_finished
        self._on_close = on_close
        self._signals = _QueueSignals()
        self._signals.item_changed.connect(self.refresh)
        self._signals.queue_changed.connect(self.refresh)
        self._signals.item_finished.connect(self._handle_item_finished)
        self._lock = threading.Lock()
        self._items: dict[str, Model3dQueueItem] = {}
        self._order: list[str] = []  # 入队顺序的人物 key
        self._pending: deque[str] = deque()
        self._worker_alive = False

        self.setWindowTitle("3D 生成队列")
        self.resize(760, 420)
        self.summary_label = QLabel("队列为空。")
        self.empty_label = QLabel(
            "队列为空。去人物库点「生成 3D 模型」，人物会自动排进这里；"
            "8GB 显存下一次只生成一个，前一个完成/失败后才跑下一个。"
        )
        self.empty_label.setWordWrap(True)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["人物", "状态", "进度 / 阶段", "失败原因", "操作"])
        self.table.setColumnWidth(0, 120)
        self.table.setColumnWidth(1, 70)
        self.table.setColumnWidth(2, 280)
        self.table.setColumnWidth(3, 200)
        self.table.verticalHeader().setDefaultSectionSize(34)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        row = QHBoxLayout()
        row.addWidget(refresh_btn)
        row.addStretch(1)
        row.addWidget(close_btn)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("一次只生成一个（8GB 显存）。失败的行会显示原因，点 [重试] 重新排队。"))
        layout.addWidget(self.summary_label)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.table, 1)
        layout.addLayout(row)
        self.refresh()

    # -- 入队 / 重试（第二波接线直接用这两个方法） -------------------------

    def enqueue(self, character: str, image_ref: str | Path | None = None) -> Model3dQueueItem:
        """把一个人物的 3D 生成排进队列，返回该行。

        同一人物只有一行：已在排队/生成中时重复入队直接返回原行，不
        重复排；已完成/失败时重新入队会把该行重置为排队并重新跑（再
        生成一次）。``image_ref`` 为空时生成函数用人物第一张抠像。
        """
        character = (character or "").strip()
        if not character:
            raise ValueError("入队 3D 生成队列需要人物名/人物 ID。")
        with self._lock:
            item = self._items.get(character)
            if item is None:
                item = Model3dQueueItem(character=character, image_ref=image_ref)
                self._items[character] = item
                self._order.append(character)
            elif item.status in (ST_QUEUED, ST_RUNNING):
                return item  # 已在队列里，不重复排
            else:
                item.image_ref = image_ref
                item.status = ST_QUEUED
                item.stage = STAGE_WAITING
                item.error = ""
                item.result = None
            if character not in self._pending:
                self._pending.append(character)
            self._ensure_worker_locked()
        self.refresh()
        return item

    def retry(self, character: str) -> Model3dQueueItem | None:
        """失败行重新入队；不是失败行返回 None（什么都不做）。"""
        character = (character or "").strip()
        with self._lock:
            item = self._items.get(character)
            if item is None or item.status != ST_FAILED:
                return None
        return self.enqueue(character, item.image_ref)

    def pending_count(self) -> int:
        """还没跑完的行数（排队 + 生成中），供关窗时的“进行中”提示用。"""
        with self._lock:
            return sum(
                1 for item in self._items.values() if item.status in (ST_QUEUED, ST_RUNNING)
            )

    def _handle_item_finished(self, character: str, status: str, error: str) -> None:
        """GUI 线程：把 worker 的完成/失败通知交给人物库（只调回调，不弹框）。"""
        if self._on_item_finished is not None:
            self._on_item_finished(character, status, error)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """关窗只隐藏不销毁（人物库缓存了实例，worker 继续跑）。

        队列未空时通过 on_close 告诉人物库在状态栏留一行“进行中”；
        队列已空时不打扰。绝不改关窗行为、不弹任何框。
        """
        if self._on_close is not None:
            self._on_close(self.pending_count())
        super().closeEvent(event)

    def queue_items(self) -> list[dict[str, Any]]:
        """当前队列快照（dict 列表，供第二波接线 / 测试读状态）。"""
        with self._lock:
            return [
                {
                    "character": item.character,
                    "status": item.status,
                    "stage": item.stage,
                    "error": item.error,
                    "result": item.result,
                }
                for key in self._order
                for item in [self._items[key]]
            ]

    def wait_until_idle(self, timeout: float = 30.0) -> bool:
        """等到队列里既无排队也无生成中的行（测试用）；超时返回 False。

        轮询时处理 Qt 事件，worker 发来的信号能正常刷新表格。
        """
        import time

        from PySide6.QtWidgets import QApplication

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                busy = any(
                    item.status in (ST_QUEUED, ST_RUNNING) for item in self._items.values()
                )
            if not busy:
                QApplication.processEvents()
                return True
            QApplication.processEvents()
            time.sleep(0.05)
        QApplication.processEvents()
        return False

    # -- 串行 worker：一次只跑一个 ------------------------------------------

    def _ensure_worker_locked(self) -> None:
        if self._worker_alive:
            return
        self._worker_alive = True
        threading.Thread(target=self._worker_loop, name="recast-model3d-queue", daemon=True).start()

    def _worker_loop(self) -> None:
        """后台线程：取下一个排队的行，调 generate_fn，跑完再取下一个。

        N36：整个 generate_fn 调用都持共享 meta 写锁
        （``recast.core.write_lock.META_WRITE_LOCK``，与 GUI 线程
        ``save_meta`` 用的同一把）。真注册发生在调用内部
        （``service.generate_model_3d`` -> ``generate_model_glb`` ->
        ``save_meta`` / ``set_assembly``），队列侧拆不开，只能整段
        持锁；锁是 RLock，调用内部再调 ``save_meta`` 不会死锁。
        代价是生成进行时 GUI 线程的 meta 写要等这一段跑完——读
        meta 不受影响。把持锁收窄到注册尾巴需要改 generate3d.py /
        service.py，不在本波文件边界内，见报告后续项。
        """
        while True:
            with self._lock:
                if not self._pending:
                    self._worker_alive = False
                    return
                key = self._pending.popleft()
                item = self._items[key]
                item.status = ST_RUNNING
                item.stage = STAGE_RUNNING
                item.error = ""
            self._signals.item_changed.emit(key)
            try:
                with meta_write_lock():
                    result = self._generate_fn(item.character, item.image_ref)
            except Exception as exc:  # noqa: BLE001 - 失败要写进行，不抛回线程
                with self._lock:
                    item.status = ST_FAILED
                    item.stage = "生成失败"
                    item.error = str(exc) or exc.__class__.__name__
                    item.result = None
                    error = item.error
                self._signals.item_changed.emit(key)
                self._signals.item_finished.emit(key, ST_FAILED, error)
                continue
            with self._lock:
                item.status = ST_DONE
                path = (result or {}).get("path") if isinstance(result, dict) else None
                item.stage = f"已完成：{path}" if path else "已完成"
                item.error = ""
                item.result = result if isinstance(result, dict) else {"result": result}
            self._signals.item_changed.emit(key)
            self._signals.item_finished.emit(key, ST_DONE, "")

    # -- 界面 -----------------------------------------------------------------

    def refresh(self) -> None:
        """按当前队列状态重建表格（GUI 线程调用；信号槽也调它）。"""
        with self._lock:
            items = [self._items[key] for key in self._order]
        counts = {ST_QUEUED: 0, ST_RUNNING: 0, ST_FAILED: 0, ST_DONE: 0}
        for item in items:
            counts[item.status] = counts.get(item.status, 0) + 1
        total = len(items)
        self.summary_label.setText(
            f"共 {total} 个：排队 {counts[ST_QUEUED]} · 生成中 {counts[ST_RUNNING]} · "
            f"完成 {counts[ST_DONE]} · 失败 {counts[ST_FAILED]}（一次只跑一个）"
        )
        self.empty_label.setVisible(total == 0)
        self.table.setRowCount(total)
        for row, item in enumerate(items):
            self.table.setItem(row, 0, QTableWidgetItem(item.character))
            self.table.setItem(row, 1, QTableWidgetItem(item.status))
            self.table.setItem(row, 2, QTableWidgetItem(item.stage))
            self.table.setItem(row, 3, QTableWidgetItem(item.error))
            if item.status == ST_FAILED:
                btn = QPushButton("重试")
                btn.setToolTip("把这个人物重新排进 3D 生成队列")
                btn.clicked.connect(lambda _checked=False, c=item.character: self.retry(c))
                self.table.setCellWidget(row, 4, btn)
            else:
                self.table.setCellWidget(row, 4, None)
