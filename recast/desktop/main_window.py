"""Primary PySide6 workbench: drama queue, stages, live logs, character library, settings."""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from recast import __version__
from recast.core.constants import (
    ALL_STATES,
    ENG_IDLE,
    ENG_MISSING,
    ENG_RUNNING,
    PRODUCT_NAME,
    STAGE_GROUPS,
    ST_DONE,
    ST_FAILED,
    ST_NEED_ASSETS,
    ST_PENDING,
    ST_REVIEW,
    ST_RUNNING,
)
from recast.core.service import RecastService
from recast.desktop.first_run import maybe_first_run
from recast.desktop.library_window import LibraryWindow
from recast.desktop.settings_dialog import SettingsDialog
from recast.desktop.template_editor import TemplateEditor
from recast.desktop.cost_window import CostWindow
from recast.desktop.review_window import ReviewWindow
from recast.desktop.generate_dialog import GenerateDialog
from recast.desktop.prop_window import PropWindow

COLS = ["选", "原名", "新名称", "集数", "状态", "进度", "成片", "创建时间"]


class _SnapshotSignal(QObject):
    ready = Signal(object)


class MainWindow(QMainWindow):
    def __init__(self, service: RecastService) -> None:
        super().__init__()
        self.service = service
        self.setWindowTitle(f"{PRODUCT_NAME}  V{__version__}")
        self.resize(1280, 800)
        self._library: LibraryWindow | None = None
        self._templates: TemplateEditor | None = None
        self._cost: CostWindow | None = None
        self._review: ReviewWindow | None = None
        self._props: PropWindow | None = None
        self._log_cursor = 0
        self._building = False

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_top())
        layout.addWidget(self._build_drama_bar())
        layout.addWidget(self._build_table(), 3)

        bottom = QSplitter()
        bottom.setOrientation(Qt.Orientation.Horizontal)
        bottom.addWidget(self._build_engine_panel())
        bottom.addWidget(self._build_log_panel())
        bottom.setStretchFactor(0, 2)
        bottom.setStretchFactor(1, 3)
        layout.addWidget(bottom, 2)
        layout.addWidget(self._build_stage_bar())

        self._snap_signal = _SnapshotSignal()
        self._snap_signal.ready.connect(self._apply_snapshot)
        self._snap_busy = False
        self._snap_again = False
        self._snap_full = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(400)
        self.refresh(full=True)
        QTimer.singleShot(200, lambda: maybe_first_run(self.service, self))

    # ----- builders -----

    def _build_top(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topBar")
        row = QHBoxLayout(bar)
        name = QLabel(PRODUCT_NAME)
        name.setObjectName("productName")
        ver = QLabel(f"V{__version__}  本地短剧再铸工作台")
        ver.setObjectName("versionLabel")
        self.engine_label = QLabel()
        self.hw_label = QLabel()
        row.addWidget(name)
        row.addWidget(ver)
        row.addStretch(1)
        row.addWidget(QLabel("本地引擎"))
        row.addWidget(self.engine_label)
        row.addSpacing(16)
        row.addWidget(QLabel("硬件档"))
        row.addWidget(self.hw_label)
        return bar

    def _build_drama_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("dramaBar")
        row = QHBoxLayout(bar)
        row.addWidget(QLabel("作品目录"))
        self.root_edit = QLineEdit(self.service.settings.works_root)
        self.root_edit.setPlaceholderText("选择本地剧目根目录，然后扫描")
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse_root)
        scan = QPushButton("扫描")
        scan.setObjectName("primary")
        scan.clicked.connect(self._scan)
        delete = QPushButton("删除所选剧")
        delete.clicked.connect(self._delete_selected)
        open_out = QPushButton("打开成片目录")
        open_out.clicked.connect(self._open_output)
        self.filter = QComboBox()
        self.filter.addItem("全部", "全部")
        for st in ALL_STATES:
            self.filter.addItem(st, st)
        self.filter.currentIndexChanged.connect(lambda *_: self.refresh(full=True))
        row.addWidget(self.root_edit, 1)
        row.addWidget(browse)
        row.addWidget(scan)
        row.addWidget(delete)
        row.addWidget(open_out)
        row.addWidget(QLabel("状态筛选"))
        row.addWidget(self.filter)
        return bar

    def _build_table(self) -> QWidget:
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(3, 50)
        self.table.setColumnWidth(4, 100)
        self.table.setColumnWidth(6, 70)
        self.table.setColumnWidth(7, 150)
        self.table.itemChanged.connect(self._on_item_changed)
        return self.table

    def _build_engine_panel(self) -> QWidget:
        box = QFrame()
        box.setObjectName("bottomBar")
        col = QVBoxLayout(box)
        col.addWidget(QLabel("模型 / 引擎检测"))
        self.detect_label = QLabel()
        self.detect_label.setWordWrap(True)
        col.addWidget(self.detect_label)
        col.addWidget(QLabel("总进度"))
        self.overall = QProgressBar()
        self.overall.setRange(0, 1000)
        col.addWidget(self.overall)
        self.overall_text = QLabel("空闲")
        col.addWidget(self.overall_text)
        recheck = QPushButton("重新检查资料")
        recheck.clicked.connect(self._recheck)
        col.addWidget(recheck)
        self.cloud_label = QLabel()
        self.cloud_label.setStyleSheet("color:#8b90a0;")
        self.cloud_label.setWordWrap(True)
        col.addWidget(self.cloud_label)
        col.addStretch(1)
        return box

    def _build_log_panel(self) -> QWidget:
        box = QFrame()
        box.setObjectName("bottomBar")
        col = QVBoxLayout(box)
        head = QHBoxLayout()
        head.addWidget(QLabel("实时日志"))
        self.log_task = QComboBox()
        self.log_task.addItem("全部任务", "")
        self.log_level = QComboBox()
        self.log_level.addItem("全部级别", "")
        for lv in ("INFO", "WARNING", "ERROR"):
            self.log_level.addItem(lv, lv)
        head.addWidget(self.log_task)
        head.addWidget(self.log_level)
        head.addStretch(1)
        col.addLayout(head)
        self.log_list = QListWidget()
        self.log_list.itemDoubleClicked.connect(self._log_detail)
        col.addWidget(self.log_list, 1)
        return box

    def _build_stage_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("stageBar")
        col = QVBoxLayout(bar)
        self.stage_buttons: dict[str, QPushButton] = {}
        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        for i, (sid, label, _) in enumerate(STAGE_GROUPS):
            btn = QPushButton(label)
            btn.setObjectName("stage")
            btn.setCheckable(True)
            btn.setChecked(True)
            self.stage_buttons[sid] = btn
            (row1 if i < 5 else row2).addWidget(btn)
        self.all_stages = QCheckBox("全部阶段按序执行")
        self.all_stages.setChecked(True)
        self.all_stages.toggled.connect(self._toggle_all_stages)
        row2.addWidget(self.all_stages)
        self.upload_stage = QCheckBox("上传（可选，默认关闭）")
        self.upload_stage.setChecked(False)
        row2.addWidget(self.upload_stage)
        row1.addStretch(1)
        row2.addStretch(1)
        col.addLayout(row1)
        col.addLayout(row2)
        actions = QHBoxLayout()
        start = QPushButton("开始执行")
        start.setObjectName("primary")
        start.clicked.connect(self._start)
        stop = QPushButton("停止当前任务")
        stop.setObjectName("danger")
        stop.clicked.connect(self._stop)
        clear = QPushButton("清除已完成")
        clear.clicked.connect(self._clear_completed)
        library = QPushButton("人物库")
        library.clicked.connect(self._open_library)
        settings = QPushButton("设置")
        settings.clicked.connect(self._open_settings)
        tpl = QPushButton("模板编辑器")
        tpl.clicked.connect(self._open_templates)
        gen = QPushButton("按剧本生成新集…")
        gen.clicked.connect(self._open_generate)
        cost = QPushButton("成本表")
        cost.clicked.connect(self._open_cost)
        review = QPushButton("质量终检")
        review.clicked.connect(self._open_review)
        props = QPushButton("3D道具（试点）")
        props.clicked.connect(self._open_props)
        for w in (start, stop, clear, library, tpl, gen, cost, review, props, settings):
            actions.addWidget(w)
        actions.addStretch(1)
        col.addLayout(actions)
        return bar

    # ----- data -----

    def _selected_stages(self) -> list[str]:
        if self.all_stages.isChecked():
            stages = [sid for sid, _, _ in STAGE_GROUPS]
        else:
            stages = [sid for sid, btn in self.stage_buttons.items() if btn.isChecked()]
        if getattr(self, "upload_stage", None) is not None and self.upload_stage.isChecked():
            if "upload" not in stages:
                stages.append("upload")
        return stages

    def _toggle_all_stages(self, checked: bool) -> None:
        for btn in self.stage_buttons.values():
            btn.setChecked(checked)

    def _drama_ids_checked(self) -> list[str]:
        ids: list[str] = []
        for row in range(self.table.rowCount()):
            box = self.table.cellWidget(row, 0)
            if isinstance(box, QCheckBox) and box.isChecked():
                item = self.table.item(row, 1)
                if item:
                    ids.append(item.data(Qt.ItemDataRole.UserRole))
        return ids

    def _current_drama_id(self) -> str | None:
        ids = self._drama_ids_checked()
        if ids:
            return ids[0]
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 1)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _scene_bit(self, engine: dict, scene_run: dict) -> str:
        scene = engine.get("scene") or {}
        threshold = scene_run.get("threshold", scene.get("threshold", 27))
        min_shot = scene_run.get("min_shot_sec", scene.get("min_shot_sec", 0.4))
        used = scene_run.get("backend_label") or scene_run.get("backend")
        if used:
            who = str(used)
            prefix = "本次切镜"
        elif scene.get("installed") == "pyscenedetect" or (engine.get("modules") or {}).get("scenedetect"):
            who = "PySceneDetect"
            prefix = "切镜"
        else:
            who = "FFmpeg 回退"
            prefix = "切镜"
        try:
            number = float(threshold)
            thr = str(int(number)) if number == int(number) else f"{number:.1f}"
        except (TypeError, ValueError):
            thr = str(threshold)
        try:
            shortest = f"{float(min_shot):.2f}"
        except (TypeError, ValueError):
            shortest = str(min_shot)
        return f"{prefix}：{who}（阈值 {thr}，最短 {shortest}s）"

    def refresh(self, full: bool = False) -> None:
        """Queue a snapshot on a worker thread. The 400ms timer must not block the UI."""
        if full:
            self._snap_full = True
        if self._snap_busy:
            self._snap_again = True
            return
        self._snap_busy = True
        want_full = self._snap_full
        self._snap_full = False
        status_filter = self.filter.currentData()

        def work() -> None:
            try:
                snap = self.service.snapshot(status_filter=status_filter)
                snap = dict(snap)
                snap["_full"] = want_full
            except Exception as exc:
                snap = {"_error": str(exc), "_full": want_full}
            self._snap_signal.ready.emit(snap)

        threading.Thread(target=work, name="recast-ui-snapshot", daemon=True).start()

    def _apply_snapshot(self, snap: dict) -> None:
        self._snap_busy = False
        if not snap.get("_error"):
            self._paint(snap, full=bool(snap.get("_full")))
        if self._snap_again or self._snap_full:
            self._snap_again = False
            self.refresh(full=self._snap_full)
            self._snap_full = False

    def _paint(self, snap: dict, full: bool = False) -> None:
        engine = snap.get("engine") or {}
        state = engine.get("state") or ENG_IDLE
        self.engine_label.setText(state)
        self.engine_label.setObjectName(
            "engineRun" if state == ENG_RUNNING else "engineMiss" if state == ENG_MISSING else "engineIdle"
        )
        self.engine_label.style().unpolish(self.engine_label)
        self.engine_label.style().polish(self.engine_label)
        hw = engine.get("hardware_label") or ""
        gpu = engine.get("gpu_name")
        vram = engine.get("vram_gb")
        extra = f"  ·  {gpu}" if gpu else ""
        if vram:
            extra += f"  {vram:.1f} GB"
        self.hw_label.setText(hw + extra)

        mods = engine.get("modules") or {}
        ffm = engine.get("ffmpeg") or {}
        models = engine.get("models") or []
        bits = [
            f"ffmpeg：{'OK' if ffm.get('ok') else '缺失'}",
            f"PyTorch：{'OK' if mods.get('torch') else '缺失'}",
            f"Whisper：{'已安装' if mods.get('faster_whisper') else '未安装'}",
            f"InsightFace：{'已安装' if mods.get('insightface') else '未安装'}",
            self._scene_bit(engine, snap.get("scene_run") or {}),
            f"OCR：{'RapidOCR' if mods.get('rapidocr') else '未安装（将只用 ASR）'}",
        ]
        missing_w = [m["label"] for m in models if not m.get("present") and m.get("key") == "inswapper_128"]
        if missing_w:
            bits.append("人物替换权重：未下载（首次执行时自动获取）")
        else:
            bits.append("人物替换权重：已就绪")
        disk = engine.get("disk_free_gb")
        if disk is not None:
            bits.append(f"磁盘剩余：{disk:.1f} GB")
        self.detect_label.setText("    ".join(bits))

        sched = snap.get("scheduler") or {}
        overall = float(sched.get("overall") or 0)
        self.overall.setValue(int(overall * 1000))
        if sched.get("running"):
            self.overall_text.setText(
                f"{sched.get('run_done') or 0}/{sched.get('run_total') or 0}  "
                f"{sched.get('current_progress') or ''}"
            )
        else:
            self.overall_text.setText("空闲")

        if full or self.table.rowCount() != len(snap.get("dramas") or []):
            self._rebuild_table(snap.get("dramas") or [])
        else:
            self._patch_table(snap.get("dramas") or [])

        self._refresh_logs()
        # keep works root field in sync if empty
        if not self.root_edit.text() and self.service.settings.works_root:
            self.root_edit.setText(self.service.settings.works_root)
        s = self.service.settings
        volc = "已配置" if s.cloud_volcengine.configured else "未配置"
        deep = "已配置" if s.cloud_deepseek.configured else "未配置"
        oai = "已配置" if getattr(s, "cloud_openai", None) and s.cloud_openai.configured else "未配置"
        master = "开" if s.cloud_enabled else "关（默认）"
        up = "开" if s.auto_upload else "关（默认）"
        comfy = "已连接" if (engine.get("comfyui") or {}).get("ok") else "未连接（将用分镜预览成片）"
        cap = engine.get("capability") or {}
        basic_note = cap.get("message") or ""
        self.cloud_label.setText(
            f"云通道总开关：{master}    火山 {volc}    DeepSeek {deep}    OpenAI兼容 {oai}    自动上传：{up}    ComfyUI：{comfy}"
            + (f"\n{basic_note}" if basic_note else "")
        )

    def _rebuild_table(self, dramas: list[dict]) -> None:
        self._building = True
        selected = set(self._drama_ids_checked())
        self.table.setRowCount(0)
        for d in dramas:
            self._append_row(d, selected)
        self._building = False
        self._rebuild_log_task_filter(dramas)

    def _append_row(self, d: dict, selected: set[str]) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        box = QCheckBox()
        # default checked unless user previously unchecked this id in this session
        # persist via store.selected
        want = d["id"] in selected if selected else bool(d.get("selected", 1))
        box.setChecked(bool(want))
        box.stateChanged.connect(lambda st, i=d["id"]: self.service.set_selected(i, st == Qt.CheckState.Checked or st == 2))
        self.table.setCellWidget(row, 0, box)

        origin = QTableWidgetItem(d.get("origin_name") or "")
        origin.setData(Qt.ItemDataRole.UserRole, d["id"])
        origin.setFlags(origin.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 1, origin)

        new_name = QTableWidgetItem(d.get("new_name") or d.get("origin_name") or "")
        self.table.setItem(row, 2, new_name)

        count = QTableWidgetItem(str(d.get("episode_count") or 0))
        count.setFlags(count.flags() & ~Qt.ItemFlag.ItemIsEditable)
        count.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 3, count)

        status = QTableWidgetItem(d.get("status") or "")
        status.setFlags(status.flags() & ~Qt.ItemFlag.ItemIsEditable)
        status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._color_status(status, d.get("status") or "")
        self.table.setItem(row, 4, status)

        prog = QTableWidgetItem(d.get("progress") or "")
        prog.setFlags(prog.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 5, prog)

        open_btn = QPushButton("打开")
        open_btn.setEnabled(bool(d.get("output_dir")))
        open_btn.clicked.connect(lambda *_a, i=d["id"]: self._open_output_id(i))
        self.table.setCellWidget(row, 6, open_btn)

        created = QTableWidgetItem(d.get("created_at") or "")
        created.setFlags(created.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 7, created)

    def _patch_table(self, dramas: list[dict]) -> None:
        by_id = {d["id"]: d for d in dramas}
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            if not item:
                continue
            d = by_id.get(item.data(Qt.ItemDataRole.UserRole))
            if not d:
                continue
            self.table.item(row, 3).setText(str(d.get("episode_count") or 0))
            st = self.table.item(row, 4)
            st.setText(d.get("status") or "")
            self._color_status(st, d.get("status") or "")
            self.table.item(row, 5).setText(d.get("progress") or "")
            btn = self.table.cellWidget(row, 6)
            if isinstance(btn, QPushButton):
                btn.setEnabled(bool(d.get("output_dir")))

    def _color_status(self, item: QTableWidgetItem, status: str) -> None:
        colors = {
            ST_PENDING: "#c8cde0",
            ST_RUNNING: "#ffd56a",
            ST_NEED_ASSETS: "#ffb347",
            ST_REVIEW: "#d5a6ff",
            ST_DONE: "#7ddea0",
            ST_FAILED: "#ff7b7b",
        }
        item.setForeground(QColor(colors.get(status, "#e8eaef")))

    def _rebuild_log_task_filter(self, dramas: list[dict]) -> None:
        current = self.log_task.currentData()
        self.log_task.blockSignals(True)
        self.log_task.clear()
        self.log_task.addItem("全部任务", "")
        for d in dramas:
            self.log_task.addItem(d.get("new_name") or d.get("origin_name") or d["id"], d["id"])
        idx = self.log_task.findData(current)
        self.log_task.setCurrentIndex(max(0, idx))
        self.log_task.blockSignals(False)

    def _refresh_logs(self) -> None:
        # Filter lives in memory; map drama filter to any of its task ids
        drama_id = self.log_task.currentData()
        task_ids = None
        if drama_id:
            task_ids = {t["id"] for t in self.service.store.tasks_for_drama(drama_id)}
        level = self.log_level.currentData() or None
        rows = self.service.logs.dump()
        self.log_list.clear()
        for row in rows[-400:]:
            if task_ids is not None and row.get("task_id") not in task_ids:
                continue
            if level and row.get("level") != level:
                continue
            text = f"{row['ts']}  [{row['level']}]  {row['message']}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, row)
            if row.get("level") == "ERROR":
                item.setForeground(QColor("#ff7b7b"))
            elif row.get("level") == "WARNING":
                item.setForeground(QColor("#ffb347"))
            self.log_list.addItem(item)
        self.log_list.scrollToBottom()

    def _log_detail(self, item: QListWidgetItem) -> None:
        row = item.data(Qt.ItemDataRole.UserRole) or {}
        QMessageBox.information(self, "日志详情", f"{row.get('ts')}  [{row.get('level')}]\n\n{row.get('message')}")

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._building or item.column() != 2:
            return
        origin = self.table.item(item.row(), 1)
        if not origin:
            return
        self.service.rename_drama(origin.data(Qt.ItemDataRole.UserRole), item.text())

    # ----- actions -----

    def _browse_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择作品根目录", self.root_edit.text() or str(Path.home()))
        if path:
            self.root_edit.setText(path)

    def _scan(self) -> None:
        path = self.root_edit.text().strip()
        if not path:
            path = QFileDialog.getExistingDirectory(self, "选择作品根目录", str(Path.home()))
            if not path:
                return
            self.root_edit.setText(path)
        try:
            self.service.scan(path)
        except Exception as exc:
            QMessageBox.warning(self, "扫描失败", str(exc))
            return
        self.refresh(full=True)

    def _delete_selected(self) -> None:
        ids = self._drama_ids_checked()
        if not ids:
            QMessageBox.information(self, "删除所选剧", "请先勾选要删除的剧目。")
            return
        if QMessageBox.question(self, "删除所选剧", "只会从队列里移除勾选的剧目，不会删除磁盘上的成片。确定吗？") != QMessageBox.StandardButton.Yes:
            return
        self.service.delete_dramas(ids)
        self.refresh(full=True)

    def _open_output(self) -> None:
        did = self._current_drama_id()
        if not did:
            QMessageBox.information(self, "打开成片目录", "请先勾选或选中一部剧。")
            return
        self._open_output_id(did)

    def _open_output_id(self, drama_id: str) -> None:
        try:
            self.service.open_output(drama_id)
        except Exception as exc:
            QMessageBox.warning(self, "无法打开成片目录", str(exc))

    def _recheck(self) -> None:
        ids = self._drama_ids_checked() or None
        self.service.recheck_materials(ids)
        self.refresh(full=True)

    def _start(self) -> None:
        ids = self._drama_ids_checked()
        stages = self._selected_stages()
        if not stages:
            QMessageBox.information(self, "开始执行", "请至少勾选一个底部阶段，或打开「全部阶段按序执行」。")
            return
        try:
            self.service.start(ids, stages)
        except Exception as exc:
            QMessageBox.warning(self, "无法开始", str(exc))
            return
        self.refresh(full=True)

    def _stop(self) -> None:
        self.service.stop()
        self.refresh()

    def _clear_completed(self) -> None:
        if QMessageBox.question(
            self,
            "清除已完成",
            "只会从队列中移除状态为「已完成」的剧目，不会删除磁盘上的成片文件。确定清除吗？",
        ) != QMessageBox.StandardButton.Yes:
            return
        self.service.clear_completed()
        self.refresh(full=True)

    def _open_library(self) -> None:
        if self._library is None:
            self._library = LibraryWindow(self.service, self)
        self._library.reload()
        self._library.show()
        self._library.raise_()

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.service, self)
        if dlg.exec():
            dlg.apply()
            self.refresh(full=True)

    def _open_templates(self) -> None:
        self._templates = TemplateEditor(self.service, self, drama_id=self._current_drama_id())
        self._templates.show()
        self._templates.raise_()

    def _open_generate(self) -> None:
        if self.service.settings.is_basic_tier():
            QMessageBox.information(
                self,
                "基础档提示",
                "当前是基础档（无独立 GPU）。仍可创建按剧本生成任务，但会使用「分镜预览成片」，不会走 ComfyUI 重生成。",
            )
        dlg = GenerateDialog(self.service, self)
        if dlg.exec():
            if dlg.apply():
                self.refresh(full=True)

    def _open_cost(self) -> None:
        self._cost = CostWindow(self.service, self, drama_id=self._current_drama_id())
        self._cost.show()
        self._cost.raise_()

    def _open_review(self) -> None:
        self._review = ReviewWindow(self.service, self, drama_id=self._current_drama_id())
        self._review.show()
        self._review.raise_()

    def _open_props(self) -> None:
        if self._props is None:
            self._props = PropWindow(self.service, self)
        self._props.show()
        self._props.raise_()
