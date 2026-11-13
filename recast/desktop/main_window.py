"""Primary PySide6 workbench: drama queue, stages, live logs, character library, settings."""

from __future__ import annotations

import hashlib
import re
import tempfile
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
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
    QSizePolicy,
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
    STAGE_LABELS,
    ST_DONE,
    ST_FAILED,
    ST_NEED_ASSETS,
    ST_PENDING,
    ST_REVIEW,
    ST_RUNNING,
    VIDEO_EXTS,
)
from recast.core.audio_segments import (
    character_voice_samples,
    collect_diar_segments,
    reassign_segment,
    update_voice_samples,
)
from recast.core.entry_shared import ENTRY_STAGES, check_replace_auth
from recast.core.failed_tasks import CHECKPOINT_STAGE, collect_failed_entries, last_checkpoint_stage
from recast.core.precheck import precheck_dot, probe_precheck_payload, summarize_precheck
from recast.core.review_progress import review_progress_stats
from recast.core.scanner import scan_root
from recast.core.service import RecastService
from recast.desktop.audio_panel import (
    apply_audio_result,
    audio_status_text,
    build_audio_panel,
    find_subtitle_target,
    rebuild_character_rows,
    refresh_diar_table,
)
from recast.desktop.auth_gate_dialog import auth_gate_ok
from recast.desktop.failed_cards import build_failed_group, refresh_failed_cards
from recast.desktop.precheck_card import build_precheck_panel, paint_precheck_dot
from recast.desktop.lipsync_warn_dialog import ask_lipsync_on_gui_thread
from recast.media import open_capture
from recast.desktop.first_run import maybe_first_run
from recast.desktop.library_window import LibraryWindow
from recast.desktop.settings_dialog import SettingsDialog
from recast.desktop.template_editor import TemplateEditor
from recast.desktop.cost_window import CostWindow
from recast.desktop.review_window import ReviewWindow
from recast.desktop.generate_dialog import GenerateDialog
from recast.desktop.prop_window import PropWindow

COLS = ["选", "原名", "新名称", "集数", "状态", "进度", "成片", "创建时间"]

# 统一编号（PM 第 2 轮）：全产品只留引导 ①–⑤ 这一套编号。9 个管线阶段
# （constants.STAGE_LABELS 里带 "01 …" 编号）退到日志/高级视图，主界面
# 显示时去掉数字编号，只留名字。runs 表里的步骤文案是 "01 A / 02 B"
# 这样的拼接，同样逐段去编号。
_STAGE_NUM_PREFIX = re.compile(r"^\d+\s*")


def _strip_stage_number(label: str) -> str:
    return _STAGE_NUM_PREFIX.sub("", label or "").strip()


def _strip_run_step_label(step: str) -> str:
    return " / ".join(_strip_stage_number(part) for part in (step or "").split(" / "))


# 入口–阶段映射的唯一来源在 recast.core.entry_shared（ENTRY_STAGES，
# R6 第一步，P3）：这里不再留拷贝，queue_panel 也已改成 import
#（N35，P3 第 3 波）；任务历史重试里没有这份拷贝。
# 检查点子步骤 → 所属阶段的唯一来源在 recast.core.failed_tasks
#（CHECKPOINT_STAGE，R6 第 2 步，P4）；这里只留旧名兼容。
_CHECKPOINT_STAGE: dict[str, str] = CHECKPOINT_STAGE


class _SnapshotSignal(QObject):
    ready = Signal(object)


class MainWindow(QMainWindow):
    def __init__(self, service: RecastService) -> None:
        super().__init__()
        self.service = service
        self.setWindowTitle(f"{PRODUCT_NAME}  V{__version__}")
        self.resize(1280, 800)
        self._library: LibraryWindow | None = None
        self._queue_panel = None  # 队列面板对话框（P2 接线，见 _open_queue）
        self._templates: TemplateEditor | None = None
        self._cost: CostWindow | None = None
        self._review: ReviewWindow | None = None
        self._props: PropWindow | None = None
        self._log_cursor = 0
        self._building = False
        self._log_expanded = False  # 日志面板：空闲时收窄，有输出时展开
        self._preview_drama: str | None = None  # 预览区当前是哪部剧
        self._oneclick: dict | None = None  # 一键成片进行中：{drama_id, phase, entry, ...}
        self._precheck_blocked: dict[str, str] = {}  # 剧目 id → 挡路的预检原因
        self._precheck: dict[str, dict] = {}  # 剧目 id → 预检卡结果
        self._current_guide: list[dict] = []  # 引导 ①–⑤（含第 ⑤ 步声音与台词）
        self._review_was_nonempty = False  # 审核队列上次刷新是否非空（自动跳转只在变空那一下提示）
        self._review_confirmed_when_nonempty = 0  # N21：队列上次非空时的已确认人数——清空时没涨，说明最后一次是丢弃不是确认，就不弹自动跳转提示
        self._review_marked_when_nonempty = 0  # N21 残留（P4）：队列上次非空时打上“已审核”标记的人脸样本数——低分样本确认（review_assign / 认领）只打这个标记、不涨已确认人数，只看已确认人数会把这种清空误判成丢弃
        self._review_confirms_since_nonempty = 0  # N21 残留（P4）：自队列上次变非空以来观测到的确认动作次数（已确认人数增量 + “已审核”标记增量）；队列清空时 >0 即判定为确认清空
        self._review_stats_cache: dict | None = None
        self._review_detail_cache: str | None = None
        self._diar_rows: list[dict] = []  # 分轨表当前行：{start, end, speaker, owner_id, path}
        self._audio_char_btns: dict[str, dict[str, QPushButton]] = {}

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_top())
        layout.addWidget(self._build_drama_bar())

        # 剪辑软件式三栏：左步骤引导 / 中作品表 + 预览 / 右引擎 + 日志。
        middle = QSplitter(Qt.Orientation.Horizontal)
        middle.setObjectName("mainSplitter")
        middle.addWidget(self._build_left_rail())
        middle.addWidget(self._build_center())
        middle.addWidget(self._build_right_rail())
        middle.setStretchFactor(0, 0)
        middle.setStretchFactor(1, 1)
        middle.setStretchFactor(2, 0)
        middle.setSizes([250, 680, 350])
        layout.addWidget(middle, 1)
        layout.addWidget(self._build_action_bar())

        self._snap_signal = _SnapshotSignal()
        self._snap_signal.ready.connect(self._apply_snapshot)
        self._rerender_signal = _SnapshotSignal()
        self._rerender_signal.ready.connect(self._apply_rerender_result)
        self._audio_signal = _SnapshotSignal()
        self._audio_signal.ready.connect(self._apply_audio_result)
        self._precheck_signal = _SnapshotSignal()
        self._precheck_signal.ready.connect(self._apply_precheck_result)
        self._preview_signal = _SnapshotSignal()
        self._preview_signal.ready.connect(self._apply_preview)
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
        """顶栏：左侧产品名 + 版本，右侧紧凑两行（引擎状态一行、硬件档一行）。

        P5/B3：原来单行并排在 1280x720 下会被截断；改成两行后每行
        的值标签都换行显示（wordWrap），完整文本放 tooltip；顶栏
        高度随之从一行变成两行（约 +20px），主窗口其余区域相应收窄。
        """
        bar = QFrame()
        bar.setObjectName("topBar")
        row = QHBoxLayout(bar)
        name = QLabel(PRODUCT_NAME)
        name.setObjectName("productName")
        ver = QLabel(f"V{__version__}  本地短剧再铸工作台")
        ver.setObjectName("versionLabel")
        row.addWidget(name)
        row.addWidget(ver)
        row.addStretch(1)
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(2)
        eng_row = QHBoxLayout()
        eng_row.setContentsMargins(0, 0, 0, 0)
        eng_row.addWidget(QLabel("本地引擎"))
        self.engine_label = QLabel()
        self.engine_label.setWordWrap(True)
        self.engine_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        eng_row.addWidget(self.engine_label, 1)
        right.addLayout(eng_row)
        hw_row = QHBoxLayout()
        hw_row.setContentsMargins(0, 0, 0, 0)
        hw_row.addWidget(QLabel("硬件档"))
        self.hw_label = QLabel()
        self.hw_label.setWordWrap(True)
        self.hw_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        hw_row.addWidget(self.hw_label, 1)
        right.addLayout(hw_row)
        row.addLayout(right, 1)
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
        import_single = QPushButton("导入单个视频…")
        import_single.setObjectName("importSingle")
        import_single.setToolTip(
            "选一个视频文件直接导入：自动建剧目 + 任务，不必先把文件放进作品目录结构；"
            "导入后同样进作品表（走现有扫描 / 建剧目流程）。"
        )
        self.import_single_btn = import_single
        import_single.clicked.connect(self._import_single_video)
        scan = QPushButton("扫描")
        # P5/B3：主窗口唯一 primary 是 [一键成片]；[扫描] 降级为 secondary。
        scan.setObjectName("secondary")
        scan.setMinimumWidth(96)
        scan.setMinimumHeight(32)
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
        row.addWidget(import_single)
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
        # P5/B3：第 6 列「成片」原来固定 70px，「打开」按钮只显示「…」；
        # 改成按内容自适应，按钮本身另有最小宽（见 _append_row），
        # 保证按钮文字完整显示。固定宽的三列（3 集数 / 4 状态 /
        # 7 创建时间）已核对过表头与单元格内容宽度，均不截断。
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(3, 50)
        self.table.setColumnWidth(4, 100)
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
        # P5/B3：纵向按换行后的实际高度给空间（MinimumExpanding），
        # 否则右栏竖直 splitter 只给 260px 时布局会把它压到单行高、
        # 换行后的后半截文字被裁掉。
        self.detect_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        col.addWidget(self.detect_label)
        col.addWidget(QLabel("总进度"))
        self.overall = QProgressBar()
        self.overall.setRange(0, 1000)
        col.addWidget(self.overall)
        self.overall_text = QLabel("空闲")
        self.overall_text.setWordWrap(True)  # 进度文字换行，不再截断成「自动上…」
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

    def _build_left_rail(self) -> QWidget:
        """左栏：分步引导（编号步骤条 + 下一步）+ 执行阶段竖排清单。"""
        rail = QFrame()
        rail.setObjectName("leftRail")
        col = QVBoxLayout(rail)

        guide_group = QGroupBox("分步引导")
        gcol = QVBoxLayout(guide_group)
        self.guide_list = QListWidget()
        self.guide_list.setMaximumHeight(150)
        self.guide_list.itemClicked.connect(self._on_guide_jump)
        gcol.addWidget(self.guide_list)
        self.next_btn = QPushButton("下一步")
        # P5/B3：主窗口唯一 primary 是 [一键成片]；[下一步] 降级为 secondary。
        self.next_btn.setObjectName("secondary")
        self.next_btn.setMinimumWidth(96)
        self.next_btn.setMinimumHeight(32)
        self.next_btn.clicked.connect(self._on_next_step)
        gcol.addWidget(self.next_btn)
        self.guide_hint = QLabel("先选一部剧，再点下一步。")
        self.guide_hint.setWordWrap(True)
        self.guide_hint.setStyleSheet("color:#8b90a0;")
        gcol.addWidget(self.guide_hint)
        # 审核进度条（PM 第 2 轮 3.1）：已确认 / 待审核 / 低分一眼可见。
        self.review_progress_label = QLabel("审核进度：还没有人物。")
        self.review_progress_label.setWordWrap(True)
        self.review_progress_label.setStyleSheet("color:#8b90a0;")
        gcol.addWidget(self.review_progress_label)
        self.review_progress = QProgressBar()
        self.review_progress.setRange(0, 100)
        self.review_progress.setVisible(False)
        gcol.addWidget(self.review_progress)
        self.oneclick_label = QLabel("")
        self.oneclick_label.setWordWrap(True)
        self.oneclick_label.setStyleSheet("color:#7ddea0;")
        gcol.addWidget(self.oneclick_label)
        col.addWidget(guide_group)

        stage_group = QGroupBox("执行阶段")
        scol = QVBoxLayout(stage_group)
        self.stage_buttons: dict[str, QCheckBox] = {}
        for sid, label, _ in STAGE_GROUPS:
            cb = QCheckBox(_strip_stage_number(label))
            cb.setChecked(True)
            self.stage_buttons[sid] = cb
            scol.addWidget(cb)
        self.all_stages = QCheckBox("全部阶段按序执行")
        self.all_stages.setChecked(True)
        self.all_stages.toggled.connect(self._toggle_all_stages)
        scol.addWidget(self.all_stages)
        self.upload_stage = QCheckBox("上传（可选，默认关闭）")
        self.upload_stage.setChecked(False)
        scol.addWidget(self.upload_stage)
        col.addWidget(stage_group)

        failed_group, self.failed_empty_label, self.failed_cards_layout = build_failed_group()
        self.failed_cards: list[QFrame] = []
        self._failed_sig: tuple | None = None
        col.addWidget(failed_group)

        col.addStretch(1)
        return rail

    def _build_center(self) -> QWidget:
        """中栏：作品表（合理高度，不撑黑洞）+ 预览区（海报/首帧 + 进度）。"""
        center = QWidget()
        col = QVBoxLayout(center)
        col.setContentsMargins(0, 0, 0, 0)
        # 导入预检卡（默认隐藏，选中一部剧才显示）。
        self.precheck_panel, self.precheck_label = build_precheck_panel()
        col.addWidget(self.precheck_panel)
        self._build_table()
        self.table.itemSelectionChanged.connect(self._on_table_select)
        col.addWidget(self.table)
        preview_group = QGroupBox("预览")
        pcol = QVBoxLayout(preview_group)
        self.preview_image = QLabel("先选一部剧，再点下一步")
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setMinimumHeight(200)
        self.preview_image.setWordWrap(True)
        self.preview_image.setStyleSheet("border:1px solid #3a3f4d;")
        pcol.addWidget(self.preview_image, 1)
        self.preview_text = QLabel("还没有选中的作品：点一下作品表里的一行，就能看到它的首帧和各集进度。")
        self.preview_text.setWordWrap(True)
        pcol.addWidget(self.preview_text)
        col.addWidget(preview_group, 1)

        # 第 ⑤ 步声音与台词面板（默认隐藏，点左侧第 ⑤ 步才打开）。
        # R6：建组在 recast.desktop.audio_panel.build_audio_panel。
        _audio = build_audio_panel(
            on_rediarize=self._rediarize,
            on_redub=self._redub,
            on_edit_subtitles=self._edit_subtitles,
        )
        self.audio_panel = _audio.group
        self.audio_status = _audio.status
        self.diar_table = _audio.diar_table
        self.audio_char_layout = _audio.char_layout
        self.rediar_btn = _audio.rediar_btn
        self.redub_btn = _audio.redub_btn
        self.edit_subs_btn = _audio.edit_subs_btn
        col.addWidget(self.audio_panel)
        return center

    def _build_right_rail(self) -> QWidget:
        """右栏：引擎检测（文字换行，不截断）+ 日志（空闲收窄，有输出展开）。

        P5/B3：右栏设最小宽 300，1280x720 下引擎面板的 detect_label
        才有足够宽度换行、而不是被中栏挤到只剩几十像素。
        """
        right = QSplitter(Qt.Orientation.Vertical)
        right.setObjectName("rightSplitter")
        right.setMinimumWidth(300)
        right.addWidget(self._build_engine_panel())
        right.addWidget(self._build_log_panel())
        right.setStretchFactor(0, 0)
        right.setStretchFactor(1, 1)
        right.setSizes([260, 120])
        self._right_splitter = right
        return right

    def _build_action_bar(self) -> QWidget:
        """底部操作栏：按用途分组加标题，主操作「一键成片」最显眼。

        两个启动入口主次分明：一键成片是唯一的 primary（全流程自动跑），
        「按勾选阶段执行」是普通按钮（只跑底部勾选的阶段）；「只换脸」
        只跑替换这一步。两条重试语义也分开写：失败卡片是同一任务续跑，
        历史面板的复跑是新建任务全量重跑。
        """
        bar = QFrame()
        bar.setObjectName("actionBar")
        row = QHBoxLayout(bar)

        def _group(title: str) -> QVBoxLayout:
            group = QGroupBox(title)
            layout = QVBoxLayout(group)
            row.addWidget(group)
            return layout

        exec_layout = _group("执行")
        self.oneclick_btn = QPushButton("一键成片")
        self.oneclick_btn.setObjectName("primary")  # P5/B3：主窗口唯一的 primary
        self.oneclick_btn.setMinimumHeight(36)
        self.oneclick_btn.setToolTip(
            "选一部剧，一键跑到成片：先跑前期（导入 / 提取 / 建库），到审核关口停下来等你确认人物，"
            "确认完自动继续替换 / 合成 / 导出。某阶段失败只停在该阶段，原因见左栏「失败任务」，"
            "点[从失败步续跑]继续。只想按勾选的阶段手动跑，点旁边的「按勾选阶段执行」。"
        )
        self.oneclick_btn.clicked.connect(self._one_click)
        self.start_btn = QPushButton("按勾选阶段执行")
        self.start_btn.setObjectName("secondary")  # P5/B3：次按钮，与主按钮同高
        self.start_btn.setMinimumWidth(96)
        self.start_btn.setMinimumHeight(32)
        self.start_btn.setToolTip(
            "按底部勾选的阶段、跑勾选的剧；不跑全流程。想一键跑到成片，点「一键成片」。"
        )
        self.start_btn.clicked.connect(self._start)
        stop = QPushButton("停止当前任务")
        stop.setObjectName("danger")  # P5/B3：危险按钮与次按钮同尺寸，永不做主按钮
        stop.setMinimumWidth(96)
        stop.setMinimumHeight(32)
        stop.clicked.connect(self._stop)
        clear = QPushButton("清除已完成")
        clear.clicked.connect(self._clear_completed)
        for w in (self.oneclick_btn, self.start_btn, stop, clear):
            exec_layout.addWidget(w)

        people_layout = _group("人物")
        self.library_btn = QPushButton("人物库")
        # P5/B3：主窗口唯一 primary 是 [一键成片]；[人物库] 降级为 secondary。
        self.library_btn.setObjectName("secondary")
        self.library_btn.setMinimumWidth(96)
        self.library_btn.setMinimumHeight(32)
        self.library_btn.setToolTip("打开人物库：确认人物、抠像、3D 模型、声音，一步步做完再回来跑成片。")
        self.library_btn.clicked.connect(self._open_library)
        self.reface_btn = QPushButton("只换脸（跳过前面步骤）")
        self.reface_btn.setObjectName("secondary")
        self.reface_btn.setMinimumWidth(96)
        self.reface_btn.setMinimumHeight(32)
        self.reface_btn.setToolTip("只跑替换这一步，参考脸从人物文件夹取。替换前要先完成第 ③ 步人物审核。")
        self.reface_btn.clicked.connect(self._reface)
        self.rerender_btn = QPushButton("身体重渲染（试点）")
        self.rerender_btn.setObjectName("secondary")
        self.rerender_btn.setMinimumWidth(96)
        self.rerender_btn.setMinimumHeight(32)
        self.rerender_btn.setToolTip(
            "试点：≤5 秒单人片段，用 3D 模型（GLB）重渲染身体，原头保留。\n"
            "需本机 Blender 渲染；无 Blender 时只生成姿态 + 重定向 + render-job.json，"
            "需到 Windows/Blender 上跑，不会直接出片。\n"
            "超长片段会直接拒绝；多人镜头不在试点内，请先确认片段里只有一个人。"
        )
        self.rerender_btn.clicked.connect(self._body_rerender)
        for w in (self.library_btn, self.reface_btn, self.rerender_btn):
            people_layout.addWidget(w)

        gen_layout = _group("生成")
        self.gen_btn = QPushButton("按剧本生成新集…")
        self.gen_btn.setObjectName("secondary")
        self.gen_btn.setMinimumWidth(96)
        self.gen_btn.setMinimumHeight(32)
        self.gen_btn.clicked.connect(self._open_generate)
        tpl = QPushButton("模板编辑器")
        tpl.setObjectName("secondary")
        tpl.setMinimumWidth(96)
        tpl.setMinimumHeight(32)
        tpl.clicked.connect(self._open_templates)
        for w in (self.gen_btn, tpl):
            gen_layout.addWidget(w)

        export_layout = _group("导出")
        self.export_jy_btn = QPushButton("导出剪映")
        self.export_jy_btn.setToolTip("只跑剪映工程导出这一步，说话人名字从人物文件夹取。")
        self.export_jy_btn.clicked.connect(self._export_jianying)
        self.shot_enhance_btn = QPushButton("逐镜头增强…")
        self.shot_enhance_btn.setToolTip(
            "给任意镜头开关换后增强（hqdn3d + unsharp），不只是终检标出的镜头；"
            "开关记在分集工作目录的 shots_enhance.json 里。"
            "两条路都支持：导入路逐镜头增强，剧本路在拼接后按标记时段增强。"
        )
        self.shot_enhance_btn.clicked.connect(self._open_shot_enhance)
        cost = QPushButton("成本表")
        cost.clicked.connect(self._open_cost)
        review = QPushButton("质量终检")
        review.clicked.connect(self._open_review)
        for w in (self.export_jy_btn, self.shot_enhance_btn, cost, review):
            export_layout.addWidget(w)

        tools_layout = _group("工具")
        self.queue_btn = QPushButton("队列面板")
        self.queue_btn.setToolTip(
            "打开队列面板：批量跑进行中的每集状态、当前步骤、进度、失败原因，"
            "失败行可直接[从失败步续跑]（同一任务续跑，不新建任务）。"
        )
        self.queue_btn.clicked.connect(self._open_queue)
        self.history_btn = QPushButton("任务历史")
        self.history_btn.setToolTip(
            "打开任务历史：按时间看每次任务的状态、配置快照。历史里的「一键复跑（新建任务）」会新建任务行、"
            "全量重跑；左栏失败卡片的「从失败步续跑」是同一任务续跑，不新建任务。"
        )
        self.history_btn.clicked.connect(self._open_history)
        props = QPushButton("3D道具（试点）")
        props.clicked.connect(self._open_props)
        settings = QPushButton("设置")
        settings.clicked.connect(self._open_settings)
        for w in (self.queue_btn, self.history_btn, props, settings):
            tools_layout.addWidget(w)

        row.addStretch(1)
        return bar

    # ----- 分步引导 -----

    def _audio_done(self) -> bool:
        """第 ⑤ 步是否算完成。判定在 guide.py（_audio_done），这里只委托。"""
        from recast.core.guide import _audio_done

        return _audio_done(self.service)

    def _refresh_guide(self, snap: dict) -> None:
        """步骤条 + 阶段清单的状态全部来自快照里的 guide，不再查库。

        锁住规则：只有第 ③ 步「人物审核确认」处于进行中（= 审核队列非空）
        时，第 ④ 步才锁住——替换/生成两个阶段复选框和「只换脸」「按剧本
        生成新集…」按钮一起灰掉，提示「先完成人物审核」。
        """
        guide = snap.get("guide") or []
        if not guide:
            return
        # 第 ⑤ 步已在 guide.py 的状态机里，不再这里追加。
        self._current_guide = guide
        self.guide_list.blockSignals(True)
        self.guide_list.clear()
        for step in guide:
            prefix = {"done": "✓ ", "active": "▶ ", "locked": "🔒 ", "todo": "○ "}.get(step["state"], "○ ")
            item = QListWidgetItem(f"{prefix}{step['number']} {step['label']}")
            item.setData(Qt.ItemDataRole.UserRole, step["id"])
            if step.get("detail"):
                item.setToolTip(step["detail"])
            self.guide_list.addItem(item)
        self.guide_list.blockSignals(False)
        first_open = next((s for s in guide if s["state"] in ("active", "locked")), None)
        if first_open is None:
            self.next_btn.setText("全部完成 ✓")
            self.next_btn.setEnabled(False)
            self.guide_hint.setText("五步都完成了，可以跑成片了。")
        else:
            self.next_btn.setText(f"下一步：{first_open['number']} {first_open['label']}")
            self.next_btn.setEnabled(True)
            self.guide_hint.setText(first_open.get("detail") or "")
        locked = any(s["id"] == "review" and s["state"] == "active" for s in guide)
        self._guide_locked = locked
        for sid in ("replace", "generate"):
            btn = self.stage_buttons.get(sid)
            if btn is not None:
                btn.setEnabled(not locked)
                btn.setToolTip("先完成人物审核（第 ③ 步）" if locked else "")
        for btn in (self.reface_btn, self.gen_btn):
            btn.setEnabled(not locked)
            btn.setToolTip("先完成人物审核（第 ③ 步）" if locked else "")
        self._paint_stage_states(snap, guide, locked)
        self._refresh_review_progress(guide)

    def _review_stats(self, guide: list[dict]) -> dict:
        """审核进度数：已确认 / 待审核 / 低分。

        400ms 刷新一次，队列扫描不便宜：只有第 ③ 步的详情文字变了
        （待审核数变了）才重算，否则用缓存。队列变空一定会改详情文字，
        所以自动跳转的变空检测不会漏。
        """
        review = next((s for s in guide if s["id"] == "review"), None)
        detail = review.get("detail") if review else ""
        if detail == self._review_detail_cache and self._review_stats_cache is not None:
            return self._review_stats_cache
        self._review_detail_cache = detail
        try:
            queue = self.service.list_review_queue()
        except Exception:
            queue = []
        try:
            characters = self.service.characters()
        except Exception:
            characters = []
        # 计数走共用纯函数 recast.core.review_progress（R6 拆分，P3 第 3 波）；
        # 缓存与分母规则不变（total = 已确认 + 待审核全部项，N17）。
        stats = review_progress_stats(characters, queue)
        # N21 残留（P4）：低分样本确认也算确认动作。review_assign / 认领
        # 会给样本的人脸资产行打上“已审核”标记（service.move_face_sample /
        # _adopt_trash_sample），已确认人数却不变；丢弃（review_discard）
        # 不打这个标记，文件只进 review_trash/discarded/。所以多记一个
        # “已审核”标记数，供自动跳转判定用（不进共用纯函数：它是主窗口
        # 判定专用的信号，不是进度分母）。
        stats["reviewed"] = sum(
            1
            for ch in characters
            for a in (ch.get("assets") or [])
            if a.get("kind") == "face" and "已审核" in str(a.get("notes") or "")
        )
        self._review_stats_cache = stats
        return stats

    def _refresh_review_progress(self, guide: list[dict]) -> None:
        """审核进度条 + 队列清空时自动提示并跳转到第 ④ 步（只导航，不自动开始替换）。

        分母是已确认 + 待审核全部项（未确认聚类 + 低分样本）：低分样本
        只写在标签里不进分母时，队列只剩低分样本也会显示 100%（N17）。
        """
        stats = self._review_stats(guide)
        confirmed, pending, low = stats["confirmed"], stats["pending"], stats["low"]
        marked = stats.get("reviewed", 0)  # 打上“已审核”标记的人脸样本数
        total = confirmed + pending
        if total <= 0:
            self.review_progress.setVisible(False)
            self.review_progress_label.setText("审核进度：还没有人物。")
        elif confirmed <= 0:
            # 只有低分样本、一个已确认人物都没有：不是“还没有人物”，
            # 条按 0% 显示，别隐藏。
            self.review_progress.setVisible(True)
            self.review_progress.setValue(0)
            self.review_progress_label.setText(
                f"审核进度：待审核 {pending} 项，未确认人物（低分 {low}）"
            )
        else:
            self.review_progress.setVisible(True)
            self.review_progress.setValue(int(confirmed * 100 / total))
            self.review_progress_label.setText(
                f"审核进度：已确认 {confirmed} / 待审核 {pending}（低分 {low}）"
            )
        if pending > 0:
            if not self._review_was_nonempty:
                # 队列刚变非空：新一轮审核开始，确认计数清零。
                self._review_confirms_since_nonempty = 0
            else:
                # 本轮内两次刷新之间观测到的确认动作也累计进来：已确认
                # 人数涨了，或打上“已审核”标记的低分样本多了，都算。
                # 只累计增量（max(0, …)）：人物被删导致人数回落不扣数。
                self._review_confirms_since_nonempty += max(
                    0, confirmed - self._review_confirmed_when_nonempty
                )
                self._review_confirms_since_nonempty += max(
                    0, marked - self._review_marked_when_nonempty
                )
            self._review_was_nonempty = True
            self._review_confirmed_when_nonempty = confirmed
            self._review_marked_when_nonempty = marked
            return
        if not self._review_was_nonempty:
            return
        if self._oneclick is not None:
            return  # 一键成片自己会在关口继续进后期，不重复提示。
        if QApplication.activeModalWidget() is not None:
            return  # 有模态框开着（任务历史/设置），等下次刷新再提示。
        if not self.isVisible():
            # 隐藏/已关闭的窗口不弹提示：测试里泄漏的主窗口（定时器没停）
            # 之后被别的测试的 processEvents() 泵到时，exec() 会在离屏环境
            # 里永远等不到点击而挂住全套测试。先不消费这次转变，等窗口
            # 真正显示出来再提示。
            return
        self._review_was_nonempty = False
        # N21（P4 残留已修）：只有“确认”导致清空才弹自动跳转提示。
        # * 勾过 [不再提示]：以后都不弹（记在设置 review_jump_muted 里）。
        # * 零确认人物：不弹，状态栏提示一句（光丢弃、一个没确认时，
        #   跳去第 ④ 步也没东西可替换）。
        # * 丢弃清空：自队列上次非空以来没有任何确认动作（计数为 0），
        #   说明清空是丢弃不是确认——不弹。
        # 确认动作 = 已确认人数增量 + “已审核”标记增量：低分样本确认
        # （分给已有的人物 / 认领 / 新建人物）不涨已确认人数，但会给
        # 样本资产行打“已审核”标记，所以也计入。最后一跳（上次非空
        # 刷新 → 本次清空）之间发生的确认在这里一并折进计数——400ms
        # 轮询下，确认动作常发生在两次刷新之间，清空那次刷新才看得到。
        # 注意：计数只在真正消费这次转变时才折算；一键成片 / 有模态框 /
        # 窗口隐藏时提前返回、不消费，计数留到下次刷新再算。
        self._review_confirms_since_nonempty += max(
            0, confirmed - self._review_confirmed_when_nonempty
        )
        self._review_confirms_since_nonempty += max(
            0, marked - self._review_marked_when_nonempty
        )
        self._review_confirmed_when_nonempty = confirmed
        self._review_marked_when_nonempty = marked
        if getattr(self.service.settings, "review_jump_muted", False):
            return
        if confirmed <= 0:
            self.statusBar().showMessage("待审核项已清空，尚未确认人物。")
            return
        if self._review_confirms_since_nonempty <= 0:
            return
        parent = self._library if (self._library is not None and self._library.isVisible()) else self
        box = QMessageBox(parent)
        box.setWindowTitle("审核已完成")
        box.setText("待审核项已清空，可以去第 ④ 步替换 / 生成了。")
        go = box.addButton("去第 ④ 步", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("稍后", QMessageBox.ButtonRole.RejectRole)
        checkbox = None
        try:
            checkbox = QCheckBox("不再提示")
            setter = getattr(box, "setCheckBox", None)
            if callable(setter):
                setter(checkbox)
        except Exception:
            checkbox = None
        box.exec()
        if checkbox is not None:
            try:
                if checkbox.isChecked():
                    self.service.update_settings(review_jump_muted=True)
            except Exception:
                pass
        if box.clickedButton() is go:
            self._jump_to_step("replace")

    def _paint_stage_states(self, snap: dict, guide: list[dict], locked: bool) -> None:
        """9 个阶段复选框前加状态前缀：✓ 已完成 / ▶ 进行中 / ○ 未开始 / 🔒 锁住。

        阶段清单随入口自适应（PM 第 1 轮 U2）：当前选中剧目是导入视频时，
        模板/生成/配音三个阶段不适用——灰掉、取消勾选、标“仅剧本生成”，
        不显示为待办或失败；按剧本生成时反过来，扫描/提取/建库/替换标
        “仅导入再铸”。没选中剧目、或两种入口混选时不过滤，全显示。
        """
        by_id = {s["id"]: s["state"] for s in guide}
        dramas = snap.get("dramas") or []
        any_done = any(d.get("status") == ST_DONE for d in dramas)
        states: dict[str, str] = {
            "scan": "done" if dramas else "active",
            "extract": "done" if by_id.get("extract") == "done" else "active" if by_id.get("extract") == "active" else "todo",
            "library": "done" if by_id.get("review") == "done" else "active" if by_id.get("review") == "active" else "todo",
            "replace": "locked" if locked else "done" if any_done else "todo",
            "generate": "locked" if locked else "done" if any_done else "todo",
        }
        for sid in ("compose", "template", "dub", "export"):
            states[sid] = "done" if any_done else "todo"
        prefix = {"done": "✓ ", "active": "▶ ", "locked": "🔒 ", "todo": "○ "}
        entry = self._selected_entry_type()
        applicable = ENTRY_STAGES.get(entry) if entry else None
        for sid, btn in self.stage_buttons.items():
            label = _strip_stage_number(STAGE_LABELS.get(sid, sid))
            if applicable is not None and sid not in applicable:
                btn.setEnabled(False)
                btn.setChecked(False)
                btn.setText(f"{label}{self._entry_stage_note(sid, entry)}")
                btn.setToolTip("这个阶段只在另一条产品线跑，当前剧目用不上。")
                continue
            if sid in ("replace", "generate"):
                btn.setEnabled(not locked)
                btn.setToolTip("先完成人物审核（第 ③ 步）" if locked else "")
            else:
                btn.setEnabled(True)
                btn.setToolTip("")
            btn.setText(f"{prefix[states.get(sid, 'todo')]}{label}")

    # ----- 失败任务：失败阶段 + 失败原因 + 就地重试 -----

    def _last_checkpoint_stage(self, task_id: str) -> str | None:
        """该任务检查点走到的最后一步，换算成阶段名（失败前已完成到哪）。

        R6：实现在 recast.core.failed_tasks.last_checkpoint_stage，这里只委托。
        """
        return last_checkpoint_stage(self.service.store, task_id)

    def _failed_entries(self) -> list[dict[str, str]]:
        """失败的剧目：失败阶段 + 失败原因。

        R6：实现在 recast.core.failed_tasks.collect_failed_entries，这里只委托。
        """
        return collect_failed_entries(self.service.store)

    def _refresh_failed_tasks(self) -> None:
        """失败任务卡片：失败集合没变就不重建（400ms 刷新别抢焦点）。

        R6：卡片建 UI 在 recast.desktop.failed_cards.refresh_failed_cards，
        这里只取数 + 委托；集合不变时它返回 None，什么都不动。
        """
        sig, cards = refresh_failed_cards(
            layout=self.failed_cards_layout,
            empty_label=self.failed_empty_label,
            entries=self._failed_entries(),
            prev_sig=self._failed_sig,
            on_retry=self._retry_failed,
        )
        if cards is None:
            return
        self._failed_sig = sig
        self.failed_cards = cards

    def _retry_failed(self, drama_id: str) -> None:
        """从失败步续跑一条失败任务：按入口类型重跑适用阶段，检查点跳过已完成的步。"""
        drama = self.service.store.get("dramas", drama_id) or {}
        entry = drama.get("entry_type") or "import"
        stages = [sid for sid, _, _ in STAGE_GROUPS if sid in ENTRY_STAGES.get(entry, set())]
        self._bind_lipsync_confirm()
        try:
            self.service.start([drama_id], stages)
        except Exception as exc:
            QMessageBox.warning(self, "从失败步续跑失败", str(exc))
            return
        self.service.logs.append(
            "INFO",
            f"「{drama.get('new_name') or drama.get('origin_name') or drama_id}」已从失败步续跑："
            "只重跑失败步及之后（已完成的步骤按检查点跳过）。",
        )
        self.refresh(full=True)

    def _on_guide_jump(self, item: QListWidgetItem) -> None:
        sid = item.data(Qt.ItemDataRole.UserRole)
        if sid:
            self._jump_to_step(sid)

    def _on_next_step(self) -> None:
        guide = self._current_guide or self.service.guide_state()
        first_open = next((s for s in guide if s["state"] in ("active", "locked")), None)
        if first_open is None:
            return
        self._jump_to_step(first_open["id"])

    def _jump_to_step(self, sid: str) -> None:
        """「下一步」跳到对应面板：第 ②③ 步进人物库，第 ④ 步回执行栏，第 ⑤ 步打开声音与台词面板。"""
        if sid == "audio":
            self._show_audio_panel()
        elif sid in ("extract", "review"):
            self._open_library()
        elif sid == "replace":
            locked, reason = self.service.replace_locked()
            if locked:
                QMessageBox.information(self, "先完成人物审核", reason + "\n\n点左侧第 ③ 步去人物库处理完待审核项。")
                return
            self.start_btn.setFocus()
        else:  # import: 作品表就是入口，选中第一行
            if self.table.rowCount():
                self.table.selectRow(0)
                self.table.setFocus()

    # ----- 预览区 -----

    def _on_table_select(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        item = self.table.item(row, 1)
        if item is None:
            return
        drama_id = item.data(Qt.ItemDataRole.UserRole)
        self._run_precheck(drama_id)
        self._show_preview(drama_id)

    # ----- 第 ⑥ 项：导入预检卡 -----

    def _run_precheck(self, drama_id: str) -> None:
        """导入预检卡：工作线程里跑 ffprobe + 烧录字幕检测，结果回主线程显示。

        只有真正挡路的（没有视频文件 / 没有视频流 / 时长读不到）才阻止导入；
        没有音轨、烧录字幕、OCR 不可用都只是警告，不阻止。
        R6：探测本身在 recast.core.precheck.probe_precheck_payload，这里只启动线程。
        """
        self.precheck_panel.setVisible(True)
        self.precheck_label.setText("正在预检…")
        tasks = self.service.store.tasks_for_drama(drama_id)
        video = tasks[0].get("video_path") if tasks else None
        if not video:
            reason = "这部剧下面没有视频文件。"
            self._precheck[drama_id] = {"video": None, "blocked": reason}
            self._precheck_blocked[drama_id] = reason
            self.precheck_label.setText(f"预检没通过：{reason}\n导入已被阻止。")
            return

        def work() -> None:
            self._precheck_signal.ready.emit(
                probe_precheck_payload(video, drama_id, with_burned=True)
            )

        threading.Thread(target=work, name="recast-ui-precheck", daemon=True).start()

    def _apply_precheck_result(self, payload: dict) -> None:
        """把后台预检结果写进缓存与预检卡。

        R6：判断与文案在 recast.core.precheck.summarize_precheck，这里只
        写缓存（``_precheck`` / ``_precheck_blocked``）、按需写卡、刷行上的点。
        """
        drama_id = payload.get("drama_id")
        # 行状态点用的后台轻量预检（row_only）只更新缓存和行上的点，
        # 不碰当前选中的那部剧的预检卡；选片预检才写卡。
        show_label = (not payload.get("row_only")) or drama_id == self._preview_drama
        summary = summarize_precheck(payload)
        self._precheck[drama_id] = summary["record"]
        if summary["blocked"] is not None:
            self._precheck_blocked[drama_id] = summary["blocked"]
        else:
            self._precheck_blocked.pop(drama_id, None)
        if show_label:
            self.precheck_label.setText(summary["label_text"])
        self._paint_precheck_dot(drama_id)

    def _precheck_dot(self, drama_id: str) -> tuple[str, str]:
        """剧目行的预检状态点（N8）：✓ 通过 / ⚠ 警告 / ✗ 阻止；没查过返回空。

        R6：实现在 recast.core.precheck.precheck_dot，这里只委托。
        """
        return precheck_dot(self._precheck.get(drama_id))

    def _paint_precheck_dot(self, drama_id: str) -> None:
        """刚到一个预检结果就刷新该行的状态点，不等 400ms 的整表重绘。

        R6：实现在 recast.desktop.precheck_card.paint_precheck_dot，这里只委托。
        """
        dot, tip = self._precheck_dot(drama_id)
        paint_precheck_dot(self.table, drama_id, dot, tip)

    def _probe_rows_precheck(self, dramas: list[dict]) -> None:
        """行状态点用的轻量后台预检：只跑 ffprobe，不跑 OCR（选片时才跑 OCR）。

        没查过的行才查；结果走 _precheck_signal，和选片预检同一条回主线程的路。
        R6：单次探测在 recast.core.precheck.probe_precheck_payload，这里只排队。
        """
        pending = [d for d in dramas if d["id"] not in self._precheck]
        if not pending:
            return

        def work() -> None:
            for d in pending:
                did = d["id"]
                tasks = self.service.store.tasks_for_drama(did)
                video = tasks[0].get("video_path") if tasks else None
                if not video:
                    self._precheck_signal.ready.emit({
                        "drama_id": did, "video": video, "row_only": True,
                        "error": "这部剧下面没有视频文件。",
                    })
                    continue
                self._precheck_signal.ready.emit(
                    probe_precheck_payload(video, did, row_only=True, with_burned=False)
                )

        threading.Thread(target=work, name="recast-ui-row-precheck", daemon=True).start()

    def _show_preview(self, drama_id: str) -> None:
        self._preview_drama = drama_id
        try:
            drama = self.service.store.get("dramas", drama_id) or {}
        except Exception:
            drama = {}
        name = drama.get("new_name") or drama.get("origin_name") or drama_id
        episodes = drama.get("episode_count") or 0
        status = drama.get("status") or ""
        self.preview_text.setText(f"「{name}」· {episodes} 集 · 状态：{status}\n正在取首帧…")
        threading.Thread(target=self._load_preview, args=(drama_id,), name="recast-ui-preview", daemon=True).start()

    def _load_preview(self, drama_id: str) -> None:
        """工作线程：找第一集视频，取首帧存到临时缓存，主线程再显示。"""
        payload: dict = {"drama_id": drama_id}
        try:
            drama = self.service.store.get("dramas", drama_id) or {}
            root = drama.get("root_path")
            video = None
            if root:
                try:
                    hits = scan_root(root)
                    if hits and hits[0].episodes:
                        video = hits[0].episodes[0].video_path
                except Exception:
                    video = None
                if video is None:
                    for ext in (".mp4", ".mkv", ".mov", ".avi", ".m4v"):
                        found = sorted(Path(root).rglob(f"*{ext}"))
                        if found:
                            video = found[0]
                            break
            if video is not None:
                cache = Path(tempfile.gettempdir()) / "recast-preview"
                cache.mkdir(parents=True, exist_ok=True)
                out = cache / f"{hashlib.md5(drama_id.encode()).hexdigest()}.jpg"
                cap = open_capture(video)
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    import cv2

                    good, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if good:
                        out.write_bytes(buf.tobytes())
                        payload["frame"] = str(out)
                payload["video"] = str(video)
        except Exception as exc:
            payload["error"] = str(exc)
        self._preview_signal.ready.emit(payload)

    def _apply_preview(self, payload: dict) -> None:
        if payload.get("drama_id") != self._preview_drama:
            return  # 用户已经选了别的剧，这帧过期了
        frame = payload.get("frame")
        if frame:
            pix = QPixmap(frame)
            if not pix.isNull():
                self.preview_image.setPixmap(
                    pix.scaled(640, 360, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                )
                text = self.preview_text.text().replace("正在取首帧…", f"首帧：{Path(payload.get('video', '')).name}")
                self.preview_text.setText(text)
                return
        self.preview_image.setText("没有可预览的画面：这部剧下面找不到视频文件。")

    # ----- data -----

    def _selected_stages(self) -> list[str]:
        if self.all_stages.isChecked():
            stages = [sid for sid, _, _ in STAGE_GROUPS]
        else:
            stages = [sid for sid, btn in self.stage_buttons.items() if btn.isChecked()]
        # 阶段清单随入口自适应：不相关的阶段不进执行列表（界面上已灰掉，
        # 这里再拦一道，免得“全部阶段按序执行”把它带回来）。
        entry = self._selected_entry_type()
        if entry in ENTRY_STAGES:
            stages = [s for s in stages if s in ENTRY_STAGES[entry]]
        if getattr(self, "_guide_locked", False):
            # 第 ③ 步没完成：替换/生成锁住，不参与执行（按钮本身也灰掉）。
            stages = [s for s in stages if s not in ("replace", "generate")]
        if getattr(self, "upload_stage", None) is not None and self.upload_stage.isChecked():
            if "upload" not in stages:
                stages.append("upload")
        return stages

    def _wanted_locked_stages(self) -> bool:
        """用户是否点了被锁的阶段（全部执行，或单独勾了替换/生成）。"""
        if self.all_stages.isChecked():
            return True
        return any(
            self.stage_buttons.get(sid) is not None and self.stage_buttons[sid].isChecked()
            for sid in ("replace", "generate")
        )

    def _toggle_all_stages(self, checked: bool) -> None:
        entry = self._selected_entry_type()
        applicable = ENTRY_STAGES.get(entry) if entry else None
        for sid, btn in self.stage_buttons.items():
            if checked and getattr(self, "_guide_locked", False) and sid in ("replace", "generate"):
                btn.setChecked(False)  # 锁住的阶段不跟着「全部执行」勾上
                continue
            if checked and applicable is not None and sid not in applicable:
                btn.setChecked(False)  # 不适用于当前入口的阶段不跟着勾上
                continue
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

    def _selected_entry_type(self) -> str | None:
        """当前选中剧目的入口类型：import（已有视频）/ script（按剧本生成）。

        没选中、或同时选中两种入口的剧目时返回 None（不过滤，全显示）。
        """
        ids = self._drama_ids_checked()
        if not ids:
            cur = self._current_drama_id()
            ids = [cur] if cur else []
        if not ids:
            return None
        types = set()
        for did in ids:
            drama = self.service.store.get("dramas", did) or {}
            types.add(drama.get("entry_type") or "import")
        if len(types) != 1:
            return None
        return types.pop()

    def _entry_stage_note(self, sid: str, entry: str) -> str:
        """不适用阶段的标注：告诉用户它只在另一条线跑。"""
        if entry == "import":
            return "（仅剧本生成）"
        return "（仅导入再铸）"

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
        """Queue a snapshot on a worker thread. The 400ms timer must not block the UI.

        Main-thread only: every caller is a QTimer timeout, a Qt signal
        slot, or a UI handler running on the main thread (constructor,
        filter change, scan/import/delete/recheck/start/reface/export/
        stop/clear/settings/generate handlers, the rerender-result slot,
        and the snapshot-result slot ``_apply_snapshot`` itself). The
        ``_snap_busy`` / ``_snap_again`` / ``_snap_full`` flags are
        therefore only ever read and written on the main thread; the
        worker closure ``work()`` below never touches them -- it only
        calls ``service.snapshot()`` and emits the result back via
        ``_snap_signal``. Do not call this from a worker thread.

        Timer ticks (``full=False``) take the light engine path
        (``light_engine=True``): the engine section reuses the warm
        cache past its TTL, so the tick does not re-probe heavy modules
        every 5 seconds. Explicit full refreshes (``full=True``) force a
        full recompute.
        """
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
                snap = self.service.snapshot(status_filter=status_filter, light_engine=not want_full)
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

    def _fit_wrapped_labels(self) -> None:
        """换行标签按当前宽度定最小高：顶栏两行 + 右栏 detect_label。

        嵌套布局里的 heightForWidth 需要两轮激活才收敛（第一轮常按
        旧高度排，标签被压回单行高）；直接按当前宽度算出换行所需高
        度写成最小高，一轮布局就排对。窗口未显示时宽度为 0，跳过，
        下次 _paint（400ms 刷新）或 resize 时再算。
        """
        for lab in (getattr(self, "engine_label", None), getattr(self, "hw_label", None),
                    getattr(self, "detect_label", None)):
            if lab is None:
                continue
            width = lab.width()
            if width > 0:
                # 空文本时 heightForWidth 返回 -1，不得写进最小高。
                needed = lab.heightForWidth(width)
                if needed > 0:
                    lab.setMinimumHeight(needed)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        # 窗口变窄 → 换行标签变矮变宽，最小高要跟着重算。
        self._fit_wrapped_labels()

    def _paint(self, snap: dict, full: bool = False) -> None:
        engine = snap.get("engine") or {}
        state = engine.get("state") or ENG_IDLE
        self.engine_label.setText(state)
        self.engine_label.setToolTip(state)  # P5/B3：换行显示时完整文本仍可看
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
        self.hw_label.setToolTip(hw + extra)  # P5/B3：换行显示时完整文本仍可看

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
        self.detect_label.setToolTip("    ".join(bits))  # P5/B3：换行后完整文本仍可看
        self._fit_wrapped_labels()

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
        self._refresh_failed_tasks()
        self._refresh_guide(snap)
        self._maybe_advance_one_click(snap)
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
        self._fit_table_height()
        self._rebuild_log_task_filter(dramas)
        self._probe_rows_precheck(dramas)

    def _fit_table_height(self) -> None:
        """作品表只占它需要的高度（最多约 8 行）：空出来的位置留给预览区。"""
        rows = self.table.rowCount()
        row_h = self.table.rowHeight(0) if rows else 28
        header_h = self.table.horizontalHeader().height() or 28
        height = header_h + row_h * rows + 4
        self.table.setMaximumHeight(min(340, max(120, height)))

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
        origin.setToolTip(d.get("origin_name") or "")  # P5/B3：Stretch 列内容超长时省略显示，完整文本放 tooltip
        self.table.setItem(row, 1, origin)

        new_name = QTableWidgetItem(d.get("new_name") or d.get("origin_name") or "")
        new_name.setToolTip(d.get("new_name") or d.get("origin_name") or "")
        self.table.setItem(row, 2, new_name)

        count = QTableWidgetItem(str(d.get("episode_count") or 0))
        count.setFlags(count.flags() & ~Qt.ItemFlag.ItemIsEditable)
        count.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 3, count)

        status = QTableWidgetItem(d.get("status") or "")
        status.setFlags(status.flags() & ~Qt.ItemFlag.ItemIsEditable)
        status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._color_status(status, d.get("status") or "")
        dot, tip = self._precheck_dot(d["id"])
        status.setText(f"{dot}{d.get('status') or ''}")
        if tip:
            status.setToolTip(tip)
        self.table.setItem(row, 4, status)

        prog = QTableWidgetItem(d.get("progress") or "")
        prog.setFlags(prog.flags() & ~Qt.ItemFlag.ItemIsEditable)
        prog.setToolTip(d.get("progress") or "")
        self.table.setItem(row, 5, prog)

        open_btn = QPushButton("打开")
        # P5/B3：第 6 列 ResizeToContents 按按钮尺寸定宽；按钮最小宽
        # 64（文字 + 内边距），1280x720 下「打开」两字完整显示、不再变「…」。
        open_btn.setMinimumWidth(64)
        open_btn.setEnabled(bool(d.get("output_dir")))
        open_btn.clicked.connect(lambda *_a, i=d["id"]: self._open_output_id(i))
        self.table.setCellWidget(row, 6, open_btn)

        created = QTableWidgetItem(d.get("created_at") or "")
        created.setFlags(created.flags() & ~Qt.ItemFlag.ItemIsEditable)
        created.setToolTip(d.get("created_at") or "")
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
            dot, tip = self._precheck_dot(d["id"])
            st.setText(f"{dot}{d.get('status') or ''}")
            st.setToolTip(tip)
            self._color_status(st, d.get("status") or "")
            self.table.item(row, 5).setText(d.get("progress") or "")
            self.table.item(row, 5).setToolTip(d.get("progress") or "")
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
        # 日志面板：空闲时收窄，有输出时展开（只在状态变化时调一次，不跟用户拖拽打架）。
        has_logs = self.log_list.count() > 0
        if has_logs != self._log_expanded:
            self._log_expanded = has_logs
            if getattr(self, "_right_splitter", None) is not None:
                self._right_splitter.setSizes([260, 320] if has_logs else [260, 120])

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

    def _find_scan_merge_target(self, path: str) -> list[dict]:
        """文件夹扫描前找“首集已不在本次扫描中”的已有剧目（N30 残留）。

        对 ``scan_root(path)`` 的每个 hit 调共用查重
        ``service.find_drama_for_hit``；只有当命中的剧目行身份文件
        （``source_path``，即原来的首集）不在本次扫描的分集里时才
        列入——也就是首集被删/改名后重扫的情形，旧行应被认出来并入，
        而不是静默建新行。正常重扫（首集还在）不列入、不弹框。
        ``scan_root`` 抛错（路径不存在等）时返回 []，让后面的
        ``service.scan`` 按原流程报错，不在这里多弹框。
        """
        try:
            hits = scan_root(path)
        except Exception:
            return []
        targets: list[dict] = []
        for hit in hits:
            drama = self.service.find_drama_for_hit(hit)
            if drama is None:
                continue
            hit_paths = {str(ep.video_path) for ep in hit.episodes}
            if (drama.get("source_path") or "") not in hit_paths:
                targets.append(drama)
        return targets

    def _scan(self) -> None:
        path = self.root_edit.text().strip()
        if not path:
            path = QFileDialog.getExistingDirectory(self, "选择作品根目录", str(Path.home()))
            if not path:
                return
            self.root_edit.setText(path)
        # N30 残留：首集删了再扫文件夹，旧行要先被认出来问用户是否
        # 并入/恢复，而不是静默建新行。用户选“否”则本次不扫描、不改库。
        merge_targets = self._find_scan_merge_target(path)
        if merge_targets:
            first = merge_targets[0]
            name = (
                first.get("new_name")
                or first.get("origin_name")
                or first.get("id")
            )
            extra = f"等 {len(merge_targets)} 部剧目" if len(merge_targets) > 1 else ""
            answer = QMessageBox.question(
                self,
                "扫描",
                f"检测到已有剧目「{name}」{extra}：本次扫描的首集文件已不在其中"
                "（可能已删除或改名），是否并入/恢复到该剧目？\n"
                "选“是”并入原剧目（保留原任务与进度）；选“否”本次不扫描。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.statusBar().showMessage("已取消扫描，未改动剧目。")
                return
        try:
            self.service.scan(path)
        except Exception as exc:
            QMessageBox.warning(self, "扫描失败", str(exc))
            return
        self.refresh(full=True)

    def _import_single_video(self) -> None:
        """导入单个视频文件（M9 / 设计 1.1）：选文件 → 自动建剧目 + 任务。

        不必先把文件放进作品目录结构；导入后同样进作品表。入库复用
        现有扫描 / 建剧目流程（``service.scan`` → ``scan_root`` 已支持
        单个视频文件 → ``_commit_hit`` 建剧目 + 分集任务），不另起一套。
        N30：剧目身份是首个分集视频的文件路径（``dramas.source_path``
        UNIQUE），不再是文件夹。同一文件再次入库算碰撞——导入前先由
        ``_find_import_collision`` 查，撞上了就弹确认框问“是否重新
        导入/覆盖”，用户取消则不导入、不写脏数据；同文件夹、不同视频
        不再是碰撞，直接新建一部剧目，不弹框、不替换。
        N39：非首集视频单独导入也算碰撞（该视频已是某剧目的分集任务），
        确认框问“是否并入该剧目”，确认后并入已有行，不建新行。
        """
        exts = " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTS))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入单个视频文件",
            self.root_edit.text() or str(Path.home()),
            f"视频文件 ({exts})",
        )
        if not path:
            return
        self._import_single_video_file(path)

    def _find_import_collision(self, path: str) -> dict | None:
        """单视频导入的碰撞检测（N30 + N39）：同一文件是否已在库中。

        查两处，任一命中即碰撞，返回已有的那部剧目行，撞不上返回 None：
        1. ``dramas.source_path``——剧目身份是首个分集视频的文件路径
           （与 ``_commit_hit`` 的算法一致）；
        2. ``tasks.video_path``（N39）——非首集视频只在分集任务里出现，
           只查 source_path 会漏掉：文件夹扫描建了 D1（第一集、第二集），
           再单独导入第二集也必须算碰撞，否则会建出重复剧目行 D2。
        同文件夹、不同视频、且该视频不在任何任务里，仍不是碰撞（会建
        新剧目），不要按 root_path 查。碰撞条件只集中在这里，不要散落
        在各处；去重规则再变也只改这一个 helper。
        """
        try:
            source = str(Path(path).expanduser().resolve())
        except Exception:
            return None
        try:
            row = self.service.store.find_drama_by_source(source)
            if row is not None:
                return row
            return self.service.store.find_drama_by_task_video(source)
        except Exception:
            return None

    def _import_single_video_file(self, path: str) -> bool:
        """单视频导入的校验 + 入库（可直接测，不依赖文件对话框）。

        N30：先查同一文件碰撞——该视频已在库中时弹确认框问“是否重新
        导入/覆盖”，用户取消就直接返回 False，不调 scan、不写脏数据。
        同文件夹、不同视频不算碰撞，直接入库建新剧目。
        """
        video = Path(path)
        if not video.is_file():
            QMessageBox.warning(self, "导入单个视频", f"找不到视频文件：{path}")
            return False
        if video.suffix.lower() not in VIDEO_EXTS:
            QMessageBox.warning(
                self,
                "导入单个视频",
                f"不是视频文件：{video.name}（仅支持 {', '.join(sorted(VIDEO_EXTS))}）。",
            )
            return False
        collision = self._find_import_collision(path)
        if collision is not None:
            name = (
                collision.get("new_name")
                or collision.get("origin_name")
                or collision.get("id")
            )
            try:
                resolved = str(Path(path).expanduser().resolve())
            except Exception:
                resolved = path
            if collision.get("source_path") == resolved:
                detail = f"该视频「{video.name}」已在库中（剧目「{name}」），是否重新导入/覆盖？"
            else:
                # N39：非首集视频撞上的是分集任务——问的是“并入已有剧目”，
                # 不是覆盖；确认后 _commit_hit 会并入该行，不建新行。
                detail = (
                    f"该视频「{video.name}」已在库中，是剧目「{name}」的一集，"
                    "是否并入该剧目重新导入？"
                )
            answer = QMessageBox.question(
                self,
                "导入单个视频",
                detail,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        try:
            self.service.scan(str(video))
        except Exception as exc:
            QMessageBox.warning(self, "导入单个视频失败", str(exc))
            return False
        self.refresh(full=True)
        return True

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

    # ----- 一键成片：两段跑 + 审核关口 + 失败停住 -----

    def _one_click_phases(self, entry: str) -> tuple[list[str], list[str]]:
        """一键成片分两段跑：前期（审核关口之前）→ 后期（替换/合成/导出）。

        中间卡一道审核关口：前期跑完先看审核队列，非空就停下来等用户确认，
        绝不带着未确认的人脸进替换。生成路径的前期是剧本/生成/配音。
        """
        if entry == "script":
            return ["template", "generate", "dub"], ["replace", "compose", "export"]
        return ["scan", "extract", "library"], ["replace", "compose", "export"]

    def _phase_a_done(self, drama_id: str, entry: str) -> bool:
        """前期跑完了没有：导入看有没有 identity 检查点，生成看有没有静音成片。"""
        tasks = self.service.store.tasks_for_drama(drama_id)
        if entry == "script":
            for task in tasks:
                workdir = task.get("workdir")
                if workdir and (Path(workdir) / "silent.mp4").is_file():
                    return True
            return False
        for task in tasks:
            try:
                rows = self.service.store.list_rows(
                    "checkpoints", "task_id = ?", (task["id"],)
                )
            except Exception:
                continue
            if any(row.get("step") == "identity" for row in rows):
                return True
        return False

    def _start_one_click_phase(self, drama_id: str, entry: str, phase: str) -> None:
        phase_a, phase_b = self._one_click_phases(entry)
        if phase == "A":
            stages = phase_a
            label = (
                "一键成片：前期进行中（剧本 / 生成 / 配音…）"
                if entry == "script"
                else "一键成片：前期进行中（导入 / 提取 / 建库…）"
            )
        else:
            stages = phase_b
            label = "一键成片：替换 / 合成 / 导出进行中…"
        self._bind_lipsync_confirm()
        try:
            self.service.start([drama_id], stages)
        except Exception as exc:
            QMessageBox.warning(self, "一键成片", str(exc))
            self._oneclick = None
            self.oneclick_label.setText(f"一键成片没能开始：{exc}")
            return
        self._oneclick = {
            "drama_id": drama_id,
            "phase": phase,
            "entry": entry,
            "prompted_review": False,
            "seen_running": False,
        }
        self.oneclick_label.setText(label)
        self.refresh(full=True)

    def _one_click(self) -> None:
        """一键成片：选一部剧，一键跑到成片（两段跑，中间过审核关口）。"""
        if self._oneclick is not None:
            self._oneclick = None
            self.oneclick_label.setText("一键成片已取消。")
            return
        drama_id = self._current_drama_id()
        if not drama_id:
            QMessageBox.information(self, "一键成片", "请先勾选或选中一部剧。")
            return
        drama = self.service.store.get("dramas", drama_id) or {}
        entry = drama.get("entry_type") or "import"
        locked, reason = self.service.replace_locked()
        if locked:
            QMessageBox.information(
                self,
                "一键成片停在审核关口",
                reason + "\n\n先去第 ③ 步处理完待审核项，再点一键成片。",
            )
            self._open_library()
            return
        if self._precheck_blocked.get(drama_id):
            QMessageBox.warning(
                self, "一键成片", f"导入预检没通过：{self._precheck_blocked[drama_id]}"
            )
            return
        if self._phase_a_done(drama_id, entry):
            if not self._auth_gate_ok(prompt=True, drama_ids=[drama_id]):
                self.oneclick_label.setText("一键成片停在审核关口：有未授权人物，补完授权再继续。")
                return
            self._start_one_click_phase(drama_id, entry, "B")
        else:
            self._start_one_click_phase(drama_id, entry, "A")

    def _authorized_replace_count(self) -> int:
        """有参考脸且已授权的人物数（和 unauthorized_replace_characters 同一来源）。

        关口先算这个数：零个时“继续，只替换已授权人物”兑现不了——
        替换阶段的 check_replace_ready 要求至少一个已授权参考脸，
        否则 service.start() 会把任务 parked 成待补资料（N18）。
        计数逻辑在 recast.core.entry_shared.check_replace_auth 里，
        这里只调它，不再重复算。
        """
        try:
            return int(check_replace_auth(self.service)["authorized_count"])
        except Exception:
            return 0

    def _record_skipped_characters(
        self, missing: list[dict], drama_ids: list[str] | None
    ) -> None:
        """把关口“继续”后被跳过的人物名留下来（N18）。

        落点有两处：
        * 持久记录：``store.record_auth_skip`` 写进 ``auth_skips`` 表
          （同一 drama+task+人名去重），runner 重写 tasks/dramas 的
          progress 也冲不掉，事后用 ``skipped_character_names`` 能查到；
        * 可见提示：任务 / 剧目 progress 追加“「温雪」未授权，本次未
          替换”，作品表和队列面板都能看到；日志再写一条 WARNING。
          progress 文本仍会被 runner 按步骤重写，只保留到任务真正
          开始跑为止——持久追溯以 ``auth_skips`` 为准。
        """
        names = "、".join(m["name"] for m in missing)
        note = f"{names}未授权，本次未替换"
        self.service.logs.append("WARNING", note + "（只替换已授权人物）。")
        for did in drama_ids or []:
            try:
                tasks = self.service.store.tasks_for_drama(did)
            except Exception:
                tasks = []
            for task in tasks:
                # 持久记录放 progress 写之前：progress 有“已记过就跳过”
                # 的 continue，record_auth_skip 自己按 (drama, task, 人名)
                # 去重，多调一次无害。
                for m in missing:
                    try:
                        self.service.store.record_auth_skip(
                            did, m["name"], task_id=task["id"], reason="未授权"
                        )
                    except Exception:
                        pass
                old = (task.get("progress") or "").strip()
                if note in old:
                    continue
                new = f"{old}；{note}" if old else note
                try:
                    self.service.store.update("tasks", task["id"], progress=new)
                except Exception:
                    pass
            if not tasks:
                # 剧目下面还没分集任务：先按 drama 级记一条，免得跳过记录丢了。
                for m in missing:
                    try:
                        self.service.store.record_auth_skip(
                            did, m["name"], reason="未授权"
                        )
                    except Exception:
                        pass
            try:
                drama = self.service.store.get("dramas", did) or {}
            except Exception:
                drama = {}
            if drama:
                old = (drama.get("progress") or "").strip()
                if note not in old:
                    new = f"{old}；{note}" if old else note
                    try:
                        self.service.store.update("dramas", did, progress=new)
                    except Exception:
                        pass

    def _auth_gate_ok(
        self, *, prompt: bool = True, drama_ids: list[str] | None = None
    ) -> bool:
        """审核关口的授权预检（N13）：进替换之前先查未授权人物。

        有参考脸但未授权的人物，替换时会直接跳过——关口提前告诉用户，
        而不是等替换阶段才把任务 parked 成待补资料。返回 False 表示
        先别进后期；``prompt=False`` 时只检查不弹框（关口已提示过，
        等用户补完授权再来）。

        N18：零已授权时 [继续] 灰掉——没有已授权人物，继续也无法替换，
        点了只会让任务在替换阶段被 parked；点 [继续] 时把被跳过的人物名
        写进任务 progress + 日志（见 _record_skipped_characters）。
        未授权列表与已授权数都由 recast.core.entry_shared.check_replace_auth
        一次算出，这里只负责弹框，不再重复算关口逻辑。

        P3/N32：弹框逻辑已抽到 recast.desktop.auth_gate_dialog.auth_gate_ok，
        三个替换入口（本窗口的三个 + 队列面板重试）共用；这里只是薄壳，
        把本窗口的 QMessageBox（测试替它打 fake 的那一个）、开人物库
        与记跳过人物的回调传进去，行为与原来一字不差。
        """
        return auth_gate_ok(
            self,
            self.service,
            drama_ids,
            prompt=prompt,
            on_go_library=self._open_library,
            on_skipped=self._record_skipped_characters,
            _box_cls=QMessageBox,
        )

    def _bind_lipsync_confirm(self) -> None:
        """合成前唇同步警告（4.3①）：把弹框回调挂到共用的 settings 对象上。

        任务跑在调度器的工作线程里，runner 不能直接弹框；``dispatch_job``
        从 ``settings.lipsync_confirm`` 把回调读回并传进 ``run_episode``，
        回调内部（``ask_lipsync_on_gui_thread``）再把框发到 GUI 线程执行。
        每次 ``service.start`` 前都重新挂一次：改设置会换一个新的
        settings 对象（``SettingsManager.update``），旧对象上的回调就丢了。
        没挂上时 runner 只记警告不拦路，绝不因缺回调挡住一次合成。
        """
        try:
            self.service.settings.lipsync_confirm = (
                lambda warnings: ask_lipsync_on_gui_thread(self, warnings)
            )
        except Exception:
            pass

    def _advance_after_phase_a(self, drama_id: str, oc: dict) -> None:
        """前期跑完：先过审核关口，队列空了才进后期；非空就停下并跳去第 ③ 步。"""
        queue = self.service.list_review_queue()
        if queue:
            oc["phase"] = "review"
            self.oneclick_label.setText(
                f"一键成片停在审核关口：待审核 {len(queue)} 项，先去第 ③ 步处理完。"
            )
            if not oc.get("prompted_review"):
                oc["prompted_review"] = True
                QMessageBox.information(
                    self,
                    "一键成片停在审核关口",
                    f"待审核 {len(queue)} 项：先在人物库里处理完，"
                    "处理完会自动继续替换 / 合成 / 导出。",
                )
                self._open_library()
            return
        if not self._auth_gate_ok(prompt=not oc.get("prompted_auth"), drama_ids=[drama_id]):
            oc["phase"] = "review"
            oc["prompted_auth"] = True
            self.oneclick_label.setText("一键成片停在审核关口：有未授权人物，补完授权再继续。")
            return
        self._start_one_click_phase(drama_id, oc["entry"], "B")

    def _maybe_advance_one_click(self, snap: dict) -> None:
        """一键成片的状态机，每次快照推进一次：只在“看见跑过、现在停了”时下结论。

        seen_running 是防抖：刚入队还没跑起来时，任务全是待执行，不能误判
        成“前期跑完了但没提取到人物”。失败只停在该阶段，等左栏[从失败步续跑]。
        """
        oc = self._oneclick
        if not oc:
            return
        sched = snap.get("scheduler") or {}
        drama_id = oc["drama_id"]
        tasks = self.service.store.tasks_for_drama(drama_id)
        if sched.get("running") or any(t.get("status") == ST_RUNNING for t in tasks):
            oc["seen_running"] = True
            return
        if any(t.get("status") == ST_FAILED for t in tasks):
            if oc["phase"] != "failed":
                oc["failed_phase"] = oc["phase"]
                oc["phase"] = "failed"
                self.oneclick_label.setText(
                    "一键成片在某阶段失败，原因见左栏「失败任务」，"
                    "点[从失败步续跑]继续（只重跑失败步及之后）。"
                )
            return
        if oc["phase"] == "failed":
            # 手动重试后：重试那段跑完没有，跑完就继续。
            failed_phase = oc.get("failed_phase") or "A"
            if failed_phase == "A":
                if self._phase_a_done(drama_id, oc["entry"]):
                    self._advance_after_phase_a(drama_id, oc)
            elif tasks and all(t.get("status") == ST_DONE for t in tasks):
                self.oneclick_label.setText("一键成片完成 ✓")
                self._oneclick = None
            return
        if oc["phase"] == "review":
            if not self.service.list_review_queue():
                # 队列空了：重走一次关口（队列 + 授权都再查一遍），而不是
                # 直接进后期——授权可能还没补，补完才自动继续。
                self._advance_after_phase_a(drama_id, oc)
            return
        if not oc.get("seen_running"):
            return  # 还没跑起来，等一等，别误判。
        if oc["phase"] == "A":
            if self._phase_a_done(drama_id, oc["entry"]):
                self._advance_after_phase_a(drama_id, oc)
            else:
                self.oneclick_label.setText(
                    "一键成片：前期跑完了，但没提取到人物——请检查视频里是否有人脸，或看日志找原因。"
                )
                self._oneclick = None
            return
        if oc["phase"] == "B":
            if tasks and all(t.get("status") == ST_DONE for t in tasks):
                self.oneclick_label.setText("一键成片完成 ✓")
                self._oneclick = None
            elif any(t.get("status") == ST_NEED_ASSETS for t in tasks):
                self.oneclick_label.setText(
                    "一键成片停住：缺少已授权参考脸（待补资料），补好资料再点一键成片。"
                )
                self._oneclick = None
            return

    # ----- 第 ⑤ 步声音与台词 -----

    def _show_audio_panel(self) -> None:
        self.audio_panel.setVisible(True)
        self._refresh_audio_panel()

    def _diar_segments(self) -> list[dict]:
        """分轨结果：各人物文件夹 voice/ 下的 SPK_<起>_<止>.wav 片段，按时间排。

        R6：实现在 recast.core.audio_segments.collect_diar_segments，这里只委托。
        文件名就是分轨的全部记录（run_diarization 按这个格式写），不靠
        别的索引；没跑过分轨就返回空列表。
        """
        return collect_diar_segments(self.service)

    def _refresh_audio_panel(self) -> None:
        """重建第 ⑤ 步面板：分轨表（每行一个「改派给」下拉框）+ 每个人物一行试听/调音色/改台词。

        R6：建表建按钮在 recast.desktop.audio_panel（refresh_diar_table /
        rebuild_character_rows / audio_status_text），分轨取数在
        recast.core.audio_segments.collect_diar_segments，这里只组装。
        """
        segments = self._diar_segments()
        self._diar_rows = segments
        characters = self.service.characters()
        refresh_diar_table(self.diar_table, segments, characters, self._on_reassign_segment)
        self._audio_char_btns = rebuild_character_rows(
            self.audio_char_layout,
            characters,
            on_audition=self._audition_character,
            on_timbre=self._open_timbre,
            on_dialogue=self._open_dialogue,
        )
        self.audio_status.setText(audio_status_text(segments))

    def _on_reassign_segment(self, row: int) -> None:
        """改派：把片段文件挪到目标人物的 voice/（或未分配区），并同步两边 meta.json 的 voice.samples。

        R6：挪文件与同步 meta 在 recast.core.audio_segments.reassign_segment，
        这里只读下拉框的目标、写状态文案。
        """
        if row >= len(self._diar_rows):
            return
        seg = self._diar_rows[row]
        combo = self.diar_table.cellWidget(row, 3)
        target_id = combo.currentData() if combo is not None else None
        if target_id == seg["owner_id"]:
            return
        try:
            message = reassign_segment(self.service, seg, target_id)
        except FileNotFoundError:
            self.audio_status.setText(f"片段文件找不到了：{seg['path']}")
            return
        self.audio_status.setText(message)

    def _update_voice_samples(self, char_id: str | None, *, remove_path, add_path) -> None:
        """同步一个人物 meta.json 的 voice.samples：移走一个、加一个（文件已挪完才调）。

        R6：实现在 recast.core.audio_segments.update_voice_samples，这里只委托。
        """
        update_voice_samples(self.service, char_id, remove_path=remove_path, add_path=add_path)

    def _character_voice_samples(self, char_id: str) -> list[Path]:
        """一个角色已有的声音样本：meta 登记的 + voice/ 目录里直接扫到的。

        R6：实现在 recast.core.audio_segments.character_voice_samples，这里只委托。
        """
        return character_voice_samples(self.service, char_id)

    def _open_timbre(self, char_id: str) -> None:
        from recast.desktop.timbre_dialog import TimbreDialog

        ch = self.service.store.get_character(char_id) or {}
        name = ch.get("display_name") or ch.get("name") or char_id
        TimbreDialog(self.service, str(name), parent=self).exec()

    def _open_dialogue(self, char_id: str) -> None:
        from recast.desktop.dialogue_dialog import DialogueDialog

        ch = self.service.store.get_character(char_id) or {}
        name = ch.get("display_name") or ch.get("name") or char_id
        DialogueDialog(self.service, str(name), parent=self).exec()

    def _run_audio_job(
        self, kind: str, work_fn, *, result_key: str, thread_name: str, extra: dict | None = None
    ) -> None:
        """第 ⑤ 步后台任务的公共壳：跑 work_fn，结果/异常装进 payload 发回主线程。

        只收三个任务（分轨 / 试听 / 重配音）共用的线程样板；成功时
        ``payload[result_key] = work_fn()``，失败时 ``payload["error"]``。
        """

        def work() -> None:
            payload: dict = {"kind": kind}
            if extra:
                payload.update(extra)
            try:
                payload[result_key] = work_fn()
            except Exception as exc:
                payload["error"] = str(exc)
            self._audio_signal.ready.emit(payload)

        threading.Thread(target=work, name=thread_name, daemon=True).start()

    def _rediarize(self) -> None:
        drama_id = self._current_drama_id()
        if not drama_id:
            QMessageBox.information(self, "重新分轨", "请先勾选或选中一部剧。")
            return
        tasks = self.service.store.tasks_for_drama(drama_id)
        if not tasks or not tasks[0].get("video_path"):
            QMessageBox.information(self, "重新分轨", "这部剧下面没有视频文件。")
            return
        video = tasks[0]["video_path"]
        names = [
            str(c.get("display_name") or c.get("name"))
            for c in self.service.characters()
            if c.get("display_name") or c.get("name")
        ]
        self.audio_status.setText("正在分轨…")
        self.rediar_btn.setEnabled(False)
        self._run_audio_job(
            "rediarize",
            lambda: self.service.run_diarization(video, character_names=names or None),
            result_key="report",
            thread_name="recast-ui-rediarize",
        )

    def _audition_character(self, char_id: str) -> None:
        """试音 3 秒：先用该角色音色合成；合不出来就放已有的声音样本；都没有就明说缺什么——绝不伪造音频。"""
        self.audio_status.setText("正在合成 3 秒试听…")
        self._run_audio_job(
            "audition",
            lambda: self.service.audition_voice(char_id),
            result_key="rec",
            thread_name="recast-ui-audition",
            extra={"character": char_id},
        )

    def _apply_audio_result(self, payload: dict) -> None:
        """第 ⑤ 步后台任务的结果回主线程：分轨 / 试听 / 重配音三路。

        R6：实现在 recast.desktop.audio_panel.apply_audio_result，这里只委托。
        """
        apply_audio_result(self, payload)

    def _redub(self) -> None:
        drama_id = self._current_drama_id()
        tasks = self.service.store.tasks_for_drama(drama_id) if drama_id else []
        workdir = tasks[0].get("workdir") if tasks else None
        if not workdir or not Path(workdir).is_dir():
            QMessageBox.information(
                self, "重配音", "这部剧还没有分集工作目录（先跑一次生成/配音）。"
            )
            return
        self.audio_status.setText("正在重配音（只重配改过的行）…")
        self._run_audio_job(
            "redub",
            lambda: self.service.redub_dialogue(workdir),
            result_key="report",
            thread_name="recast-ui-redub",
        )

    def _edit_subtitles(self) -> None:
        """打开字幕编辑器（改字、拆分、合并、检查时间轴）；改完刷新第 ⑤ 步面板。

        R6：找字幕文件在 recast.desktop.audio_panel.find_subtitle_target，
        这里只开对话框。
        """
        from recast.desktop.subtitle_dialog import SubtitleDialog

        srt, task_id, error = find_subtitle_target(self.service, self._current_drama_id())
        if error:
            QMessageBox.information(self, "编辑字幕", error)
            return
        shots = self.service.store.shots_for_task(task_id)
        dlg = SubtitleDialog(srt, parent=self, shots=shots)
        dlg.exec()
        self._refresh_audio_panel()

    def _start(self) -> None:
        ids = self._drama_ids_checked()
        # N8：单个坏文件只跳过它、并点名，不再整批拒绝。
        blocked = [(did, self._precheck_blocked[did]) for did in ids if did in self._precheck_blocked]
        if blocked and len(blocked) == len(ids):
            QMessageBox.warning(self, "无法开始", f"导入预检没通过：{blocked[0][1]}\n请先换一部视频或修复文件。")
            return
        blocked_ids = {did for did, _ in blocked}
        ids = [did for did in ids if did not in blocked_ids]
        stages = self._selected_stages()
        if getattr(self, "_guide_locked", False) and self._wanted_locked_stages():
            locked, reason = self.service.replace_locked()
            if not stages:
                QMessageBox.information(self, "先完成人物审核", reason + "\n\n点左侧第 ③ 步去人物库处理完待审核项。")
                return
            # 其余阶段照常跑，只把锁住的两个跳过，写进日志不再弹框打断。
            self.service.logs.append("WARNING", reason + "已跳过「人物替换 / 按剧本生成」，先跑其余阶段。")
            # N25：光写日志不够——用户点完按钮要在界面上立刻看到哪个
            # 阶段没跑。状态栏写一句被跳过的阶段名。
            if self.all_stages.isChecked():
                skipped_names = ["人物替换", "按剧本生成"]
            else:
                skipped_names = []
                if self.stage_buttons.get("replace") is not None and self.stage_buttons["replace"].isChecked():
                    skipped_names.append("人物替换")
                if self.stage_buttons.get("generate") is not None and self.stage_buttons["generate"].isChecked():
                    skipped_names.append("按剧本生成")
            skipped_label = " / ".join(skipped_names) if skipped_names else "人物替换 / 按剧本生成"
            self.statusBar().showMessage(
                f"已跳过锁住的「{skipped_label}」（先完成第 ③ 步审核）"
            )
        if not stages:
            QMessageBox.information(self, "按勾选阶段执行", "请至少勾选一个底部阶段，或打开「全部阶段按序执行」。")
            return
        # N23：手动入口也走同一授权关口——关口跟的是“替换”这个动作，
        # 不是“一键成片”这个按钮。锁住时替换已被上面 strip 掉，不会误拦。
        if "replace" in stages:
            if not self._auth_gate_ok(prompt=True, drama_ids=ids):
                return
        try:
            self._bind_lipsync_confirm()
            self.service.start(ids, stages)
        except Exception as exc:
            QMessageBox.warning(self, "无法开始", str(exc))
            return
        if blocked:
            names = []
            for did, reason in blocked:
                drama = self.service.store.get("dramas", did) or {}
                names.append(
                    f"「{drama.get('new_name') or drama.get('origin_name') or did}」：{reason}"
                )
            QMessageBox.information(
                self,
                "已跳过预检没通过的剧目",
                f"跳过了 {len(blocked)} 部：\n" + "\n".join(names) + "\n其余剧目已开始。",
            )
        self.refresh(full=True)

    def _reface(self) -> None:
        locked, reason = self.service.replace_locked()
        if locked:
            QMessageBox.information(self, "先完成人物审核", reason + "\n\n点左侧第 ③ 步去人物库处理完待审核项。")
            return
        ids = self._drama_ids_checked()
        if not ids:
            QMessageBox.information(self, "只换脸", "请先勾选要换脸的剧目。")
            return
        # N23（P3）：[只换脸]也是替换入口，进替换前先走同一授权关口
        # （_auth_gate_ok，与一键成片 / 手动入口共用；零已授权时 [继续]
        # 灰掉，见 N18；被跳过的人物由关口记 _record_skipped_characters）。
        if not self._auth_gate_ok(prompt=True, drama_ids=ids):
            return
        try:
            self.service.start(ids, ["replace"])
        except Exception as exc:
            QMessageBox.warning(self, "无法开始换脸", str(exc))
            return
        self.refresh(full=True)

    def _body_rerender(self) -> None:
        """身体重渲染试点：选 ≤5 秒单人片段 + GLB 模型，工作线程跑全链。"""
        from recast.stages.rerender import clip_duration_s, validate_pilot_clip

        clip, _ = QFileDialog.getOpenFileName(
            self, "选择 5 秒以内的单人片段", "", "视频 (*.mp4 *.mov *.mkv)"
        )
        if not clip:
            return
        glb, _ = QFileDialog.getOpenFileName(
            self, "选择 3D 人物模型（GLB）", "", "GLB 模型 (*.glb)"
        )
        if not glb:
            return
        try:
            duration = clip_duration_s(clip)
            validate_pilot_clip(duration, people=1)
        except Exception as exc:
            QMessageBox.information(self, "身体重渲染（试点）", str(exc))
            return
        single = QMessageBox.question(
            self,
            "身体重渲染（试点）",
            "请确认：这个片段里是否只有一个人？\n\n试点只支持单人片段；多人镜头点“否”会直接拒绝。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if single != QMessageBox.StandardButton.Yes:
            try:
                validate_pilot_clip(duration, people=2)
            except Exception as exc:
                QMessageBox.information(self, "身体重渲染（试点）", str(exc))
            return
        self.rerender_btn.setEnabled(False)
        self.rerender_btn.setText("身体重渲染中…")

        def work() -> None:
            payload: dict = {}
            try:
                result = self.service.run_body_rerender(clip, glb)
                payload["video"] = result["out_path"]
                payload["frames"] = result["frames"]
            except Exception as exc:
                payload["error"] = str(exc)
            self._rerender_signal.ready.emit(payload)

        threading.Thread(target=work, name="recast-ui-rerender", daemon=True).start()

    def _apply_rerender_result(self, payload: dict) -> None:
        self.rerender_btn.setEnabled(True)
        self.rerender_btn.setText("身体重渲染（试点）")
        if payload.get("error"):
            QMessageBox.warning(self, "身体重渲染（试点）", payload["error"])
            return
        QMessageBox.information(
            self,
            "身体重渲染（试点）",
            f"完成：{payload.get('frames')} 帧\n成片：{payload.get('video')}",
        )
        self.refresh(full=True)

    def _export_jianying(self) -> None:
        ids = self._drama_ids_checked()
        if not ids:
            QMessageBox.information(self, "导出剪映", "请先勾选要导出剪映工程的剧目。")
            return
        try:
            self.service.start(ids, ["export"])
        except Exception as exc:
            QMessageBox.warning(self, "无法开始导出剪映", str(exc))
            return
        self.refresh(full=True)

    def _stop(self) -> None:
        self._oneclick = None
        self.oneclick_label.setText("一键成片已停止。")
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

    def _open_queue(self) -> None:
        """打开队列面板（G3 / 9.2 的主窗口接线）。

        import 方向只能是主窗口 → queue_panel（queue_panel 不反向
        import 主窗口，见 queue_panel.py 的模块 docstring）。
        """
        from recast.desktop.queue_panel import QueuePanelDialog

        dlg = QueuePanelDialog(self.service, parent=self)
        self._queue_panel = dlg
        dlg.exec()

    def _open_shot_enhance(self) -> None:
        """逐镜头增强（N24 的图形入口）：列出当前剧目的全部镜头，逐行开关。

        开关走 ``service.set_shot_enhance``，记在分集工作目录的
        shots_enhance.json 里，和终检窗口同一份 flags。只读预览之外的
        唯一写操作就是开关本身；没有镜头 / 没有分集任务时只提示，不建
        任务、不改文件。两条路都支持：导入路逐镜头增强，剧本路在拼接后
        按标记时段增强（runner 拼接后调 enhance_script_output，N24）。
        对话框本体在 ``recast.desktop.shot_enhance_dialog``（R6 拆分，
        P3 第 3 波）；这里只留选剧、空态提示和打开。``_shot_enhance_dialog``
        / ``_shot_enhance_boxes`` 为兼容保留（现有测试按这两个属性找
        对话框和复选框）。
        """
        from recast.desktop.shot_enhance_dialog import (
            ShotEnhanceDialog,
            collect_shot_enhance_rows,
        )

        drama_id = self._current_drama_id()
        if not drama_id:
            QMessageBox.information(self, "逐镜头增强", "请先勾选或选中一部剧。")
            return
        tasks = self.service.store.tasks_for_drama(drama_id)
        if not tasks:
            QMessageBox.information(self, "逐镜头增强", "这部剧还没有分集任务（先扫描导入）。")
            return
        rows = collect_shot_enhance_rows(self.service, drama_id)
        if not rows:
            QMessageBox.information(
                self, "逐镜头增强", "这部剧还没有镜头记录（先跑「解析提取」，切镜后才有镜头）。"
            )
            return
        drama = self.service.store.get("dramas", drama_id) or {}
        name = drama.get("new_name") or drama.get("origin_name") or drama_id
        dlg = ShotEnhanceDialog(self.service, drama_id, rows, drama_name=name, parent=self)
        self._shot_enhance_boxes = dlg.boxes
        self._shot_enhance_dialog = dlg
        dlg.exec()

    def _open_history(self) -> None:
        """打开任务历史面板（只打开，不碰面板内部逻辑；复跑执行在面板里走 service.rerun_task）。"""
        from recast.desktop.task_history_dialog import TaskHistoryDialog

        TaskHistoryDialog(self.service, parent=self).exec()

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
        locked, reason = self.service.replace_locked()
        if locked:
            QMessageBox.information(self, "先完成人物审核", reason + "\n\n点左侧第 ③ 步去人物库处理完待审核项。")
            return
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
