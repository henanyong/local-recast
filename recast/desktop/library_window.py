"""Character library: confirm names, authorize, inspect face samples side by side."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QIcon, QImage, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QDialog,
)

from recast.core.character_assets import resolve_character_assets
from recast.core.characters import list_character_folders
from recast.core.constants import AUTH_TYPES, ASSET_LABELS
from recast.core.review import (
    filter_review_items,
    group_review_items,
    high_confidence_items,
    score_sample_file,
)
from recast.core.review_progress import review_progress_stats
from recast.core.service import RecastService, asset_flags
from recast.desktop.cluster_compare import ClusterCompareDialog
from recast.desktop.dialogue_dialog import DialogueDialog
from recast.desktop.timbre_dialog import TimbreDialog


MODEL_SOURCE_LABELS = {
    "manual": "手动导入",
    "auto": "自动生成",
    "face_card": "脸卡占位",
    "none": "无",
    "": "未知",
}


def model_source_label(source: str) -> str:
    """Human label for a meta.json model.source value."""
    return MODEL_SOURCE_LABELS.get(str(source or ""), str(source or "未知"))


def auth_flag_text(ch: dict | None) -> str:
    """Row flag for one character: authorization + its type."""
    if not ch:
        return "未授权"
    if ch.get("authorized"):
        auth_type = ch.get("auth_type") or ""
        return f"已授权（{auth_type}）" if auth_type else "已授权（未选类型）"
    return "未授权"


def replace_block_reason(ch: dict | None, name: str = "") -> str | None:
    """Why this character cannot be replaced yet, or None when it can.

    Replace requires authorization AND an authorization type (the replace
    stage enforces both and parks the task otherwise). Saying it here, in
    the library, is the upfront version of that same check.
    """
    if ch is None:
        return None
    who = f"「{name}」" if name else "该人物"
    if not ch.get("authorized"):
        return f"{who}未授权，不可替换：先在右侧勾选「已授权」并选择授权类型。"
    if not ch.get("auth_type"):
        return f"{who}已授权但未选授权类型，不可替换：先在右侧选择授权类型。"
    return None


def _format_file_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def describe_glb_model(path: str | Path) -> dict[str, Any]:
    """Inspect a GLB file: triangle count, skeleton, textures, size.

    Pure function (no Qt): validates the glTF 2.0 header, then reads the
    JSON chunk. Returns {"ok": True, "triangles": int | None,
    "skeleton_count": int, "texture_count": int, "size": int} or
    {"ok": False, "error": str}. A file that fails validation reports
    ok=False instead of raising, so the detail card can show the reason.
    """
    import struct

    from recast.assets3d.glb import validate_glb

    path = Path(path)
    if not path.is_file():
        return {"ok": False, "error": f"找不到模型文件：{path}"}
    try:
        validate_glb(path)
    except Exception as exc:  # noqa: BLE001 - reported in the card, not raised
        return {"ok": False, "error": str(exc)}
    try:
        with path.open("rb") as fh:
            _magic, _version, _length = struct.unpack("<4sII", fh.read(12))
            chunk_length, _chunk_type = struct.unpack("<II", fh.read(8))
            payload = json.loads(fh.read(chunk_length) or b"null")
    except (OSError, ValueError, struct.error) as exc:
        return {"ok": False, "error": f"GLB 的 JSON 块读不懂：{exc}"}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "GLB 的 JSON 块不是对象。"}
    accessors = payload.get("accessors") or []
    triangles: int | None = 0
    for mesh in payload.get("meshes") or []:
        for prim in (mesh or {}).get("primitives") or []:
            try:
                if "indices" in prim:
                    count = int(accessors[int(prim["indices"])]["count"])
                else:
                    pos = int((prim.get("attributes") or {})["POSITION"])
                    count = int(accessors[pos]["count"])
                triangles += count // 3
            except (KeyError, IndexError, TypeError, ValueError):
                triangles = None
                break
        if triangles is None:
            break
    return {
        "ok": True,
        "triangles": triangles,
        "skeleton_count": len(payload.get("skins") or []),
        "texture_count": len(payload.get("images") or []),
        "size": path.stat().st_size,
    }


_SWAP_PREVIEW_HINT = (
    "取选中的一张人脸样本（没选就取第一张样本/抠像），用当前装配的脸换上去，"
    "弹窗并排对照原图和预览。没装 InsightFace 时是参考脸对齐叠加预览，会明确标出来。"
)


class _TaskSignal(QObject):
    finished = Signal(object)


class _StatusLabel(QLabel):
    """底部状态栏：换行显示，长文报出真实需要的高度。

    放在 widgetResizable 的 QScrollArea 里时，普通 QLabel 的
    minimumSizeHint 很小，滚动区会把它压到视口高度、长文被夹掉还不出
    滚动条。这里按当前宽度报 heightForWidth()，滚动条就正常出现了。
    """

    def minimumSizeHint(self) -> QSize:
        if self.wordWrap() and self.width() > 0:
            return QSize(0, self.heightForWidth(self.width()))
        return super().minimumSizeHint()


def _sample_paths_for(ch: dict) -> list[str]:
    """人物字典里的 face 资产路径列表（样本对照、审核行计低分都用它）。"""
    return [a.get("path") for a in (ch.get("assets") or []) if a.get("kind") == "face" and a.get("path")]


class LibraryWindow(QDialog):
    def __init__(self, service: RecastService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.setWindowTitle("人物库")
        self.resize(1280, 800)
        self.setMinimumSize(1024, 600)
        self._current_id: str | None = None
        self._selected_samples: list[str] = []  # 纠错时勾中的样本文件路径
        # P0–P3 管线状态：导入的源文件、任务是否在跑（测试时可置 _sync_tasks=True 同步跑）。
        self._source_path: str | None = None
        self._task_running = False
        self._sync_tasks = False
        # 3D 生成队列对话框（P2 第二波）：复用同一个对话框实例，重复点
        # [生成 3D 模型] 只往同一队列里加行。_model3d_generate_fn 只给
        # 测试注入假后端，正常走 service.generate_model_3d。
        self._model3d_queue_dialog_obj = None
        self._model3d_generate_fn = None
        self._task_signal = _TaskSignal()
        self._task_signal.finished.connect(self._on_task_finished)

        self.list = QListWidget()
        self.list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.list.currentItemChanged.connect(self._on_select)

        self.name_edit = QLineEdit()
        self.auth = QCheckBox("已授权，可用于人物替换")
        self.auth_type = QComboBox()
        self.auth_type.addItem("（未选择）", "")
        for t in AUTH_TYPES:
            self.auth_type.addItem(t, t)
        self.flags = QLabel()
        self.flags.setWordWrap(True)
        self.samples = QWidget()
        self.samples_layout = QHBoxLayout(self.samples)
        self.samples_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.samples)
        scroll.setMinimumHeight(220)

        save = QPushButton("保存确认")
        save.setObjectName("primary")
        save.clicked.connect(self._save)
        add_face = QPushButton("指定参考脸…")
        add_face.setObjectName("secondary")
        add_face.setToolTip("从电脑里选一张图，设为这个人物的参考脸")
        add_face.clicked.connect(self._pick_face)
        add_char = QPushButton("手动添加人物…")
        add_char.clicked.connect(self._add_manual)
        self.l2_btn = QPushButton("生成头像卡片（占位）")
        self.l2_btn.setObjectName("secondary")
        self.l2_btn.setToolTip(
            "生成头像卡片（占位）：关键点 + 漫反射贴图 + 驱动配置，不是生产级完整网格；需要先选一个已授权人物。"
        )
        self.l2_btn.clicked.connect(self._build_l2)
        self.l2_del_btn = QPushButton("删除头像卡片")
        self.l2_del_btn.setObjectName("danger")
        self.l2_del_btn.setToolTip("删除头像卡片（占位）：只删该人物的头像卡片文件，人脸照片保留。")
        self.l2_del_btn.clicked.connect(self._delete_l2)
        clean_l2 = QPushButton("清理未使用的头像卡片")
        clean_l2.setObjectName("quiet")
        clean_l2.setToolTip(
            "清理未使用的头像卡片：只删 L2 目录内未被任何人物使用的头像卡片，不删人脸、服装、声音。"
        )
        clean_l2.clicked.connect(self._clean_unused_l2)
        self.delete_char_btn = QPushButton("删除人物")
        self.delete_char_btn.setObjectName("danger")
        self.delete_char_btn.setEnabled(False)
        self.delete_char_btn.clicked.connect(self._delete_character)
        clean_samples = QPushButton("清理失效样本")
        clean_samples.setObjectName("quiet")
        clean_samples.setToolTip("清理引用文件已不存在的失效样本（只清记录，不删文件）。")
        clean_samples.clicked.connect(self._clean_invalid_samples)

        # ===== 左栏：操作清单（不编号；全产品只留主窗口引导 ①–⑤ 这一套编号）+ 人物列表 =====
        steps_group = QGroupBox("操作清单")
        steps_layout = QVBoxLayout(steps_group)
        self.step_labels: dict[str, QLabel] = {}
        for key, title in (
            ("import", "导入视频 / 照片"),
            ("define", "自动定义人物"),
            ("folder", "建文件夹 / 抠像"),
            ("model", "3D 模型"),
            ("voice", "声音"),
        ):
            step_row = QHBoxLayout()
            step_row.addWidget(QLabel(title))
            state = QLabel("○ 未开始")
            state.setStyleSheet("color:#8b90a0;")
            state.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            step_row.addWidget(state, 1)
            steps_layout.addLayout(step_row)
            self.step_labels[key] = state
        bind_step_row = QHBoxLayout()
        bind_step_row.addWidget(QLabel("绑定服装到模型"))
        bind_state_label = QLabel("○ 未绑定")
        bind_state_label.setStyleSheet("color:#6c7080;")
        bind_state_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bind_step_row.addWidget(bind_state_label, 1)
        steps_layout.addLayout(bind_step_row)
        self.bind_step_label = bind_state_label

        self.import_btn = QPushButton("导入视频/照片")
        self.import_btn.clicked.connect(self._import_source)
        self.auto_define_btn = QPushButton("自动定义人物")
        self.auto_define_btn.setEnabled(False)
        self.auto_define_btn.setToolTip("先导入视频/照片。一次跑完：说话人分离切声音、名字建议、衣服打标签；人脸聚类请在主界面跑「03 人物提取与建库」。")
        self.auto_define_btn.clicked.connect(self._auto_define)
        self.migrate_btn = QPushButton("迁移旧资产到人物文件夹")
        self.migrate_btn.setToolTip("把旧的 face_samples / L2 脸卡 / roster 服装声音拷进各人物文件夹，只读源文件，可重复跑。")
        self.migrate_btn.clicked.connect(self._migrate_folders)
        steps_layout.addWidget(self.import_btn)
        steps_layout.addWidget(self.auto_define_btn)
        steps_layout.addWidget(self.migrate_btn)
        self.source_label = QLabel("未导入视频/照片。")
        self.source_label.setWordWrap(True)
        self.source_label.setStyleSheet("color:#8b90a0;")
        steps_layout.addWidget(self.source_label)

        # ===== 人物列表搜索 + 状态筛选（PM §二 2.1/M2）=====
        # 搜索框按名筛，筛选下拉按状态筛；筛后空了提示去审核队列。
        self.character_search = QLineEdit()
        self.character_search.setPlaceholderText("搜索人物名字…")
        self.character_search.setClearButtonEnabled(True)
        self.character_search.textChanged.connect(self._apply_character_filter)
        self.character_status_filter = QComboBox()
        self.character_status_filter.addItem("全部状态", "all")
        self.character_status_filter.addItem("未确认", "unconfirmed")
        self.character_status_filter.addItem("无模型", "no_model")
        self.character_status_filter.addItem("无声音", "no_voice")
        self.character_status_filter.addItem("未授权", "unauthorized")
        self.character_status_filter.setToolTip("按状态筛人物列表")
        self.character_status_filter.currentIndexChanged.connect(self._apply_character_filter)
        search_row = QHBoxLayout()
        search_row.addWidget(self.character_search, 1)
        search_row.addWidget(self.character_status_filter)

        self.status_label = _StatusLabel()
        self.status_label.setWordWrap(True)

        left = QVBoxLayout()
        left.addWidget(steps_group)
        left.addLayout(search_row)
        left.addWidget(QLabel("人物列表（聚类候选需确认姓名）"))
        left.addWidget(self.list, 1)
        self.character_empty_label = QLabel("没有符合条件的人物，去审核队列看看。")
        self.character_empty_label.setWordWrap(True)
        self.character_empty_label.setStyleSheet("color:#8b90a0;")
        self.character_empty_label.setVisible(False)
        left.addWidget(self.character_empty_label)
        left.addWidget(add_char)

        # ===== 第 ③ 步内部切换：左右翻人物，逐个审核，不用点回列表 =====
        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 上一个")
        self.prev_btn.clicked.connect(self._prev_character)
        self.next_btn = QPushButton("下一个 ▶")
        self.next_btn.clicked.connect(self._next_character)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.next_btn)
        left.addLayout(nav_row)

        # ===== 人工审核队列：不清楚的聚类 + 低分样本，都在这里处理 =====
        # 215 项审得动的设计：先筛（类型/分数/分组）再按组折叠，一次只看一组；
        # 目标人物是带缩略图的列表（点谁移给谁），不是小下拉框。
        review_group = QGroupBox("人工审核队列（第 ③ 步）")
        review_layout = QVBoxLayout(review_group)
        # 审核进度（PM §三 N22）：与主窗口同一份数，分母按 N17 修复规则
        # （已确认 + 待审核全部项，聚类 + 低分样本都算），见 _review_progress_stats。
        self.review_progress = QProgressBar()
        self.review_progress.setRange(0, 100)
        self.review_progress.setVisible(False)
        self.review_progress_label = QLabel("审核进度：还没有人物。")
        self.review_progress_label.setWordWrap(True)
        review_layout.addWidget(self.review_progress)
        review_layout.addWidget(self.review_progress_label)
        self.review_count_label = QLabel("待审核：0 项")
        review_layout.addWidget(self.review_count_label)
        filter_row = QHBoxLayout()
        self.review_kind_filter = QComboBox()
        self.review_kind_filter.addItem("全部类型", None)
        self.review_kind_filter.addItem("聚类", "cluster")
        self.review_kind_filter.addItem("样本", "sample")
        self.review_kind_filter.setToolTip("按类型筛：未确认的聚类 / 低分或移出的样本")
        self.review_kind_filter.currentIndexChanged.connect(self._refresh_review)
        self.review_score_filter = QComboBox()
        self.review_score_filter.addItem("全部分数", "all")
        self.review_score_filter.addItem("分数 < 0.3", "low")
        self.review_score_filter.addItem("分数 0.3–0.5", "mid")
        self.review_score_filter.setToolTip("按样本分数筛（聚类行没有分数，不会被分数筛藏起来）")
        self.review_score_filter.currentIndexChanged.connect(self._refresh_review)
        filter_row.addWidget(self.review_kind_filter, 1)
        filter_row.addWidget(self.review_score_filter, 1)
        review_layout.addLayout(filter_row)
        group_row = QHBoxLayout()
        group_row.addWidget(QLabel("分组："))
        self.review_group_filter = QComboBox()
        self.review_group_filter.addItem("全部分组", None)
        self.review_group_filter.setToolTip("一次只看一个聚类/人物的组，分批处理")
        self.review_group_filter.currentIndexChanged.connect(self._refresh_review)
        group_row.addWidget(self.review_group_filter, 1)
        review_layout.addLayout(group_row)
        # B0：被挡下的低质量样本默认不进队列（见 build_review_queue 的
        # include_rejected）；勾选后重新刷新，把它们显示出来。
        self.review_show_rejected = QCheckBox("显示低质量样本")
        self.review_show_rejected.setChecked(False)
        self.review_show_rejected.setToolTip(
            "默认隐藏提取时被挡下的样本（模糊/太小/侧脸/检测分过低）；勾选后它们重新出现在队列里"
        )
        self.review_show_rejected.toggled.connect(self._refresh_review)
        review_layout.addWidget(self.review_show_rejected)
        self.review_list = QListWidget()
        self.review_list.setMinimumHeight(110)
        self.review_list.setMaximumHeight(140)
        self.review_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.review_list.currentItemChanged.connect(self._on_review_select)
        self.review_list.itemClicked.connect(self._on_review_item_clicked)
        self.review_list.itemDoubleClicked.connect(self._open_cluster_compare)
        self.review_list.itemSelectionChanged.connect(self._update_review_buttons)
        review_layout.addWidget(self.review_list)
        review_layout.addWidget(QLabel("移到哪个人物（点谁就移给谁）："))
        self.review_target_list = QListWidget()
        self.review_target_list.setMinimumHeight(72)
        self.review_target_list.setMaximumHeight(120)
        self.review_target_list.setIconSize(QSize(40, 40))
        self.review_target_list.setToolTip("点一行就是目标人物；高亮边框的是当前选中的目标")
        self.review_target_list.setStyleSheet(
            "QListWidget::item:selected { border: 2px solid #7ddea0; background: #2a3345; }"
        )
        review_layout.addWidget(self.review_target_list)
        assign_row = QHBoxLayout()
        self.review_assign_btn = QPushButton("分给已有的人物")
        self.review_assign_btn.setObjectName("secondary")
        self.review_assign_btn.clicked.connect(self._review_assign)
        assign_row.addWidget(self.review_assign_btn)
        review_layout.addLayout(assign_row)
        batch_row = QHBoxLayout()
        self.review_batch_assign_btn = QPushButton("批量分给选中人物")
        self.review_batch_assign_btn.setObjectName("secondary")
        self.review_batch_assign_btn.setToolTip("把队列里选中的多项一起分给上面点中的目标人物")
        self.review_batch_assign_btn.clicked.connect(self._review_batch_assign)
        self.review_batch_create_btn = QPushButton("批量新建人物")
        self.review_batch_create_btn.setObjectName("secondary")
        self.review_batch_create_btn.setToolTip("选中的多项各建一个人物（聚类用自己的名字，样本用前缀+序号）")
        self.review_batch_create_btn.clicked.connect(self._review_batch_create)
        self.review_batch_discard_btn = QPushButton("批量丢弃")
        self.review_batch_discard_btn.setObjectName("danger")
        self.review_batch_discard_btn.setToolTip("把队列里选中的多项一起丢弃（文件进 review_trash，不真删）")
        self.review_batch_discard_btn.clicked.connect(self._review_batch_discard)
        batch_row.addWidget(self.review_batch_assign_btn)
        batch_row.addWidget(self.review_batch_create_btn)
        batch_row.addWidget(self.review_batch_discard_btn)
        review_layout.addLayout(batch_row)
        review_btn_row = QHBoxLayout()
        self.review_create_btn = QPushButton("新建人物…")
        self.review_create_btn.setObjectName("secondary")
        self.review_create_btn.clicked.connect(self._review_create)
        self.review_discard_btn = QPushButton("丢弃")
        self.review_discard_btn.setObjectName("danger")
        self.review_discard_btn.clicked.connect(self._review_discard)
        review_btn_row.addWidget(self.review_create_btn)
        review_btn_row.addWidget(self.review_discard_btn)
        review_layout.addLayout(review_btn_row)
        self.review_confirm_high_btn = QPushButton("一键确认全部高置信度")
        self.review_confirm_high_btn.setObjectName("secondary")
        self.review_confirm_high_btn.setToolTip(
            "只确认有清楚代表脸的聚类行（代表脸清晰度 ≥ 审核阈值 review_min_score）；样本行、低分聚类不确认"
        )
        self.review_confirm_high_btn.clicked.connect(self._review_confirm_high)
        review_layout.addWidget(self.review_confirm_high_btn)
        self.finish_review_btn = QPushButton("完成审核，去替换")
        self.finish_review_btn.setObjectName("primary")
        self.finish_review_btn.clicked.connect(self._finish_review)
        review_layout.addWidget(self.finish_review_btn)
        # 刷新时用的状态：完整队列、筛后队列、已折叠的分组。
        self._review_all_items: list[dict] = []
        self._review_filtered_items: list[dict] = []
        self._review_collapsed: set[tuple[str, str]] = set()
        left.addWidget(review_group)
        left_pane = QWidget()
        left_pane.setLayout(left)
        left_pane.setMinimumWidth(250)
        # 左栏现在很高（管线 + 列表 + 审核队列的筛选/目标列表/批量行）：
        # 包一层滚动区，800 高下也不会把审核组切在窗口底边外面。
        left_scroll = QScrollArea()
        left_scroll.setObjectName("pipelinePane")
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_pane)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # ===== 中间：预览舞台（卡片式，不再是纯文字）=====
        # 详情区控件先建出来，舞台和检查器共用同一份数据。
        self.detail_label = QLabel()
        self.detail_label.setWordWrap(True)
        # N27：详情卡缺项直达按钮（每选一次人物就按缺项重建，见 _rebuild_detail_buttons）。
        self.detail_action_buttons: list[QPushButton] = []
        self.detail_images = QWidget()
        self.detail_images_layout = QHBoxLayout(self.detail_images)
        self.detail_images_layout.setContentsMargins(0, 0, 0, 0)
        detail_scroll = QScrollArea()
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setWidget(self.detail_images)
        detail_scroll.setMinimumHeight(120)

        stage = QWidget()
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(4, 4, 4, 4)

        # ===== 装配区：人物 3D 预览 + 脸 / 身体 / 衣服三行点选 =====
        assembly_group = QGroupBox("人物装配（点选即应用，不用再确认）")
        assembly_layout = QVBoxLayout(assembly_group)
        self.assembly_name = QLabel("请从左侧选择人物。")
        self.assembly_name.setStyleSheet("font-weight:bold;")
        assembly_layout.addWidget(self.assembly_name)
        self.assembly_preview = QLabel("请从左侧选择人物。")
        self.assembly_preview.setFixedSize(280, 280)
        self.assembly_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.assembly_preview.setWordWrap(True)
        self.assembly_preview.setStyleSheet("border:1px solid #3a3f4d;")
        assembly_layout.addWidget(self.assembly_preview, alignment=Qt.AlignmentFlag.AlignCenter)
        self.assembly_face_list = self._assembly_row("脸（face/ 样本）")
        assembly_layout.addWidget(QLabel("脸"))
        assembly_layout.addWidget(self.assembly_face_list)
        self.assembly_face_empty = self._assembly_empty_label(
            "没有人脸样本：先跑主界面「03 人物提取与建库」，或点右侧「指定参考脸…」。"
        )
        assembly_layout.addWidget(self.assembly_face_empty)
        self.assembly_body_list = self._assembly_row("身体（本地模型库的 GLB，含换装变体：点一个变体就是换上那身衣服）")
        assembly_layout.addWidget(QLabel("身体"))
        assembly_layout.addWidget(self.assembly_body_list)
        self.assembly_body_empty = self._assembly_empty_label(
            "没有 3D 模型：点右侧「3D 模型」→「生成 3D 模型」或「导入 3D 模型」。"
        )
        assembly_layout.addWidget(self.assembly_body_empty)
        self.assembly_body_detail = QLabel("点身体行看模型详情。")
        self.assembly_body_detail.setWordWrap(True)
        self.assembly_body_detail.setStyleSheet("border:1px solid #3a3f4d; padding:6px;")
        assembly_layout.addWidget(self.assembly_body_detail)
        self.assembly_costume_list = self._assembly_row("衣服（本地衣柜）")
        assembly_layout.addWidget(QLabel("衣服"))
        assembly_layout.addWidget(self.assembly_costume_list)
        self.assembly_costume_empty = self._assembly_empty_label(
            "衣柜是空的：先把衣服图片放进人物文件夹 wardrobe/，或点右侧「衣服打标签…」。"
        )
        assembly_layout.addWidget(self.assembly_costume_empty)
        self.assembly_summary = QLabel("当前装配：—")
        self.assembly_summary.setWordWrap(True)
        assembly_layout.addWidget(self.assembly_summary)
        self._preview_attempted_for: str | None = None
        # Pick rows: connect once here; _fill_assembly_row only rebuilds items.
        for row, slot in (
            (self.assembly_face_list, "face"),
            (self.assembly_body_list, "body"),
            (self.assembly_costume_list, "costume"),
        ):
            row.setProperty("assembly_slot", slot)
            row.itemClicked.connect(self._on_assembly_pick)
        self.assembly_body_list.currentItemChanged.connect(self._on_body_detail_select)
        stage_layout.addWidget(assembly_group)

        # ===== 审核一眼信息：模型怎么样、抠像怎么样、样本有没有低分 =====
        self.audit_label = QLabel("审核：请从左侧选择人物。")
        self.audit_label.setWordWrap(True)
        self.audit_label.setStyleSheet("font-weight:bold;")
        stage_layout.addWidget(self.audit_label)

        face_group = QGroupBox("人脸样本（并排对照，便于确认是否同一人）")
        face_layout = QVBoxLayout(face_group)
        face_layout.addWidget(scroll, 1)
        preview_row = QHBoxLayout()
        self.swap_preview_btn = QPushButton("替换预览")
        self.swap_preview_btn.setToolTip(_SWAP_PREVIEW_HINT)
        self.swap_preview_btn.clicked.connect(self._open_swap_preview)
        preview_row.addWidget(self.swap_preview_btn)
        preview_row.addStretch(1)
        face_layout.addLayout(preview_row)
        correct_row = QHBoxLayout()
        self.select_all_samples_btn = QPushButton("全选样本")
        self.select_all_samples_btn.setToolTip("脸样本可以多选：全选后下面的移到/移出/拆分对选中的全部生效")
        self.select_all_samples_btn.clicked.connect(self._toggle_select_all_samples)
        self.correct_move_btn = QPushButton("移到另一人物…")
        self.correct_move_btn.setToolTip("把选中的样本移到另一人物：文件搬过去，两边 meta.json 和数据库记录都更新")
        self.correct_move_btn.clicked.connect(self._correct_move)
        self.correct_remove_btn = QPushButton("移出到待审核")
        self.correct_remove_btn.setToolTip("把选中的样本从这个人物移出，放回人工审核队列（不删文件，删错了找得回来）")
        self.correct_remove_btn.clicked.connect(self._correct_remove)
        self.correct_split_btn = QPushButton("拆分为新人物…")
        self.correct_split_btn.setToolTip("这个人物文件夹里混进了别人：把选中的样本一次性移到新建人物")
        self.correct_split_btn.clicked.connect(self._correct_split)
        correct_row.addWidget(self.select_all_samples_btn)
        correct_row.addWidget(self.correct_move_btn)
        correct_row.addWidget(self.correct_remove_btn)
        correct_row.addWidget(self.correct_split_btn)
        face_layout.addLayout(correct_row)
        stage_layout.addWidget(face_group, 1)
        media_group = QGroupBox("服装 / 抠像（第一张）")
        media_layout = QVBoxLayout(media_group)
        media_layout.addWidget(detail_scroll, 1)
        stage_layout.addWidget(media_group, 1)
        model_group = QGroupBox("3D 模型")
        model_layout = QVBoxLayout(model_group)
        self.model_card_label = QLabel("请从左侧选择人物。")
        self.model_card_label.setWordWrap(True)
        model_layout.addWidget(self.model_card_label)
        stage_layout.addWidget(model_group)
        detail_group = QGroupBox("人物文件夹详情")
        detail_layout = QVBoxLayout(detail_group)
        detail_layout.addWidget(self.detail_label)
        self.detail_buttons_layout = QHBoxLayout()
        detail_layout.addLayout(self.detail_buttons_layout)
        stage_layout.addWidget(detail_group, 1)
        stage_scroll = QScrollArea()
        stage_scroll.setObjectName("stagePane")
        stage_scroll.setWidgetResizable(True)
        stage_scroll.setWidget(stage)
        stage_scroll.setMinimumWidth(320)

        # ===== 右栏：检查器（按步骤分组，包一层滚动区）=====
        # 角色（P5/B3 §5.2）：[保存确认] 是检查器唯一 primary；其余检查器
        # 按钮都是 secondary（危险按钮除外），每组最多 1 个 primary。
        self.build_folder_btn = QPushButton("建人物文件夹")
        self.build_folder_btn.setObjectName("secondary")
        self.build_folder_btn.clicked.connect(self._build_folder)
        self.matte_btn = QPushButton("抠像")
        self.matte_btn.setObjectName("secondary")
        self.matte_btn.clicked.connect(self._matte)
        self.gen3d_btn = QPushButton("生成 3D 模型")
        self.gen3d_btn.setObjectName("secondary")
        self.gen3d_btn.clicked.connect(self._generate_3d)
        self.gen_variant_btn = QPushButton("生成换装变体…")
        self.gen_variant_btn.setObjectName("secondary")
        self.gen_variant_btn.clicked.connect(self._generate_variant)
        self.switch_variant_btn = QPushButton("切换换装变体…")
        self.switch_variant_btn.setObjectName("secondary")
        self.switch_variant_btn.clicked.connect(self._switch_variant)
        self.import_glb_btn = QPushButton("导入 3D 模型")
        self.import_glb_btn.setObjectName("secondary")
        self.import_glb_btn.clicked.connect(self._import_glb)
        self.tag_btn = QPushButton("衣服打标签…")
        self.tag_btn.setObjectName("secondary")
        self.tag_btn.clicked.connect(self._tag_costume)
        self.audition_btn = QPushButton("试听声音")
        self.audition_btn.setObjectName("secondary")
        self.audition_btn.clicked.connect(self._audition)
        self.timbre_btn = QPushButton("调节音色")
        self.timbre_btn.setObjectName("secondary")
        self.timbre_btn.clicked.connect(self._open_timbre)
        self.dialogue_btn = QPushButton("编辑对话")
        self.dialogue_btn.setObjectName("secondary")
        self.dialogue_btn.clicked.connect(self._open_dialogue)
        self.bind_btn = QPushButton("绑定服装到模型")
        self.bind_btn.setObjectName("secondary")
        self.bind_btn.clicked.connect(self._bind_costume)
        self.bind_btn.setToolTip(
            "把当前装配的衣服登记到 GLB 模型（全身部位，登记 + 校验）。"
            "贴图本身不写进 GLB 二进制：Windows 上用 Blender（导入 GLB → 换材质 Base Color 贴图 → 导出）或 pygltflib 完成最后一步。"
        )
        # 设定板与换装预览走 2D 路线（A+B）：不替换身体，只把角色的标准
        # 参考图和配色固化下来。按钮放进已有行，检查器高度不再增长。
        self.sheet_btn = QPushButton("生成设定板")
        self.sheet_btn.setObjectName("secondary")
        self.sheet_btn.clicked.connect(self._build_sheet)
        self.garment_btn = QPushButton("换装预览…")
        self.garment_btn.setObjectName("secondary")
        self.garment_btn.clicked.connect(self._preview_garment)

        inspector = QWidget()
        insp = QVBoxLayout(inspector)
        insp.setContentsMargins(4, 4, 4, 4)
        insp.setSpacing(2)

        def _compact(layout) -> None:
            """检查器各组收紧边距/间距：1280×800 下第 4 组按钮要首屏可见。"""
            layout.setContentsMargins(6, 4, 6, 6)
            layout.setSpacing(3)

        info_group = QGroupBox("基本信息")
        info_layout = QVBoxLayout(info_group)
        _compact(info_layout)
        # N20：别名（meta.json aliases，曾用名/别名列表）可查看、可手动加。
        # 别名行与“显示名称”标题同行、[加别名…] 并进下方操作行：检查器在
        # 1280×800 下刚好一屏，别名单独占两行会把内容顶出 42px、出现滚动条
        #（test_inspector_group4_fully_visible_at_1280x800）。别名很长时会
        # 换行增高，属于数据本身变长，滚动条兜底。
        self.aliases_label = QLabel("别名：—")
        self.aliases_label.setWordWrap(True)
        self.aliases_label.setStyleSheet("color:#8b90a0;")
        name_title_row = QHBoxLayout()
        name_title_row.addWidget(QLabel("显示名称"))
        name_title_row.addWidget(self.aliases_label, 1)
        info_layout.addLayout(name_title_row)
        info_layout.addWidget(self.name_edit)
        auth_row = QHBoxLayout()
        auth_row.addWidget(self.auth)
        auth_row.addWidget(QLabel("授权类型"))
        self.auth_type.setToolTip("自有拍摄 / 已授权 / 自有生成角色")
        auth_row.addWidget(self.auth_type, 1)
        info_layout.addLayout(auth_row)
        info_layout.addWidget(self.flags)
        self.add_alias_btn = QPushButton("加别名…")
        self.add_alias_btn.setToolTip("给这个人物加一个曾用名/别名，记进人物文件夹 meta.json 的 aliases")
        self.add_alias_btn.clicked.connect(self._add_alias)
        info_row = QHBoxLayout()
        info_row.addWidget(save)
        info_row.addWidget(add_face)
        info_row.addWidget(self.add_alias_btn)
        info_row.addStretch(1)
        info_layout.addLayout(info_row)
        insp.addWidget(info_group)

        g1 = QGroupBox("1 建文件夹")
        g1_layout = QVBoxLayout(g1)
        _compact(g1_layout)
        g1_layout.addWidget(self.build_folder_btn)
        insp.addWidget(g1)

        g2 = QGroupBox("2 抠像")
        g2_layout = QVBoxLayout(g2)
        _compact(g2_layout)
        g2_layout.addWidget(self.matte_btn)
        insp.addWidget(g2)

        g3 = QGroupBox("3 3D 模型")
        g3_layout = QVBoxLayout(g3)
        _compact(g3_layout)
        # 三行小标题：真模型 / 占位资产 / 变体（P5/B3 §5.2）。每行内按钮
        # 等宽等高（Expanding + 同一 stretch）；文案过长时样式省略显示，
        # 全称放 tooltip。变体行只有一个按钮，后面补 2 份 stretch，让它
        # 和上面两行的按钮同宽，不拉满整行。
        for _title, _btns in (
            ("真模型", (self.gen3d_btn, self.import_glb_btn, self.switch_variant_btn)),
            ("占位资产", (self.l2_btn, self.l2_del_btn, clean_l2)),
            ("变体 / 设定板", (self.gen_variant_btn, self.sheet_btn)),
        ):
            _sub = QLabel(_title)
            _sub.setStyleSheet("color:#8b90a0; font-size:12px;")
            g3_layout.addWidget(_sub)
            _row = QHBoxLayout()
            _row.setSpacing(3)
            for _b in _btns:
                _b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                _b.setMinimumHeight(32)
                _row.addWidget(_b, 1)
            if len(_btns) < 3:
                _row.addStretch(3 - len(_btns))
            g3_layout.addLayout(_row)
        insp.addWidget(g3)

        g4 = QGroupBox("4 声音与服装")
        g4_layout = QGridLayout(g4)
        _compact(g4_layout)
        for i, w in enumerate((self.tag_btn, self.audition_btn, self.timbre_btn, self.dialogue_btn)):
            g4_layout.addWidget(w, i // 2, i % 2)
        insp.addWidget(g4)

        g5 = QGroupBox("5 应用")
        g5_row = QHBoxLayout(g5)
        _compact(g5_row)
        g5_row.setSpacing(3)
        for _b in (self.bind_btn, self.garment_btn):
            _b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            _b.setMinimumHeight(32)
            g5_row.addWidget(_b, 1)
        insp.addWidget(g5)

        # 检查器按步骤折叠（P5/B3 §5.2）：g1–g5 是可折叠的 QGroupBox，
        # checked = 展开；info_group（基本信息）不参与折叠，始终展开。
        # 展开状态跟随当前选中人物的管线状态，见 _apply_inspector_expansion。
        self._step_groups = {"g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5}
        for _grp in self._step_groups.values():
            _grp.setCheckable(True)
            _grp.toggled.connect(self._on_step_group_toggled)

        gm = QGroupBox("管理")
        gm_layout = QHBoxLayout(gm)
        _compact(gm_layout)
        gm_layout.addWidget(self.delete_char_btn)
        gm_layout.addWidget(clean_samples)
        insp.addWidget(gm)

        # 长说明默认折叠（原来 QToolBox 默认展开，占 ~150px，顶掉了第 4 组）。
        self.hint_label = QLabel(
            "L1「人物替换」是本地换脸，不是提取了3D模型。"
            "L2「头像卡片（占位）」是关键点 + 漫反射贴图 + 驱动配置，不是生产级完整网格；未放入官方 FLAME 时也不会假装是。"
            "质量偏低会标成「偏低」。同一人物 ID 可跨两集复用。未勾选授权的人物不能替换，也不能生成头像卡片（占位）。"
        )
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color:#8b90a0;")
        self.hint_label.setVisible(False)
        self.hint_toggle = QPushButton()
        self.hint_toggle.setObjectName("quiet")
        self.hint_toggle.setText("说明：L1 / L2 是什么 ▸")
        self.hint_toggle.setCheckable(True)
        self.hint_toggle.setChecked(False)
        self.hint_toggle.toggled.connect(self._toggle_hint)
        insp.addWidget(self.hint_toggle)
        insp.addWidget(self.hint_label)
        insp.addStretch(1)

        inspector_scroll = QScrollArea()
        inspector_scroll.setObjectName("inspectorPane")
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setWidget(inspector)
        inspector_scroll.setMinimumWidth(280)

        # ===== 三栏拼起来：全部 QSplitter，用户能拖 =====
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("librarySplitter")
        splitter.addWidget(left_scroll)
        splitter.addWidget(stage_scroll)
        splitter.addWidget(inspector_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([300, 560, 340])

        # 底部状态栏：长文（比如 TripoSG 准备步骤）换行显示，放不下就滚，
        # 不再只给一行高度把字截掉。
        self.status_scroll = QScrollArea()
        self.status_scroll.setObjectName("statusScroll")
        self.status_scroll.setWidgetResizable(True)
        self.status_scroll.setWidget(self.status_label)
        self.status_scroll.setFixedHeight(48)
        self.status_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.status_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.status_scroll.setFrameShape(QFrame.Shape.NoFrame)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.status_scroll)
        self.reload()

    def _sync_folder_embeddings(self) -> None:
        """Mirror store face_embeddings into folder meta.json files.

        The pipeline writes embeddings to the store but never passes
        output_root, so without this the folder mirror silently never
        happens (and folder-level merge has nothing to compare against).
        Silent no-op when the service has no store; failures are logged,
        never raised — a mirror problem must not break the library view.
        """
        from recast.core.merge import sync_folder_embeddings

        store = getattr(self.service, "store", None)
        output_root = getattr(getattr(self.service, "settings", None), "output_root", None)
        if store is None or output_root is None:
            return
        try:
            report = sync_folder_embeddings(store, output_root)
        except Exception as exc:  # noqa: BLE001 - mirror must not break reload
            logging.getLogger("recast").warning("同步人物文件夹 embedding 失败：%s", exc)
            return
        if report.get("error"):
            logging.getLogger("recast").warning("同步人物文件夹 embedding 跳过：%s", report["error"])

    def reload(self) -> None:
        self._sync_folder_embeddings()
        current = self._current_id
        self.list.clear()
        for ch in self.service.characters():
            name = ch.get("display_name") or ch.get("name") or ch["id"]
            flags = []
            if ch.get("confirmed"):
                flags.append("已确认")
            flags.append(auth_flag_text(ch))
            item = QListWidgetItem(f"{name}  ·  {' / '.join(flags)}{self._folder_status_text(ch)}")
            item.setToolTip(item.text())  # 行太长只显示省略号，悬停看全文
            if not ch.get("authorized"):
                # 未授权的行置灰：一眼看出这个人物现在不可替换。
                item.setForeground(QBrush(QColor("#8b90a0")))
            item.setData(Qt.ItemDataRole.UserRole, ch["id"])
            # 搜索/状态筛选用的信息（UserRole+1）：名字 + 确认/授权/模型/声音。
            item.setData(Qt.ItemDataRole.UserRole + 1, self._character_filter_info(ch))
            self.list.addItem(item)
            if current and ch["id"] == current:
                self.list.setCurrentItem(item)
        if self.list.count() and self.list.currentItem() is None:
            self.list.setCurrentRow(0)
        self.delete_char_btn.setEnabled(self.list.currentItem() is not None)
        self._refresh_action_states()
        self._refresh_review()
        self._apply_character_filter()

    def _character_filter_info(self, ch: dict) -> dict:
        """搜索/状态筛选用的每人物信息：名字 + 确认/授权/模型/声音。

        模型/声音读人物文件夹解析结果（和 _folder_status_text 同一个源）；
        读不到就当没有，不让筛选把人物库弄崩。
        """
        name = ch.get("display_name") or ch.get("name") or ch["id"]
        info = {
            "name": str(name),
            "confirmed": bool(ch.get("confirmed")),
            "authorized": bool(ch.get("authorized")),
            "has_model": False,
            "has_voice": False,
        }
        try:
            output_root = getattr(getattr(self.service, "settings", None), "output_root", None)
            store = getattr(self.service, "store", None)
            if output_root is None:
                return info
            assets = resolve_character_assets(output_root, str(name), store=store)
        except Exception:
            return info
        if assets is not None:
            info["has_model"] = assets.model_source != "none"
            info["has_voice"] = bool(assets.voice_samples)
        return info

    def _apply_character_filter(self) -> None:
        """按搜索框（名字子串）+ 状态筛选下拉，藏起不符合的人物行。

        筛后一行都不剩时提示去审核队列；当前选中的行被筛掉时，自动选
        第一行可见的，不让选中态停在一个藏起来的行上。
        """
        search = getattr(self, "character_search", None)
        status_filter = getattr(self, "character_status_filter", None)
        if search is None or status_filter is None:
            return
        query = (search.text() or "").strip().casefold()
        status = status_filter.currentData() or "all"
        visible = 0
        for row in range(self.list.count()):
            item = self.list.item(row)
            info = item.data(Qt.ItemDataRole.UserRole + 1) or {}
            name = str(info.get("name") or item.text())
            ok = (not query) or query in name.casefold()
            if ok and status != "all":
                if status == "unconfirmed":
                    ok = not info.get("confirmed")
                elif status == "no_model":
                    ok = not info.get("has_model")
                elif status == "no_voice":
                    ok = not info.get("has_voice")
                elif status == "unauthorized":
                    ok = not info.get("authorized")
            item.setHidden(not ok)
            if ok:
                visible += 1
        self.character_empty_label.setVisible(self.list.count() > 0 and visible == 0)
        current = self.list.currentItem()
        if current is not None and current.isHidden():
            for row in range(self.list.count()):
                if not self.list.item(row).isHidden():
                    self.list.setCurrentRow(row)
                    return

    def _folder_status_text(self, ch: dict) -> str:
        """人物文件夹状态，拼进列表行：衣服 / 声音 / 3D（读不到就留空）。"""
        try:
            assets = resolve_character_assets(
                self.service.settings.output_root,
                ch.get("display_name") or ch.get("name") or ch["id"],
                store=self.service.store,
            )
        except Exception:
            return ""
        if assets is None:
            return "  ·  未建人物文件夹"
        costume = "有描述" if assets.costume_description else "未打标签"
        voice = f"{len(assets.voice_samples)} 条声音" if assets.voice_samples else "无声音"
        if assets.model_source == "none":
            model = "无 3D 模型"
        elif assets.model_source == "face_card":
            model = "3D：脸卡占位"
        elif assets.model_format == "glb":
            model = f"3D：GLB（{assets.model_source}）"
        else:
            model = f"3D：{assets.model_source}"
        return f"  ·  衣服：{costume}  ·  {voice}  ·  {model}"

    def _on_select(self, item: QListWidgetItem | None) -> None:
        if item is None:
            self._current_id = None
            self.delete_char_btn.setEnabled(False)
            self.name_edit.clear()
            self.auth.setChecked(False)
            self.auth_type.setCurrentIndex(0)
            self.flags.clear()
            self._show_samples({"assets": []}, empty="请从左侧选择人物。")
            self._show_detail(None)
            self._refresh_assembly()
            self._refresh_action_states()
            return
        self.delete_char_btn.setEnabled(True)
        cid = item.data(Qt.ItemDataRole.UserRole)
        self._current_id = cid
        ch = next((c for c in self.service.characters() if c["id"] == cid), None)
        if not ch:
            return
        self.name_edit.setText(ch.get("display_name") or ch.get("name") or "")
        self.auth.setChecked(bool(ch.get("authorized")))
        idx = self.auth_type.findData(ch.get("auth_type") or "")
        self.auth_type.setCurrentIndex(max(0, idx))
        flags = asset_flags(ch, output_root=self.service.settings.output_root)
        parts = [f"{ASSET_LABELS.get(k, k)}：{v}" for k, v in flags.items()]
        self.flags.setText("资料完整度    " + "    ".join(parts))
        self._show_samples(ch)
        self._show_detail(ch)
        self._refresh_assembly()
        self._refresh_action_states()

    def _current_character_name(self) -> str | None:
        ch = self._current_character()
        if not ch:
            return None
        return ch.get("display_name") or ch.get("name") or ch["id"]

    def _current_character(self) -> dict | None:
        if not self._current_id:
            return None
        return next((c for c in self.service.characters() if c["id"] == self._current_id), None)

    def _current_assets(self):
        """选中人物的人物文件夹解析结果，没有文件夹时为 None。"""
        name = self._current_character_name()
        if not name:
            return None
        try:
            return resolve_character_assets(
                self.service.settings.output_root, name, store=self.service.store
            )
        except Exception:
            return None

    def _refresh_action_states(self) -> None:
        """按管线顺序亮按钮：跑完一步出结果后，下一步才亮。

        未选人 → 全部人物操作灰掉；没有文件夹 → 只能建文件夹；
        没有抠像 → [生成 3D 模型] 灰掉；[绑定服装到模型] 要先在装配区
        选一件衣服（没衣服可绑就灰掉，tooltip 写清缺什么）。
        检查器折叠状态也在这里跟随管线状态（见 _apply_inspector_expansion）。
        """
        has_selection = self._current_id is not None
        assets = self._current_assets() if has_selection else None
        self._apply_inspector_expansion(assets)
        self._refresh_inspector_buttons()
        self._update_steps(assets)
        self._refresh_correction_states()
        self._refresh_swap_preview_state()
        self._ensure_disabled_tooltips()

    def _refresh_inspector_buttons(self) -> None:
        """检查器按钮的启用态 + tooltip（不碰折叠状态）。

        单独抽出来：QGroupBox 手动勾选/取消勾选时 Qt 会自动启用/禁用
        子控件，toggled 处理里调这里把管线状态重新断言一遍，不让手动
        展开把本该灰掉的按钮点亮。g3 三行里文案较长的按钮，tooltip 前
        缀全称（行内太窄时样式会省略显示，悬停看全称）。
        """
        has_selection = self._current_id is not None
        assets = self._current_assets() if has_selection else None
        has_folder = assets is not None
        has_cutout = bool(assets and assets.cutouts)
        busy = self._task_running
        self.auto_define_btn.setEnabled(
            self._source_path is not None and not busy
        )
        self.build_folder_btn.setEnabled(has_selection and not has_folder and not busy)
        self.build_folder_btn.setToolTip(
            "已经建好人物文件夹了" if has_folder else "在 output/characters/<名字>/ 下建人物文件夹"
        )
        self.matte_btn.setEnabled(has_folder and not busy)
        self.matte_btn.setToolTip("" if has_folder else "先建人物文件夹")
        self.gen3d_btn.setEnabled(has_cutout and not busy)
        self.gen3d_btn.setToolTip(
            "生成 3D 模型："
            + (
                "用第一张抠像自动生成 GLB（需 CUDA 显卡 + TripoSG；只出形状，不带贴图、不带骨骼，不能直接用于身体重渲染）。"
                "首次使用先按准备步骤装好 TripoSG：克隆仓库 → conda python 3.10 → 装 CUDA 版 torch → 先跑 example 图 → 配置里填 triposg_repo / triposg_python。"
                if has_cutout else "先抠像（cutout/ 为空）"
            )
        )
        self.gen_variant_btn.setEnabled(has_folder and not busy)
        self.gen_variant_btn.setToolTip(
            "生成换装变体…："
            + (
                "用一张穿着这身衣服的全身照生成一个着装变体（整身 GLB，不是独立衣片；也要 CUDA 显卡 + TripoSG）"
                if has_folder else "先建人物文件夹"
            )
        )
        try:
            from recast.assets3d.variants import list_costume_variants

            has_variants = bool(assets and list_costume_variants(assets.folder))
        except Exception:
            has_variants = False
        self.switch_variant_btn.setEnabled(has_folder and has_variants and not busy)
        self.switch_variant_btn.setToolTip(
            "切换换装变体…："
            + (
                "在主模型和已生成的换装变体之间切换（切换就是把装配的身体换成对应变体的 GLB）"
                if has_variants else "还没有换装变体，先点「生成换装变体…」"
            )
        )
        self.import_glb_btn.setEnabled(has_folder and not busy)
        self.import_glb_btn.setToolTip(
            "导入 3D 模型：从电脑里选一个 GLB 文件导入" if has_folder else "先建人物文件夹"
        )
        self.tag_btn.setEnabled(has_folder and not busy)
        self.tag_btn.setToolTip("" if has_folder else "先建人物文件夹")
        self.audition_btn.setEnabled(has_folder and not busy)
        self.audition_btn.setToolTip("" if has_folder else "先建人物文件夹")
        self.timbre_btn.setEnabled(has_folder and not busy)
        self.timbre_btn.setToolTip("" if has_folder else "先建人物文件夹")
        self.dialogue_btn.setEnabled(has_folder and not busy)
        self.dialogue_btn.setToolTip("" if has_folder else "先建人物文件夹")
        bind_ok, bind_tip = self._bind_availability()
        self.bind_btn.setEnabled(bind_ok and not busy)
        self.bind_btn.setToolTip(bind_tip)
        sheet_ok, sheet_tip = self._sheet_availability()
        self.sheet_btn.setEnabled(sheet_ok and not busy)
        self.sheet_btn.setToolTip(sheet_tip)
        garment_ok, garment_tip = self._garment_availability()
        self.garment_btn.setEnabled(garment_ok and not busy)
        self.garment_btn.setToolTip(garment_tip)
        # [删除人物] 的启用态由 reload/_on_select 管，这里只补 tooltip：
        # 没选人时灰掉，必须写清缺什么。
        self.delete_char_btn.setToolTip(
            "" if self.delete_char_btn.isEnabled() else "先在左侧选一个人。"
        )

    def _on_step_group_toggled(self, checked: bool) -> None:
        """用户手动勾选/取消勾选检查器步骤组：只收放内容，不改管线状态。

        QGroupBox 取消勾选会自动禁用子控件、勾选时又全放开，所以收放完
        立刻调 _refresh_inspector_buttons() 把管线启用态重新断言一遍；
        下次 _refresh_action_states 还会按管线状态重新定展开组。
        """
        group = self.sender()
        if group is None:
            return
        self._set_step_group_expanded(group, checked)
        self._refresh_inspector_buttons()
        self._ensure_disabled_tooltips()

    @staticmethod
    def _set_step_group_expanded(group: QGroupBox, expanded: bool) -> None:
        """收起/展开一个步骤组：隐藏/显示布局里的全部子控件（含 g3 的小标题行）。

        只靠 setCheckable 的勾选态收不起来（子控件还在布局里占高度），
        必须把子控件逐个隐藏，组盒才缩到只剩标题栏。收起时顺手把几何
        清零：按钮创建时没有父控件、默认是 640×480，从未参与布局就被
        隐藏时会一直留着这个假几何，读几何的旧测试会误判。
        重展开时布局会重新分配几何，清零没有副作用。
        """
        layout = group.layout()
        if layout is None:
            return

        def _walk(lay) -> None:
            for i in range(lay.count()):
                item = lay.itemAt(i)
                widget = item.widget()
                if widget is not None:
                    widget.setVisible(expanded)
                    if not expanded:
                        widget.resize(0, 0)
                elif item.layout() is not None:
                    _walk(item.layout())

        _walk(layout)

    def _inspector_expansion(self, assets) -> dict[str, bool]:
        """当前管线状态下该展开哪一组：单步展开，只展开当前这一步。

        未选人 / 没建文件夹 → g1；有文件夹、没抠像、且还没真模型 → g2；
        有抠像、没真模型（没有或只是头像卡片占位）→ g3；有真模型、没声音 → g4；
        都有了 → g5（绑定服装到模型）。info_group 始终展开，不参与。
        """
        if assets is None:
            current = "g1"
        elif not assets.cutouts and assets.model_source in ("none", "face_card"):
            current = "g2"
        elif assets.model_source in ("none", "face_card"):
            current = "g3"
        elif not assets.voice_samples:
            current = "g4"
        else:
            current = "g5"
        return {key: key == current for key in ("g1", "g2", "g3", "g4", "g5")}

    def _apply_inspector_expansion(self, assets) -> None:
        """按管线状态定 g1–g5 的勾选态（checked = 展开）。

        已经是目标态就不碰 setChecked，少一次无用的 toggled 信号。
        用户手动展开的组会在下次刷新时按状态复位，这是设计如此。
        """
        groups = getattr(self, "_step_groups", None)
        if not groups:
            return
        for key, expanded in self._inspector_expansion(assets).items():
            group = groups.get(key)
            if group is not None and group.isChecked() != expanded:
                group.setChecked(expanded)

    def _ensure_disabled_tooltips(self) -> None:
        """兜底：窗口里所有禁用按钮必须有非空 tooltip，写清缺什么。

        各处主逻辑已经给常见禁用原因写了具体 tooltip；这里只补漏——
        比如任务进行中临时灰掉的按钮、或可折叠组收起时 QGroupBox 自动
        禁用的子控件。兜底文案是通用句，不替代具体原因。
        """
        for btn in self.findChildren(QPushButton):
            if not btn.isEnabled() and not (btn.toolTip() or "").strip():
                btn.setToolTip("暂不可用：先完成前面的步骤。")

    def _refresh_swap_preview_state(self) -> None:
        """[替换预览] 只在选了人、且有样本/抠像可当采集帧时才亮。"""
        btn = getattr(self, "swap_preview_btn", None)
        if btn is None:
            return
        has_source = False
        if self._current_id is not None:
            assets = self._current_assets()
            has_source = bool(assets and (assets.face_files or assets.cutouts))
        reason = self._current_preview_block_reason()
        btn.setEnabled(has_source and reason is None and not self._task_running)
        if reason is not None:
            btn.setToolTip(reason)
        elif not has_source:
            btn.setToolTip("先选一个有人脸样本或抠像的人物。")
        else:
            btn.setToolTip(_SWAP_PREVIEW_HINT)

    def _update_steps(self, assets) -> None:
        """左栏操作清单指示器：当前做到哪一步一眼能看见。

        导入、自动定义看全局；建文件夹/抠像、3D 模型、声音看选中人物。
        """
        labels = getattr(self, "step_labels", None)
        if not labels:
            return

        def _set(key: str, state: str, text: str) -> None:
            lab = labels.get(key)
            if lab is None:
                return
            lab.setText(f"{state} {text}")
            color = {"✓": "#7ddea0", "◐": "#ffd56a"}.get(state, "#8b90a0")
            lab.setStyleSheet(f"color:{color};")

        _set("import", "✓" if self._source_path else "○", "已导入" if self._source_path else "未开始")
        try:
            folders = list_character_folders(self.service.settings.output_root)
        except Exception:
            folders = []
        _set("define", "✓" if folders else "○", "已定义" if folders else "未开始")
        # 绑定服装行：已在 meta.json 登记过绑定就是 ✓，否则 ○ 未绑定。
        bind_label = getattr(self, "bind_step_label", None)
        if bind_label is not None:
            bound = False
            if assets is not None:
                try:
                    from recast.assets3d.binding import list_bindings

                    bound = bool(list_bindings(assets.folder))
                except Exception:
                    bound = False
            bind_label.setText("✓ 已绑定" if bound else "○ 未绑定")
            bind_label.setStyleSheet(f"color:{'#7ddea0' if bound else '#8b90a0'};")
        if assets is None:
            _set("folder", "○", "未开始")
            _set("model", "○", "未开始")
            _set("voice", "○", "未开始")
            return
        if assets.cutouts:
            _set("folder", "✓", "已抠像")
        else:
            _set("folder", "◐", "已建文件夹")
        if assets.model_source == "none":
            _set("model", "○", "未开始")
        elif assets.model_source == "face_card":
            _set("model", "◐", "脸卡占位")
        else:
            _set("model", "✓", f"已有模型（{assets.model_source}）")
        if assets.voice_samples:
            _set("voice", "✓", f"{len(assets.voice_samples)} 条声音")
        else:
            _set("voice", "○", "未开始")

    # ----- 第 ③ 步：人物切换 + 人工审核队列 + 纠错 -----

    def _prev_character(self) -> None:
        self._step_character(-1)

    def _next_character(self) -> None:
        self._step_character(1)

    def _step_character(self, delta: int) -> None:
        """人物列表就是切换器：上一个/下一个，循环翻，逐个审核。

        被搜索/状态筛选藏起来的行跳过，不停在看不见的人物上。
        """
        if self.list.count() == 0:
            return
        row = self.list.currentRow()
        for _ in range(self.list.count()):
            row = (row + delta) % self.list.count()
            if not self.list.item(row).isHidden():
                self.list.setCurrentRow(row)
                return

    def _refresh_review(self) -> None:
        """重建人工审核队列：未确认聚类 + 低分样本；空了第 ③ 步才算完成。

        215 项的批处理：先按类型/分数/分组筛，再按（类型，人物）分组折叠
        显示——组头点一下就折起/展开；总数 > 20 时默认只展开第一组。
        只有一组时不画组头，列表行数就是数据行数（旧测试依赖这一点）。
        """
        items = self.service.list_review_queue(
            include_rejected=self.review_show_rejected.isChecked()
        )
        self._review_all_items = items
        kind = self.review_kind_filter.currentData()
        band = self.review_score_filter.currentData() or "all"
        filtered = filter_review_items(
            items, kinds={kind} if kind else None, score_band=band
        )
        # 分组下拉框的 userData 是 JSON 字符串（QComboBox.findData 比对元组不可靠）。
        raw_group = self.review_group_filter.currentData()
        group_key = tuple(json.loads(raw_group)) if raw_group else None
        group_options = group_review_items(filtered)
        if group_key is not None and group_key not in {k for k, _t, _g in group_options}:
            group_key = None  # 这个组被类型/分数筛筛没了，分组筛选跟着放掉
        if group_key is not None:
            filtered = [i for i in filtered if (i.get("kind"), i.get("character_name")) == group_key]
        self._review_filtered_items = filtered
        total = len(items)
        shown = len(filtered)
        count_text = f"待审核：{total} 项"
        if shown != total:
            count_text += f"（已筛出 {shown} 项）"
        if not items:
            count_text += "（已清空，第 ③ 步完成）"
        self.review_count_label.setText(count_text)

        # 分组下拉框：筛后还剩哪些组（选中的组保留，信号先屏蔽防递归）。
        # 选项来自类型/分数筛后的列表（不是分组筛后的），否则选了一个组，
        # 别的组就从下拉框里消失了。
        self.review_group_filter.blockSignals(True)
        try:
            prev_group = group_key
            self.review_group_filter.clear()
            self.review_group_filter.addItem("全部分组", None)
            for key, title, _g in group_options:
                self.review_group_filter.addItem(
                    title, json.dumps(list(key), ensure_ascii=False)
                )
            if prev_group is not None:
                idx = self.review_group_filter.findData(
                    json.dumps(list(prev_group), ensure_ascii=False)
                )
                if idx >= 0:
                    self.review_group_filter.setCurrentIndex(idx)
        finally:
            self.review_group_filter.blockSignals(False)

        groups = group_review_items(filtered)
        show_headers = len(groups) > 1
        # 新出现的组默认折叠（总数 > 20 时，只展开第一组）；用户手动折过的组保持原样。
        seen = getattr(self, "_review_seen_groups", set())
        for index, (key, _title, _g) in enumerate(groups):
            if key not in seen:
                if shown > 20 and index > 0:
                    self._review_collapsed.add(key)
                seen.add(key)
        self._review_seen_groups = {key for key, _t, _g in groups}
        self._review_collapsed &= {key for key, _t, _g in groups}

        self.review_list.clear()
        for key, title, gitems in groups:
            collapsed = key in self._review_collapsed
            if show_headers:
                header = QListWidgetItem(
                    f"{'▶' if collapsed else '▼'} {title}（{len(gitems)}）"
                )
                header.setData(
                    Qt.ItemDataRole.UserRole,
                    {"review_header": True, "group_key": key},
                )
                header.setFlags(Qt.ItemFlag.ItemIsEnabled)
                font = header.font()
                font.setBold(True)
                header.setFont(font)
                self.review_list.addItem(header)
            for item in gitems:
                kind_text = "聚类" if item["kind"] == "cluster" else "样本"
                score = f" · 分数 {item['score']:.2f}" if item.get("score") is not None else ""
                text = f"{kind_text} · {item['character_name']} · {item['source']}{score}"
                li = QListWidgetItem(text)
                li.setToolTip(text)
                li.setData(Qt.ItemDataRole.UserRole, item)
                if item.get("thumb"):
                    pix = QPixmap(item["thumb"])
                    if not pix.isNull():
                        li.setIcon(QIcon(pix))
                self.review_list.addItem(li)
                li.setHidden(collapsed)  # 必须先 addItem 再 setHidden，否则是空操作
        self._refresh_review_targets()
        self.finish_review_btn.setEnabled(not items)
        self.finish_review_btn.setToolTip(
            "" if not items else f"还剩 {total} 项待审核，处理完才能去替换"
        )
        self._refresh_review_progress(items)
        self._update_review_buttons()

    def _review_progress_stats(self, items: list[dict] | None = None) -> dict:
        """审核进度数：已确认 / 待审核 / 聚类 / 低分（与主窗口同一份数）。

        分母按 N17 修复规则：total = 已确认 + 待审核全部项（聚类 +
        低分样本都算）。主窗口 _review_stats 同样走共用纯函数
        recast.core.review_progress.review_progress_stats，两边同一份数。
        """
        if items is None:
            try:
                items = self.service.list_review_queue()
            except Exception:
                items = []
        try:
            characters = self.service.characters()
        except Exception:
            characters = []
        # 计数走共用纯函数 recast.core.review_progress（R6 拆分，P3 第 3 波）；
        # 取数与返回的键集合不变（total/percent 都在里面，N17/N22 测试盯着）。
        return review_progress_stats(characters, items)

    def _refresh_review_progress(self, items: list[dict] | None = None) -> None:
        """人物库审核队列顶部的进度条 + 标签（PM §三 N22）。

        total=0（既没已确认人物也没待审核项）时条隐藏、写“还没有人物”；
        已确认 0 人但队列非空时条按 0% 显示、写“待审核 N 项，未确认人物”，
        不再像老分母那样把“有待审核项”报成“还没有人物”。
        """
        stats = self._review_progress_stats(items)
        confirmed, pending, low, total = (
            stats["confirmed"],
            stats["pending"],
            stats["low"],
            stats["total"],
        )
        if total <= 0:
            self.review_progress.setVisible(False)
            self.review_progress_label.setText("审核进度：还没有人物。")
            return
        self.review_progress.setVisible(True)
        self.review_progress.setValue(stats["percent"])
        if confirmed == 0:
            self.review_progress_label.setText(f"待审核 {pending} 项，未确认人物")
        else:
            self.review_progress_label.setText(
                f"审核进度：已确认 {confirmed} / 待审核 {pending}（低分 {low}）"
            )

    def _refresh_review_targets(self) -> None:
        """目标人物列表：每行脸缩略图 + 名字 + 状态，点谁就移给谁。

        复用人物库同一份数据（store.list_characters），不另起数据源。
        保留上次选中的目标；没选过时默认选第一个已确认人物（分给未确认的
        聚类自己没有意义），都没有就选第一行。
        """
        prev = self._selected_target_id()
        self.review_target_list.clear()
        characters = self.service.characters()
        for ch in characters:
            name = ch.get("display_name") or ch.get("name") or ch["id"]
            status = ("已确认" if ch.get("confirmed") else "未确认") + " · " + (
                "已授权" if ch.get("authorized") else "未授权"
            )
            li = QListWidgetItem(f"{name} · {status}")
            li.setToolTip(f"{name}（{ch['id']}）· {status}")
            li.setData(Qt.ItemDataRole.UserRole, ch["id"])
            thumb = None
            for a in ch.get("assets") or []:
                if a.get("kind") == "face" and a.get("path") and Path(str(a["path"])).is_file():
                    thumb = str(a["path"])
                    break
            if thumb:
                pix = QPixmap(thumb)
                if not pix.isNull():
                    li.setIcon(QIcon(pix))
            self.review_target_list.addItem(li)
        if self.review_target_list.count() == 0:
            return
        for row in range(self.review_target_list.count()):
            if self.review_target_list.item(row).data(Qt.ItemDataRole.UserRole) == prev:
                self.review_target_list.setCurrentRow(row)
                return
        for row in range(self.review_target_list.count()):
            ch = characters[row]
            if ch.get("confirmed"):
                self.review_target_list.setCurrentRow(row)
                return
        self.review_target_list.setCurrentRow(0)

    def _selected_target_id(self) -> str | None:
        item = getattr(self, "review_target_list", None)
        if item is None:
            return None
        current = item.currentItem()
        if current is None:
            return None
        return current.data(Qt.ItemDataRole.UserRole)

    def _selected_target_name(self) -> str:
        item = self.review_target_list.currentItem()
        if item is None:
            return ""
        return item.text().split(" · ")[0]

    @staticmethod
    def _is_review_header(item: QListWidgetItem | None) -> bool:
        if item is None:
            return False
        data = item.data(Qt.ItemDataRole.UserRole)
        return isinstance(data, dict) and bool(data.get("review_header"))

    def _selected_review_items(self) -> list[dict]:
        """多选选中的数据行（组头行不算）。"""
        out: list[dict] = []
        for item in self.review_list.selectedItems():
            if self._is_review_header(item):
                continue
            data = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(data, dict):
                out.append(data)
        return out

    def _selected_review_item(self) -> dict | None:
        item = self.review_list.currentItem()
        if item is None or self._is_review_header(item):
            selected = self._selected_review_items()
            return selected[0] if selected else None
        data = item.data(Qt.ItemDataRole.UserRole)
        return data if isinstance(data, dict) else None

    def _on_review_select(self, current, previous=None) -> None:
        """队列里选了一行：处理按钮才亮，没选就灰掉。"""
        self._update_review_buttons()

    def _on_review_item_clicked(self, item: QListWidgetItem) -> None:
        """点组头：折起/展开那一组；点数据行：只刷新按钮状态。"""
        data = item.data(Qt.ItemDataRole.UserRole)
        if not (isinstance(data, dict) and data.get("review_header")):
            return
        key = data["group_key"]
        if key in self._review_collapsed:
            self._review_collapsed.discard(key)
        else:
            self._review_collapsed.add(key)
        collapsed = key in self._review_collapsed
        for row in range(self.review_list.count()):
            row_item = self.review_list.item(row)
            row_data = row_item.data(Qt.ItemDataRole.UserRole)
            if isinstance(row_data, dict) and not row_data.get("review_header"):
                if (row_data.get("kind"), row_data.get("character_name")) == key:
                    row_item.setHidden(collapsed)
        grouped = group_review_items(self._review_filtered_items)
        count = len(next((g for k, _t, g in grouped if k == key), []))
        title = next((t for k, t, _g in grouped if k == key), key[1])
        item.setText(f"{'▶' if collapsed else '▼'} {title}（{count}）")

    def _open_cluster_compare(self, item: QListWidgetItem) -> None:
        """双击队列行：打开聚类对比视图（PM §三 3.2）。

        组头行不打开；视图里确认/拆分/丢弃成功后（Accepted）重载人物库。
        对话框是用户双击主动打开的，不是在 paint/定时器刷新里弹的模态框。
        """
        if item is None or self._is_review_header(item):
            return
        data = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            return
        dlg = ClusterCompareDialog(self.service, data, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.reload()

    def _update_review_buttons(self) -> None:
        """单项按钮看当前行，批量按钮看多选，一键确认看筛后有没有候选。"""
        has_current = self._selected_review_item() is not None
        self.review_assign_btn.setEnabled(has_current)
        self.review_assign_btn.setToolTip(
            "把选中的这一行分给上面点中的目标人物" if has_current else "先在人工审核队列里选一行。"
        )
        self.review_create_btn.setEnabled(has_current)
        self.review_create_btn.setToolTip(
            "为选中的这一行新建一个人物" if has_current else "先在人工审核队列里选一行。"
        )
        self.review_discard_btn.setEnabled(has_current)
        self.review_discard_btn.setToolTip(
            "丢弃选中的这一行（文件进 review_trash，不真删）" if has_current else "先在人工审核队列里选一行。"
        )
        selected = self._selected_review_items()
        for btn in (
            self.review_batch_assign_btn,
            self.review_batch_create_btn,
            self.review_batch_discard_btn,
        ):
            btn.setEnabled(bool(selected))
        try:
            min_score = self.service._review_min_score()
        except Exception:
            from recast.core.review import DEFAULT_REVIEW_MIN_SCORE

            min_score = DEFAULT_REVIEW_MIN_SCORE
        candidates = high_confidence_items(self._review_filtered_items, min_score=min_score)
        self._review_high_confidence = candidates
        self.review_confirm_high_btn.setEnabled(bool(candidates))
        if candidates:
            self.review_confirm_high_btn.setText(f"一键确认全部高置信度（{len(candidates)} 项）")
        else:
            self.review_confirm_high_btn.setText("一键确认全部高置信度")
        self._ensure_disabled_tooltips()

    def _review_assign(self) -> None:
        item = self._selected_review_item()
        if item is None:
            self.status_label.setText("先在人工审核队列里选一行。")
            return
        target = self._selected_target_id()
        if not target:
            self.status_label.setText("没有可分给的人物：先在目标列表里点一个人，或新建一个人物。")
            return
        if item.get("kind") == "cluster" and target == item.get("key"):
            self.status_label.setText("不能分给自己：换一个目标人物。")
            return
        try:
            result = self.service.review_assign(item["key"], target, owner_id=item.get("character_id"))
        except Exception as exc:
            self.status_label.setText(f"分配失败：{exc}")
            return
        self.status_label.setText(
            f"已分给「{self._selected_target_name()}」（{result.get('kind')}，移了 {len(result.get('moved', []))} 个样本）。"
        )
        self.reload()

    def _review_batch_assign(self) -> None:
        """批量分给某人：目标用目标列表里点中的那个人，操作前确认几项。"""
        items = self._selected_review_items()
        if not items:
            self.status_label.setText("先在人工审核队列里多选几行（按住 Ctrl / Shift 点选）。")
            return
        target = self._selected_target_id()
        if not target:
            self.status_label.setText("先在目标列表里点一个目标人物。")
            return
        names = "、".join(i["character_name"] for i in items[:5])
        if len(items) > 5:
            names += f" 等 {len(items)} 项"
        answer = QMessageBox.question(
            self,
            "批量分给已有的人物",
            f"把选中的 {len(items)} 项（{names}）一起分给「{self._selected_target_name()}」？\n"
            "聚类整组并过去，样本单张移过去；分完这些行就从队列里消失。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        ok, failed = 0, []
        for item in items:
            if item.get("kind") == "cluster" and target == item.get("key"):
                failed.append((item, "不能分给自己"))
                continue
            try:
                self.service.review_assign(item["key"], target, owner_id=item.get("character_id"))
                ok += 1
            except Exception as exc:
                failed.append((item, str(exc)))
        self.status_label.setText(
            f"批量分配完成：成功 {ok} 项" + (f"，失败 {len(failed)} 项：{failed[0][1]}" if failed else "。")
        )
        self.reload()

    def _review_batch_create(self) -> None:
        """批量新建人物：聚类用自己的名字，样本用前缀+序号；操作前确认几项。"""
        items = self._selected_review_items()
        if not items:
            self.status_label.setText("先在人工审核队列里多选几行（按住 Ctrl / Shift 点选）。")
            return
        default = next(
            (i["character_name"] for i in items if i.get("kind") == "cluster"),
            "人物",
        )
        prefix, ok = QInputDialog.getText(
            self,
            "批量新建人物",
            "新人物名字前缀（聚类用自己的名字，样本用“前缀+序号”）：",
            text=default,
        )
        if not ok:
            return
        prefix = (prefix or "").strip() or "人物"
        answer = QMessageBox.question(
            self,
            "批量新建人物",
            f"为选中的 {len(items)} 项各建一个人物？\n"
            f"聚类用自己的名字，样本依次叫 {prefix}1、{prefix}2……\n"
            "建完这些行就从队列里消失。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        done, failed = 0, []
        sample_no = 0
        for item in items:
            if item.get("kind") == "cluster":
                name = item["character_name"]
            else:
                sample_no += 1
                name = f"{prefix}{sample_no}"
            try:
                self.service.review_create_character(
                    item["key"], name, owner_id=item.get("character_id")
                )
                done += 1
            except Exception as exc:
                failed.append((item, str(exc)))
        self.status_label.setText(
            f"批量新建完成：成功 {done} 项" + (f"，失败 {len(failed)} 项：{failed[0][1]}" if failed else "。")
        )
        self.reload()

    def _review_batch_discard(self) -> None:
        """批量丢弃：文件进 review_trash，不真删；操作前确认几项。"""
        items = self._selected_review_items()
        if not items:
            self.status_label.setText("先在人工审核队列里多选几行（按住 Ctrl / Shift 点选）。")
            return
        names = "、".join(i["character_name"] for i in items[:5])
        if len(items) > 5:
            names += f" 等 {len(items)} 项"
        answer = QMessageBox.question(
            self,
            "批量丢弃",
            f"丢弃选中的 {len(items)} 项（{names}）？\n"
            "样本文件会移到 output/review_trash/，聚类只删数据库行，文件不动。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        done, failed = 0, []
        for item in items:
            try:
                self.service.review_discard(item["key"], owner_id=item.get("character_id"))
                done += 1
            except Exception as exc:
                failed.append((item, str(exc)))
        self.status_label.setText(
            f"批量丢弃完成：成功 {done} 项" + (f"，失败 {len(failed)} 项：{failed[0][1]}" if failed else "。")
        )
        self.reload()

    def _review_confirm_high(self) -> None:
        """一键确认全部高置信度：只确认有清楚代表脸的聚类行。

        规则见 recast/core/review.py high_confidence_items：聚类行 + 代表脸
        清晰度 >= review_min_score。确认就是把聚类标为已确认（不改名）。
        cluster_distance 本身没存进人物表，这里用代表脸清晰度作代理。
        """
        candidates = getattr(self, "_review_high_confidence", []) or high_confidence_items(
            self._review_filtered_items
        )
        if not candidates:
            self.status_label.setText("没有可一键确认的高置信度聚类。")
            return
        names = "、".join(i["character_name"] for i in candidates[:8])
        if len(candidates) > 8:
            names += f" 等 {len(candidates)} 项"
        answer = QMessageBox.question(
            self,
            "一键确认全部高置信度",
            f"确认以下 {len(candidates)} 个聚类（{names}）？\n"
            "只确认代表脸够清楚的聚类行，名字不变；样本行和模糊的聚类不会动，"
            "还留在队列里等人工处理。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        done, failed = 0, []
        for item in candidates:
            try:
                self.service.confirm_character(item["key"])
                done += 1
            except Exception as exc:
                failed.append((item, str(exc)))
        self.status_label.setText(
            f"已确认 {done} 个高置信度聚类" + (f"，失败 {len(failed)} 项" if failed else "。")
        )
        self.reload()

    def _review_create(self) -> None:
        item = self._selected_review_item()
        if item is None:
            self.status_label.setText("先在人工审核队列里选一行。")
            return
        name, ok = QInputDialog.getText(self, "新建人物", "新人物名字：")
        if not ok or not name.strip():
            return
        try:
            self.service.review_create_character(item["key"], name.strip(), owner_id=item.get("character_id"))
        except Exception as exc:
            self.status_label.setText(f"新建人物失败：{exc}")
            return
        self.status_label.setText(f"已为「{name.strip()}」新建人物。")
        self.reload()

    def _review_discard(self) -> None:
        item = self._selected_review_item()
        if item is None:
            self.status_label.setText("先在人工审核队列里选一行。")
            return
        answer = QMessageBox.question(
            self,
            "丢弃",
            f"丢弃这一项（{item['character_name']} · {item['source']}）？\n"
            "样本文件会移到 output/review_trash/，聚类只删数据库行，文件不动。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.service.review_discard(item["key"], owner_id=item.get("character_id"))
        except Exception as exc:
            self.status_label.setText(f"丢弃失败：{exc}")
            return
        self.status_label.setText("已丢弃（文件没删，在 review_trash/ 里）。")
        self.reload()

    def _current_replace_block_reason(self) -> str | None:
        """Why the selected character cannot go to replace yet, or None."""
        if self._current_id is None:
            return None
        ch = next(
            (c for c in self.service.characters() if c["id"] == self._current_id), None
        )
        return replace_block_reason(ch, self._current_character_name() or "")

    def _current_preview_block_reason(self) -> str | None:
        """Why [替换预览] is unavailable for the selected character (None = OK).

        The preview is a look, not a replace run: it only needs the
        character to be authorized. The missing-type case is already
        flagged on the row and the audit line ("已授权（未选类型），不可替换"),
        and the actual replace run (which does require a type) is gated
        where it starts.
        """
        ch = self._current_character()
        if ch is None or ch.get("authorized"):
            return None
        return replace_block_reason(ch, self._current_character_name() or ch["id"])

    def _finish_review(self) -> None:
        """第 ③ 步完成：队列已空，关掉人物库，回主窗口点「下一步」去替换。

        注意：这里只管审核队列，不拦未授权人物——关库只是导航，
        真正的替换在主窗口点「下一步」/替换时才跑，那里才是拦的点
        （不在本轮文件边界内）。人物库里已经把未授权写清了：
        列表行置灰 +「未授权」，审核行写「未授权，不可替换」。
        """
        if self.service.list_review_queue():
            self.status_label.setText("队列还没空：处理完所有待审核项才能去替换。")
            return
        self.accept()

    def _audit_text(self, ch: dict | None, assets) -> str:
        """第 ③ 步审核一眼信息：模型怎么样、抠像怎么样、样本有没有低分。"""
        if ch is None:
            return "审核：请从左侧选择人物。"
        if assets is None:
            return "审核：未建人物文件夹，先点「建人物文件夹」。"
        if assets.model_source == "none":
            model = "模型：无模型"
        elif assets.model_source == "face_card":
            model = "模型：脸卡占位（无 GLB）"
        elif assets.model_format == "glb":
            model = f"模型：GLB（{assets.model_source}）"
        else:
            model = f"模型：{assets.model_source}"
        cut = f"抠像：{len(assets.cutouts)} 张" if assets.cutouts else "抠像：未抠像"
        paths = _sample_paths_for(ch)
        low = 0
        threshold = self.service._review_min_score()
        for p in paths:
            try:
                if score_sample_file(p) < threshold:
                    low += 1
            except Exception:
                continue
        samples = f"样本：{len(paths)} 张" + (f"（{low} 张低分，见待审核）" if low else "")
        auth = auth_flag_text(ch)
        if not ch.get("authorized") or not ch.get("auth_type"):
            auth = f"{auth}，不可替换"
        return f"审核：{model} · {cut} · {samples} · 授权：{auth}"

    def _correct_move(self) -> None:
        """纠错：把选中的样本移到另一人物（文件搬过去，两边记录都更新）。"""
        if not self._current_id or not self._selected_samples:
            return
        names = [
            (ch.get("display_name") or ch.get("name") or ch["id"], ch["id"])
            for ch in self.service.characters()
            if ch["id"] != self._current_id
        ]
        if not names:
            self.status_label.setText("没有别的人物可移：先新建一个人物。")
            return
        label, ok = QInputDialog.getItem(
            self, "移到另一人物", f"把选中的 {len(self._selected_samples)} 张样本移到：",
            [n for n, _ in names], 0, False,
        )
        if not ok:
            return
        target = dict(names)[label]
        try:
            for path in list(self._selected_samples):
                self.service.move_face_sample(self._current_id, path, target)
        except Exception as exc:
            self.status_label.setText(f"移动失败：{exc}")
            self.reload()
            return
        self.status_label.setText(f"已把 {len(self._selected_samples)} 张样本移到「{label}」。")
        self.reload()

    def _correct_remove(self) -> None:
        """纠错：把选中的样本从这个人物移出，放回待审核队列（不删文件）。"""
        if not self._current_id or not self._selected_samples:
            return
        try:
            for path in list(self._selected_samples):
                self.service.remove_face_sample(self._current_id, path)
        except Exception as exc:
            self.status_label.setText(f"移出失败：{exc}")
            self.reload()
            return
        self.status_label.setText(f"已把 {len(self._selected_samples)} 张样本移出到待审核（文件未删）。")
        self.reload()

    def _correct_split(self) -> None:
        """纠错：这个人物文件夹里混进了别人，把选中的样本一次性拆到新建人物。"""
        if not self._current_id or not self._selected_samples:
            return
        name, ok = QInputDialog.getText(self, "拆分为新人物", "新人物名字（选中的样本会移过去）：")
        if not ok or not name.strip():
            return
        try:
            result = self.service.split_character(self._current_id, list(self._selected_samples), name.strip())
        except Exception as exc:
            self.status_label.setText(f"拆分失败：{exc}")
            return
        self.status_label.setText(f"已拆分出新人物「{name.strip()}」，移了 {len(result.get('moved', []))} 张样本。")
        self.reload()

    def _show_detail(self, ch: dict | None) -> None:
        """详情区：脸（上面已显示）、服装截图、抠像、3D 模型状态。

        3D 不渲染预览，只写状态和路径（没有 3D 渲染器，不造假图）。
        """
        while self.detail_images_layout.count():
            item = self.detail_images_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        if ch is None:
            self.detail_label.setText("请从左侧选择人物。")
            self.model_card_label.setText("请从左侧选择人物。")
            self.audit_label.setText("审核：请从左侧选择人物。")
            self._detail_placeholder("请从左侧选择人物。")
            self._rebuild_detail_buttons(None, None)
            self._refresh_aliases_label(None)
            return
        assets = self._current_assets()
        self.audit_label.setText(self._audit_text(ch, assets))
        name = self._current_character_name() or ""
        if assets is None:
            self.detail_label.setText(
                f"「{name}」还没有人物文件夹。点「建人物文件夹」在 "
                f"output/characters/{name}/ 下建一个，或点「迁移旧资产到人物文件夹」从旧资产迁过来。"
            )
            self.model_card_label.setText("还没有人物文件夹，先建文件夹。")
            self._detail_placeholder("还没有人物文件夹：点右侧「建人物文件夹」。")
            self._rebuild_detail_buttons(ch, None)
            self._refresh_aliases_label(None)
            return
        lines = [
            f"文件夹：{assets.folder}",
            f"衣服：{assets.costume_description or '未打标签'}（截图 {len(assets.costume_files)} 张）",
            f"声音：{len(assets.voice_samples)} 条样本，音色：音调 {assets.timbre.get('pitch', 0):+.1f} 半音 / 语速 {assets.timbre.get('rate', 1.0):.2f} 倍 / 风格 {assets.timbre.get('style') or '—'}",
            f"抠像：{len(assets.cutouts)} 张",
        ]
        if assets.model_source == "none":
            lines.append("3D 模型：无。先抠像再点「生成 3D 模型」，或点「导入 3D 模型」手动选 GLB。本界面不渲染 3D 预览。")
        elif assets.model_source == "face_card":
            lines.append("3D 模型：脸卡占位（关键点 + 贴图，不是完整网格），见 model/。")
        else:
            lines.append(
                f"3D 模型：{assets.model_source}（{assets.model_format or '未知格式'}），{assets.model_path or ''}。本界面不渲染 3D 预览，请用 Blender 打开查看。"
            )
        try:
            from recast.assets3d.variants import active_costume_variant, list_costume_variants

            variants = list_costume_variants(assets.folder)
            if variants:
                active = active_costume_variant(assets.folder)
                names = "、".join(str(v.get("name") or "") for v in variants)
                lines.append(
                    f"换装变体：{len(variants)} 个（{names}），当前穿：{active or '主模型'}。"
                    "衣服是整身变体，不是独立衣片；装配区点身体或点「切换换装变体…」换衣服。"
                )
        except Exception:
            variants = []
        self.detail_label.setText("\n".join(lines))
        if assets.model_source == "none":
            self.model_card_label.setText("无 3D 模型。先抠像再点「生成 3D 模型」，或点「导入 3D 模型」手动选 GLB。本界面不渲染 3D 预览。")
        elif assets.model_source == "face_card":
            self.model_card_label.setText("脸卡占位（关键点 + 贴图，不是完整网格），见 model/。")
        else:
            self.model_card_label.setText(
                f"{assets.model_source}（{assets.model_format or '未知格式'}），{assets.model_path or ''}。本界面不渲染 3D 预览，请用 Blender 打开查看。"
            )
        if variants:
            try:
                from recast.assets3d.variants import active_costume_variant

                active = active_costume_variant(assets.folder)
            except Exception:
                active = ""
            self.model_card_label.setText(
                self.model_card_label.text()
                + f"\n换装变体：{len(variants)} 个，当前穿：{active or '主模型'}。"
            )
        shown_any = False
        for caption, files in (("服装", assets.costume_files), ("抠像", assets.cutouts)):
            if not files:
                continue
            shown_any = True
            path = files[0]
            box = QVBoxLayout()
            img = QLabel()
            img.setFixedSize(140, 160)
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pix = QPixmap(str(path))
            if not pix.isNull():
                img.setPixmap(pix.scaled(140, 160, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                img.setText(Path(path).name)
            cap = QLabel(f"{caption}：{Path(path).name}")
            cap.setStyleSheet("color:#8b90a0; font-size:11px;")
            wrap = QWidget()
            box.addWidget(img)
            box.addWidget(cap)
            wrap.setLayout(box)
            self.detail_images_layout.addWidget(wrap)
        # N26：换装变体行——缩略图 + [删除变体]。缩略图复用变体登记的源图
        # （source_image，在人物文件夹内才有可解析的相对路径）；源图找
        # 不到时只显示占位文字，不编造图片。没有变体时这一段什么都不画。
        if variants:
            shown_any = True
            try:
                from recast.assets3d.variants import active_costume_variant

                active_name = active_costume_variant(assets.folder)
            except Exception:
                active_name = ""
            for variant in variants:
                vname = str(variant.get("name") or "")
                if not vname:
                    continue
                box = QVBoxLayout()
                img = QLabel()
                img.setFixedSize(140, 160)
                img.setAlignment(Qt.AlignmentFlag.AlignCenter)
                thumb = self._variant_thumbnail_path(assets.folder, variant)
                pix = QPixmap(str(thumb)) if thumb is not None else QPixmap()
                if not pix.isNull():
                    img.setPixmap(pix.scaled(140, 160, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                else:
                    img.setText("无预览图")
                    img.setStyleSheet("color:#8b90a0; border:1px solid #3a3f4d;")
                worn = "（当前穿）" if vname == active_name else ""
                cap = QLabel(f"变体：{vname}{worn}")
                cap.setStyleSheet("color:#8b90a0; font-size:11px;")
                box.addWidget(img)
                box.addWidget(cap)
                del_btn = QPushButton("删除变体")
                del_btn.setObjectName("danger")
                del_btn.setToolTip(f"删除换装变体「{vname}」：删掉变体的 GLB 文件并注销登记")
                del_btn.clicked.connect(lambda _checked=False, n=vname: self._delete_costume_variant(n))
                box.addWidget(del_btn)
                wrap = QWidget()
                wrap.setLayout(box)
                self.detail_images_layout.addWidget(wrap)
        if not shown_any:
            self._detail_placeholder("还没有服装截图或抠像：先点右侧「衣服打标签…」或「抠像」。")
        self.detail_images_layout.addStretch(1)
        self._rebuild_detail_buttons(ch, assets)
        self._refresh_aliases_label(assets)

    def _rebuild_detail_buttons(self, ch: dict | None, assets) -> None:
        """N27：详情卡缺项行后直接挂直达按钮，缺什么补什么。

        每个按钮都调已有的入口，不另起一套：缺模型 → _generate_3d，
        缺声音 → _audition（已有的试听入口），缺抠像 → _matte，
        缺授权 → _go_authorize（焦点落到右侧授权勾选框）。不缺的项
        不出按钮；没建文件夹时只出 [建人物文件夹]。
        """
        while self.detail_buttons_layout.count():
            item = self.detail_buttons_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.detail_action_buttons = []
        if ch is None:
            return

        def _add(text: str, tooltip: str, slot, enabled: bool = True) -> None:
            btn = QPushButton(text)
            btn.setToolTip(tooltip)
            btn.setEnabled(enabled)
            btn.clicked.connect(slot)
            self.detail_buttons_layout.addWidget(btn)
            self.detail_action_buttons.append(btn)

        if assets is None:
            _add("建人物文件夹", "在 output/characters/<名字>/ 下建人物文件夹", self._build_folder)
            self.detail_buttons_layout.addStretch(1)
            return
        if assets.model_source == "none":
            _add(
                "生成 3D 模型",
                "用第一张抠像生成 GLB" if assets.cutouts else "先抠像（cutout/ 为空）",
                self._generate_3d,
                enabled=bool(assets.cutouts),
            )
        if not assets.voice_samples:
            _add("试听/导入声音", "试听该人物的声音（走已有的试听入口）", self._audition)
        if not assets.cutouts:
            _add("抠像", "给该人物跑抠像", self._matte)
        if not ch.get("authorized") or not ch.get("auth_type"):
            _add("去补授权", "到右侧「基本信息」勾选已授权、选授权类型，再点保存确认", self._go_authorize)
        self.detail_buttons_layout.addStretch(1)

    def _go_authorize(self) -> None:
        """N27：缺授权的直达动作——焦点落到右侧「已授权」勾选框。

        不替用户勾选（授权是用户自己的决定），只把焦点送过去并在
        状态栏写清下一步：勾选 + 选类型 + 保存确认。
        """
        self.auth.setFocus()
        self.status_label.setText("在右侧「基本信息」勾选「已授权」，选择授权类型，再点「保存确认」。")

    # ----- N20：别名表（meta.json aliases，人物库 UI 侧） -----

    @staticmethod
    def _read_aliases(folder) -> list[str]:
        """人物文件夹 meta.json 里的 aliases（字符串列表），读不到就当没有。"""
        from recast.core.characters import load_meta

        try:
            meta = load_meta(Path(folder)) or {}
        except Exception:
            return []
        aliases = meta.get("aliases")
        if not isinstance(aliases, list):
            return []
        return [str(a).strip() for a in aliases if str(a or "").strip()]

    def _refresh_aliases_label(self, assets) -> None:
        """基本信息区的别名行：有就列出来，没有写“别名：无”。"""
        if assets is None:
            self.aliases_label.setText("别名：—")
            return
        aliases = self._read_aliases(assets.folder)
        self.aliases_label.setText(
            "别名：" + "、".join(aliases) if aliases else "别名：无"
        )

    def _add_alias(self) -> None:
        """N20：手动给当前人物加一个别名，写进 meta.json aliases。"""
        if self._current_id is None:
            return
        assets = self._current_assets()
        if assets is None:
            self.status_label.setText("先建人物文件夹，再加别名。")
            return
        current_name = self._current_character_name() or ""
        alias, ok = QInputDialog.getText(self, "加别名", "别名（曾用名，比如改名前的名字）：")
        if not ok:
            return
        alias = (alias or "").strip()
        if not alias:
            return
        if alias == current_name:
            self.status_label.setText(f"「{alias}」就是当前名字，不用记成别名。")
            return
        from recast.core.characters import load_meta, save_meta

        folder = Path(assets.folder)
        aliases = self._read_aliases(folder)
        if alias in aliases:
            self.status_label.setText(f"「{alias}」已经在别名里了。")
            return
        meta = load_meta(folder) or {}
        meta["aliases"] = [*aliases, alias]
        save_meta(folder, meta)
        self.reload()
        # 状态放 reload 之后：reload 会触发装配预览任务，任务结果会写
        # 状态栏，先写会被盖掉。
        self.status_label.setText(f"已给「{current_name}」加别名「{alias}」。")

    def _detail_placeholder(self, text: str) -> None:
        """服装/抠像区空时的占位说明，不留黑框。"""
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setStyleSheet("color:#8b90a0;")
        self.detail_images_layout.addWidget(lab)

    # ----- 人物装配视图 -----

    @staticmethod
    def _assembly_empty_label(text: str) -> QLabel:
        """装配行空时的占位说明：缺什么、点哪个按钮补，不留黑框。"""
        lab = QLabel(text)
        lab.setProperty("hint_text", text)
        lab.setWordWrap(True)
        lab.setStyleSheet("color:#8b90a0;")
        lab.setVisible(False)
        return lab

    def _sync_assembly_empty(self, row: QListWidget, label: QLabel, *, no_selection: bool) -> None:
        """装配行空 → 藏起空行，显示占位说明；有内容 → 反过来。"""
        empty = row.count() == 0
        row.setVisible(not empty)
        label.setVisible(empty)
        if empty:
            label.setText(
                "请从左侧选择人物。" if no_selection else str(label.property("hint_text") or "")
            )

    @staticmethod
    def _assembly_row(tooltip: str) -> QListWidget:
        """One horizontal pick row: icon items, single selection, click to apply."""
        row = QListWidget()
        row.setToolTip(tooltip)
        row.setViewMode(QListView.ViewMode.IconMode)
        row.setFlow(QListView.Flow.LeftToRight)
        row.setWrapping(False)
        row.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        row.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        row.setFixedHeight(120)
        row.setIconSize(QSize(96, 96))
        return row

    def _refresh_assembly(self) -> None:
        """Rebuild the assembly view for the selected character.

        Preview: the body's cached model/preview.png, else a placeholder
        card (model info + "请用 Blender 打开查看") — never a fake picture.
        A missing/stale preview kicks off a background render once per body;
        a failed render is not retried until the body changes.
        """
        assets = self._current_assets()
        name = self._current_character_name()
        if assets is None or not name:
            self.assembly_name.setText("请从左侧选择人物。")
            self.assembly_preview.setText("请从左侧选择人物。")
            self.assembly_preview.setPixmap(QPixmap())
            for row in (self.assembly_face_list, self.assembly_body_list, self.assembly_costume_list):
                row.clear()
            self.assembly_summary.setText("当前装配：—")
            detail = getattr(self, "assembly_body_detail", None)
            if detail is not None:
                detail.setText("请从左侧选择人物。")
            self._sync_assembly_empty(self.assembly_face_list, self.assembly_face_empty, no_selection=not name)
            self._sync_assembly_empty(self.assembly_body_list, self.assembly_body_empty, no_selection=not name)
            self._sync_assembly_empty(self.assembly_costume_list, self.assembly_costume_empty, no_selection=not name)
            return
        self.assembly_name.setText(f"「{name}」当前装配")
        self._show_assembly_preview(assets)
        self._fill_assembly_row(
            self.assembly_face_list, list(assets.face_files), assets.face_path
        )
        try:
            bodies = self.service.list_bodies()
        except Exception:
            bodies = []
        body_paths = [Path(str(item["path"])) for item in bodies]
        body_icons = {
            Path(str(item["path"])): item.get("preview")
            for item in bodies
            if item.get("preview")
        }
        body_labels = {
            Path(str(item["path"])): f"{item['name']}\n（{item['character']}）"
            for item in bodies
        }
        self._fill_assembly_row(
            self.assembly_body_list,
            body_paths,
            assets.body_path,
            icons=body_icons,
            labels=body_labels,
        )
        try:
            wardrobe = self.service.list_wardrobe()
        except Exception:
            wardrobe = []
        self._fill_assembly_row(
            self.assembly_costume_list,
            [Path(str(item["path"])) for item in wardrobe],
            assets.costume_path,
        )
        self._sync_assembly_empty(self.assembly_face_list, self.assembly_face_empty, no_selection=False)
        self._sync_assembly_empty(self.assembly_body_list, self.assembly_body_empty, no_selection=False)
        self._sync_assembly_empty(self.assembly_costume_list, self.assembly_costume_empty, no_selection=False)
        self.assembly_summary.setText(self._assembly_summary_text(assets))
        self._show_body_detail(self.assembly_body_list.currentItem())
        self._maybe_render_preview(assets)

    def _show_assembly_preview(self, assets) -> None:
        """Big preview picture, or an honest placeholder card."""
        preview: Path | None = None
        folder = assets.folder
        candidate = folder / "model" / "preview.png"
        if candidate.is_file():
            preview = candidate
        if preview is not None:
            pix = QPixmap(str(preview))
            if not pix.isNull():
                self.assembly_preview.setPixmap(
                    pix.scaled(
                        280,
                        280,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                self.assembly_preview.setText("")
                return
        if assets.body_path is not None:
            self.assembly_preview.setPixmap(QPixmap())
            self.assembly_preview.setText(
                f"身体：{assets.body_path.name}\n"
                f"（{assets.model_source}，{assets.model_format or '未知格式'}）\n"
                "没有预览图：本机渲染不了 3D 预览，\n请用 Blender 打开模型查看。"
            )
        elif assets.model_source == "face_card":
            self.assembly_preview.setPixmap(QPixmap())
            self.assembly_preview.setText("身体：脸卡占位（不是完整网格）。\n先导入或生成 GLB 身体模型。")
        else:
            self.assembly_preview.setPixmap(QPixmap())
            self.assembly_preview.setText("身体：无 3D 模型。\n先抠像再点「生成 3D 模型」，\n或点「导入 3D 模型」手动选 GLB。")

    def _body_source_label(self, path: str | Path, assets) -> str:
        """Where this body GLB came from (character's own / variant / shared)."""
        if assets is not None and getattr(assets, "body_path", None) is not None:
            try:
                if Path(path) == Path(assets.body_path):
                    return model_source_label(assets.model_source)
            except Exception:
                pass
        if "variants" in Path(path).parts:
            return "换装变体（自动生成）"
        try:
            Path(path).relative_to(Path(self.service.settings.output_root) / "models")
            return "共享模型库"
        except Exception:
            pass
        return "未知"

    def _body_detail_text(self, path: str | Path | None, assets) -> str:
        """Detail card text for one body GLB (or the no-model placeholder)."""
        if path is None:
            return "没有 3D 模型：点右侧「生成 3D 模型」或「导入 3D 模型」。"
        info = describe_glb_model(path)
        if not info.get("ok"):
            return f"模型：{Path(path).name}\n模型文件读不懂：{info.get('error') or '未知原因'}"
        triangles = info.get("triangles")
        lines = [
            f"模型：{Path(path).name}",
            f"来源：{self._body_source_label(path, assets)}",
            f"面数：{'约 ' + format(triangles, ',') + ' 个三角面' if triangles is not None else '未知'}",
            f"骨骼：{'有（' + str(info.get('skeleton_count')) + ' 套）' if info.get('skeleton_count') else '无'}",
            f"贴图：{'有（' + str(info.get('texture_count')) + ' 张）' if info.get('texture_count') else '无'}",
            f"大小：{_format_file_size(int(info.get('size') or 0))}",
        ]
        if not info.get("skeleton_count"):
            lines.append(
                "【不可直接重定向，需先绑骨】身体重渲染试点要求带骨骼的模型；"
                "TripoSG 生成的模型没有骨骼，先自动绑骨或换一个带骨骼的模型。"
            )
        return "\n".join(lines)

    def _show_body_detail(self, item: QListWidgetItem | None = None) -> None:
        """Show the detail card for the given body row item (or current body)."""
        label = getattr(self, "assembly_body_detail", None)
        if label is None:
            return
        assets = self._current_assets()
        path: str | None = None
        if item is not None:
            raw = item.data(Qt.ItemDataRole.UserRole) or ""
            path = str(raw) if raw else None
        if path is None and assets is not None:
            body = assets.body_path
            path = str(body) if body is not None else None
        label.setText(self._body_detail_text(path, assets))

    def _on_body_detail_select(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        self._show_body_detail(current)

    def _fill_assembly_row(self, row: QListWidget, paths, selected, *, icons=None, labels=None) -> None:
        """Fill one pick row; the effective pick is highlighted."""
        row.clear()
        icons = icons or {}
        labels = labels or {}
        for path in paths:
            path = Path(str(path))
            item = QListWidgetItem(labels.get(path, path.name))
            icon_path = icons.get(path)
            if icon_path is None and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                icon_path = str(path)
            if icon_path:
                pix = QPixmap(str(icon_path))
                if not pix.isNull():
                    item.setIcon(pix)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            row.addItem(item)
            if selected is not None and Path(str(path)) == Path(str(selected)):
                row.setCurrentItem(item)

    def _on_assembly_pick(self, item: QListWidgetItem) -> None:
        """Click a pick: apply immediately and save to meta.json."""
        row = self.sender()
        cid = self._current_id
        if not cid or row is None:
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not path:
            return
        slot = str(row.property("assembly_slot") or "")
        label = {"face": "脸", "body": "身体", "costume": "衣服"}.get(slot)
        if not label:
            return
        try:
            report = self.service.set_character_assembly(cid, **{slot: path})
        except Exception as exc:
            QMessageBox.warning(self, "无法更换装配", str(exc))
            return
        if slot == "costume":
            binding = report.get("binding") or {}
            if binding.get("applied"):
                self.status_label.setText(f"衣服已换成 {Path(path).name}，已绑定到身体模型。")
            else:
                self.status_label.setText(
                    f"衣服已换成 {Path(path).name}（未绑定：{binding.get('reason') or '—'}）。"
                )
        else:
            self.status_label.setText(f"{label}已换成 {Path(path).name}。")
        self._refresh_assembly()
        self.reload()

    def _assembly_summary_text(self, assets) -> str:
        """Current assembly, one line: face / body / costume (+ binding)."""
        face = assets.face_path.name if assets.face_path else "未选"
        body = assets.body_path.name if assets.body_path else "未选"
        costume = assets.costume_path.name if assets.costume_path else "未选"
        note = ""
        if assets.costume_path is not None:
            from recast.assets3d.binding import list_bindings
            from recast.core.characters import get_assembly

            stored = get_assembly(assets.meta).get("costume") or ""
            bound = any(
                isinstance(b, dict)
                and (b.get("costume") == stored or (b.get("source") or "") == stored)
                for b in list_bindings(assets.folder)
            )
            note = "（已绑定）" if bound else "（未绑定：没有 GLB 模型或未登记）"
        try:
            from recast.assets3d.variants import active_costume_variant

            active = active_costume_variant(assets.folder)
            if active:
                note += f" 换装变体：正穿着「{active}」（衣服是整身变体，不是独立衣片）"
        except Exception:
            pass
        return f"当前装配：脸 {face} / 身体 {body} / 衣服 {costume}{note}"

    def _maybe_render_preview(self, assets) -> None:
        """Render the body's preview in the background, once per body."""
        body = assets.body_path
        if body is None:
            return
        dest = assets.folder / "model" / "preview.png"
        try:
            stale = (not dest.is_file()) or (dest.stat().st_mtime < body.stat().st_mtime)
        except OSError:
            return
        if not stale:
            return
        if self._preview_attempted_for == str(body):
            return
        self._preview_attempted_for = str(body)
        cid = self._current_id
        if not cid or self._task_running:
            return
        self._run_task(
            "渲染 3D 预览",
            lambda: self.service.render_character_preview(cid),
            after=lambda _outcome: self._refresh_assembly(),
        )

    def _show_samples(self, ch: dict, *, empty: str | None = None) -> None:
        """人脸样本并排显示：脸卡（face/）的原图 + 并排对照。

        每个样本是可点选的按钮：点中后下面的「移到另一人物…」「移出到待审核」
        「拆分为新人物…」才亮——纠错就是对选中的样本操作。
        """
        while self.samples_layout.count():
            item = self.samples_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._selected_samples = []
        paths = [a.get("path") for a in (ch.get("assets") or []) if a.get("kind") == "face" and a.get("path")]
        if not paths:
            lab = QLabel(empty or "还没有人脸样本。完成「03 人物提取与建库」后会自动出现，也可手动指定参考脸。")
            lab.setWordWrap(True)
            self.samples_layout.addWidget(lab)
            return
        for path in paths[:12]:
            pix = QPixmap(str(path))
            btn = QToolButton()
            btn.setCheckable(True)
            if not pix.isNull():
                btn.setIcon(QIcon(pix))
            btn.setIconSize(QSize(140, 160))
            btn.setText(Path(path).name)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            btn.setToolTip(str(path))
            btn.setProperty("sample_path", str(path))
            btn.clicked.connect(self._on_sample_pick)
            self.samples_layout.addWidget(btn)
        self.samples_layout.addStretch(1)

    def _toggle_select_all_samples(self) -> None:
        """脸样本全选/取消全选：样本是可多选的打勾按钮，批量纠错对选中的全部生效。"""
        buttons = [b for b in self.samples.findChildren(QToolButton)]
        if not buttons:
            return
        target = not all(b.isChecked() for b in buttons)
        for b in buttons:
            b.setChecked(target)
        # setChecked 不触发 clicked，自己维护选中列表。
        self._selected_samples = [
            str(b.property("sample_path")) for b in buttons if b.isChecked()
        ]
        self._refresh_correction_states()

    def _on_sample_pick(self) -> None:
        """样本按钮点选切换：维护选中列表，刷新纠错按钮状态。"""
        btn = self.sender()
        if btn is None:
            return
        path = btn.property("sample_path")
        if path is None:
            return
        if btn.isChecked():
            if path not in self._selected_samples:
                self._selected_samples.append(path)
        else:
            if path in self._selected_samples:
                self._selected_samples.remove(path)
        self._refresh_correction_states()

    def _toggle_hint(self, checked: bool) -> None:
        """检查器底部的长说明：默认折叠，点一下展开。"""
        self.hint_label.setVisible(checked)
        self.hint_toggle.setText("说明：L1 / L2 是什么 ▾" if checked else "说明：L1 / L2 是什么 ▸")

    def _refresh_correction_states(self) -> None:
        """纠错三按钮只在选中了样本时才亮，没选人或没选样本都灰掉。"""
        enabled = self._current_id is not None and bool(self._selected_samples)
        for btn in (self.correct_move_btn, self.correct_remove_btn, self.correct_split_btn):
            btn.setEnabled(enabled)

    # ----- C 线：替换预览（单帧换脸对照）-----

    def _open_swap_preview(self) -> None:
        """取一张采集帧，用当前装配的脸换上去，弹窗并排对照。

        采集帧：优先选中的人脸样本，没选中就取第一张样本/抠像（跳过
        与参考脸同一个文件）。参考脸：装配区选的脸，没选就取第一张
        样本。真换脸跑不动时 preview_swap 会诚实降级为对齐叠加预览，
        弹窗里会用红字标明，绝不拿假图充数。
        """
        from recast.stages.swap_preview import preview_swap

        name = self._current_character_name()
        assets = self._current_assets()
        if not name or assets is None:
            QMessageBox.warning(self, "替换预览", "请先从左侧选择人物。")
            return
        reference = (assets.assembly or {}).get("face") or assets.face_path
        if reference is None:
            QMessageBox.warning(
                self,
                "替换预览",
                "没有参考脸：先在装配区点选一张脸（或先跑「03 人物提取与建库」出人脸样本）。",
            )
            return
        candidates = [Path(p) for p in self._selected_samples]
        candidates += list(assets.face_files) + list(assets.cutouts)
        source = next(
            (p for p in candidates if p.is_file() and Path(p) != Path(reference)),
            None,
        )
        if source is None:
            QMessageBox.warning(
                self,
                "替换预览",
                "没有可用的采集帧：只有一张人脸样本且它就是参考脸，换上去看不出区别。"
                "再加一张样本，或在装配区换一张参考脸。",
            )
            return
        try:
            result = preview_swap(source, reference)
        except Exception as exc:  # noqa: BLE001 - show the failure, don't crash
            QMessageBox.warning(self, "替换预览失败", str(exc))
            return
        self._last_swap_preview = result
        self._show_swap_preview_dialog(name, source, result)

    def _show_swap_preview_dialog(self, name: str, source, result: dict) -> None:
        """对照弹窗：左原图、右预览图；降级预览红字标明，身体只放占位说明。"""
        from recast.stages.swap_preview import (
            BODY_PLACEHOLDER_TEXT,
            NOT_FINAL_TEXT,
        )

        dialog = QDialog(self)
        dialog.setWindowTitle(f"替换预览 — {name}")
        dialog.resize(900, 620)
        layout = QVBoxLayout(dialog)

        def _image_label(pixmap: QPixmap) -> QLabel:
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setPixmap(
                pixmap.scaled(
                    400,
                    400,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            return label

        images_row = QHBoxLayout()
        for caption, pixmap in (
            ("原图", QPixmap(str(source))),
            ("预览图", self._preview_pixmap(result["image"])),
        ):
            column = QVBoxLayout()
            title = QLabel(caption)
            title.setStyleSheet("font-weight:bold;")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(title)
            column.addWidget(_image_label(pixmap), 1)
            images_row.addLayout(column, 1)
        layout.addLayout(images_row, 1)

        method_label = QLabel(str(result.get("note") or ""))
        method_label.setWordWrap(True)
        layout.addWidget(method_label)
        if not result.get("is_final"):
            warn = QLabel(NOT_FINAL_TEXT)
            warn.setWordWrap(True)
            warn.setStyleSheet("color:#ff5d5d; font-weight:bold; font-size:15px;")
            layout.addWidget(warn)
        body_note = QLabel(BODY_PLACEHOLDER_TEXT)
        body_note.setWordWrap(True)
        body_note.setStyleSheet("color:#8b90a0;")
        layout.addWidget(body_note)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
        self._swap_preview_dialog = dialog
        dialog.show()

    @staticmethod
    def _preview_pixmap(image) -> QPixmap:
        """BGR numpy 帧 → QPixmap（不经过磁盘）。"""
        import numpy as np

        rgb = np.ascontiguousarray(image[:, :, ::-1])
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimage.copy())

    # ----- P0–P3 管线：耗时操作走工作线程，不卡界面 -----

    def _run_task(self, label: str, func, *, after=None) -> None:
        """Run a slow service call off the UI thread; show the result in the status bar."""
        if self._task_running:
            return
        self._task_running = True
        self._refresh_action_states()
        self.status_label.setText(f"正在{label}…")
        if self._sync_tasks:
            try:
                result = func()
                outcome: dict = {"ok": True, "result": result}
            except Exception as exc:
                outcome = {"ok": False, "message": str(exc)}
            self._on_task_finished((label, outcome, after))
            return

        def work() -> None:
            try:
                result = func()
                outcome = {"ok": True, "result": result}
            except Exception as exc:
                outcome = {"ok": False, "message": str(exc)}
            self._task_signal.finished.emit((label, outcome, after))

        threading.Thread(target=work, name="recast-library-task", daemon=True).start()

    def _on_task_finished(self, payload) -> None:
        label, outcome, after = payload
        self._task_running = False
        if outcome.get("ok"):
            result = outcome.get("result")
            if isinstance(result, dict) and result.get("message"):
                message = str(result["message"])
            else:
                message = self._summarize_result(label, result)
            self.status_label.setText(message)
            self.reload()
        else:
            message = f"{label}失败：{outcome.get('message')}"
            if label in ("生成 3D 模型", "生成换装变体"):
                # Generation failures are almost always a missing prerequisite:
                # show the exact next steps instead of only the error line.
                try:
                    from recast.assets3d.generate3d import TRIPOSG_SETUP_TEXT

                    message += "\n\n" + TRIPOSG_SETUP_TEXT
                except Exception:
                    pass
            self.status_label.setText(message)
            QMessageBox.warning(self, f"{label}失败", message)
        self._refresh_action_states()
        if after is not None:
            after(outcome)

    def _summarize_result(self, label: str, result) -> str:
        if not isinstance(result, dict):
            return f"{label}完成。"
        if label == "抠像":
            failed = result.get("failed") or []
            detail = ""
            if failed:
                detail = "；失败：" + "、".join(
                    f"{item.get('source')}: {item.get('reason')}" for item in failed[:3]
                )
            backend = result.get("backend") or ""
            via = f"（{backend}）" if backend else ""
            skipped = len(result.get("skipped") or [])
            tail = f"，跳过 {skipped} 张（已存在）" if skipped else ""
            return (
                f"抠像完成{via}：成功 {len(result.get('outputs') or [])} 张，"
                f"失败 {len(failed)} 张{tail}{detail}。"
            )
        if label == "生成 3D 模型":
            return f"3D 模型已生成：{result.get('path')}（只出形状，不带贴图、不带骨骼，不能直接用于身体重渲染）。已登记进人物文件夹 model/，装配的身体已指向它。"
        if label == "生成换装变体":
            return (
                f"换装变体「{result.get('variant')}」已生成：{result.get('path')}。"
                "衣服是整身变体，不是独立衣片；装配区点身体或点「切换换装变体…」换上它。"
            )
        if label == "导入 3D 模型":
            return f"已导入 GLB 模型：{result.get('path')}。"
        if label == "试听声音":
            duration = result.get("duration")
            dur = f"，{float(duration):.1f} 秒" if duration else ""
            return f"试听已生成：{result.get('path')}{dur}（已用默认播放器打开）。"
        if label == "迁移旧资产":
            return (
                f"迁移完成：新建 {len(result.get('created') or [])} 个人物文件夹，"
                f"复制 {result.get('copied', 0)} 个文件，跳过 {result.get('skipped', 0)} 个，"
                f"问题 {len(result.get('errors') or [])} 条。"
            )
        if label == "衣服打标签":
            return (
                f"衣服打标签完成：成功 {len(result.get('tagged') or [])} 张，"
                f"失败 {len(result.get('failed') or [])} 张；描述：{result.get('description') or '—'}。"
            )
        if label == "生成设定板":
            turns = len(result.get("turns") or [])
            if result.get("model_rendered"):
                return f"设定板已建：{turns} 个角度。位置：{result.get('folder')}\\sheet"
            return f"设定板已建（只有照片）：{result.get('render_reason') or ''}"
        if label == "换装预览":
            audit = result.get("audit") or {}
            return (
                f"换装预览完成：分割 {result.get('method')}，"
                f"背景损伤 {float(audit.get('background_damage_frac') or 0):.3%}，"
                f"接缝增量 {audit.get('seam_increase')}× → {audit.get('verdict')}。"
                f"对比图：{result.get('after')}"
            )
        return f"{label}完成。"

    def _open_audition_file(self, outcome: dict) -> None:
        if not outcome.get("ok"):
            return
        path = (outcome.get("result") or {}).get("path")
        if path:
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            except Exception:
                pass

    # ----- P0–P3 按钮处理 -----

    def _import_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "导入视频/照片", "", "视频/图片 (*.mp4 *.mov *.mkv *.avi *.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not path:
            return
        self._source_path = path
        self.source_label.setText(f"已导入：{Path(path).name}")
        self.status_label.setText("可以点「自动定义人物」。")
        self._refresh_action_states()

    def _auto_define(self) -> None:
        if not self._source_path:
            QMessageBox.information(self, "自动定义人物", "请先导入视频/照片。")
            return
        workdir = QFileDialog.getExistingDirectory(
            self, "选择分集工作目录（有 identity.json，可取消跳过名字建议）", str(Path.home())
        )
        source = self._source_path
        self._run_task("自动定义人物", lambda: self._auto_define_work(source, workdir or None))

    def _auto_define_work(self, source: str, workdir: str | None) -> dict:
        """一次跑完：说话人分离切声音 → 名字建议 → 衣服打标签。

        人脸聚类不在这里：请在主界面跑「03 人物提取与建库」。
        """
        notes: list[str] = []
        names = [entry["name"] for entry in list_character_folders(self.service.settings.output_root)]
        report = self.service.run_diarization(source, character_names=names or None)
        clips = report.get("clips") or {}
        skipped = report.get("skipped") or []
        if skipped and not names:
            speakers = sorted({str(seg.get("speaker")) for seg in skipped if seg.get("speaker")})
            created = []
            for index, _speaker in enumerate(speakers, 1):
                self.service.ensure_character_folder(f"人物{index}", name_source="auto")
                created.append(f"人物{index}")
            if created:
                second = self.service.run_diarization(source, character_names=created)
                clips = second.get("clips") or {}
                notes.append(
                    f"说话人分离：{report.get('segments', 0)} 段；新建人物文件夹 {'、'.join(created)}，"
                    f"重新分离后 {len(clips)} 个人物有声音样本"
                )
            else:
                notes.append(f"说话人分离：{report.get('segments', 0)} 段，但没有对应人物，全部跳过")
        else:
            notes.append(f"说话人分离：{report.get('segments', 0)} 段，{len(clips)} 个人物有声音样本")
            if skipped:
                notes.append(f"{len(skipped)} 段没有对应人物，已跳过")
        if workdir:
            try:
                suggestions = self.service.suggest_names(workdir=workdir)
                notes.append(f"名字建议：{len(suggestions)} 条，已写入各人物文件夹（请在人物库确认，不会自动改名）")
            except Exception as exc:
                notes.append(f"名字建议跳过：{exc}")
        else:
            notes.append("名字建议跳过：未选分集工作目录")
        folders = list_character_folders(self.service.settings.output_root)
        if len(folders) == 1 and Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            try:
                rep = self.service.tag_costume(folders[0]["name"], sources=[source])
                notes.append(f"衣服打标签：{rep.get('description') or '完成'}")
            except Exception as exc:
                notes.append(f"衣服打标签跳过：{exc}")
        else:
            notes.append("衣服打标签跳过：只有一张全身照片且只有一个人时才能在自动定义里打标签，多人请逐个点「衣服打标签…」")
        notes.append("人脸聚类请在主界面跑「03 人物提取与建库」。")
        return {"ok": True, "message": "；".join(notes) + "。"}

    def _migrate_folders(self) -> None:
        self._run_task("迁移旧资产", self.service.migrate_character_folders)

    def _build_folder(self) -> None:
        name = self._current_character_name()
        if not name:
            return
        try:
            result = self.service.ensure_character_folder(name)
        except Exception as exc:
            QMessageBox.warning(self, "无法建人物文件夹", str(exc))
            return
        self.status_label.setText(f"已建人物文件夹：{result.get('folder')}")
        self.reload()

    def _matte(self) -> None:
        cid = self._current_id
        if not cid:
            return
        self._run_task("抠像", lambda: self.service.run_matting(cid))

    # ----- 2D 路线：设定板 / 换装预览 -----

    def _sheet_availability(self) -> tuple[bool, str]:
        """[生成设定板] 能不能点，以及缺什么。"""
        assets = self._current_assets() if self._current_id else None
        if assets is None:
            return False, "先在左侧选一个人，并建好人物文件夹"
        return True, (
            "生成设定板：有 GLB 就渲染多角度转盘，同时从照片里量主色和肤色。"
            "发色和体型不做测量（没有分割就不猜），留空比猜错便宜。"
            "结果写进人物文件夹 sheet/，并登记到 meta.json。"
        )

    def _garment_availability(self) -> tuple[bool, str]:
        """[换装预览…] 能不能点，以及缺什么。"""
        assets = self._current_assets() if self._current_id else None
        if assets is None:
            return False, "先在左侧选一个人，并建好人物文件夹"
        try:
            from recast.stages.garment_transfer import costume_reference

            has_costume = costume_reference(assets.folder) is not None
        except Exception:
            has_costume = False
        if not has_costume:
            return False, "还没有服装参考图：先跑[衣服打标签…]，或把全身截图放进 costume/"
        return True, (
            "换装预览…：挑一张图当帧，把 costume/ 里的服装配色换上去。"
            "只改衣物的色度，亮度/褶皱/光影保持不变，所以不会出现贴图感，也不会换身材。"
            "跑完会生成 before/after 两张对比图并自动打开 after。"
        )

    def _build_sheet(self) -> None:
        cid = self._current_id
        if not cid:
            return
        self._run_task("生成设定板", lambda: self.service.build_sheet_for_character(cid))

    def _preview_garment(self) -> None:
        cid = self._current_id
        if not cid:
            return
        self._run_task(
            "换装预览",
            lambda: self.service.preview_garment_swap(cid),
            after=self._open_after_image,
        )

    def _open_after_image(self, outcome: dict) -> None:
        if not outcome.get("ok"):
            return
        path = (outcome.get("result") or {}).get("after")
        if path:
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            except Exception:
                pass

    def _generation_check(self) -> dict | None:
        """Pre-flight for TripoSG generation (UI: show before starting).

        Returns the service check dict {"ok", "reason", "setup_steps"}, or
        None when there is no checker (e.g. a stubbed service in tests) —
        the caller then tries the generation anyway.
        """
        try:
            return self.service.check_model_generation()
        except Exception:
            return None

    def _generation_blocked(self) -> bool:
        """True when a prerequisite is missing (dialog shown, do not start).

        A missing prerequisite is shown with the exact Windows preparation
        steps, instead of starting a run bound to fail.
        """
        check = self._generation_check()
        if check is None or check.get("ok"):
            return False
        QMessageBox.warning(
            self,
            "还不能生成 3D 模型",
            f"{check.get('reason') or '缺少生成 3D 模型的前提条件。'}\n\n{check.get('setup_steps') or ''}",
        )
        return True

    def _bind_availability(self) -> tuple[bool, str]:
        """[绑定服装到模型] 现在能不能点，以及不能点时缺什么。

        能点 = 选了人 + 有人物文件夹 + 装配区有一件衣服（已选的装配服
        装，没有就先在装配区衣服行点一件）。有没有 GLB 不在这里拦：
        没 GLB 时点绑定会如实报“先去生成/导入模型”，不是死按钮。
        """
        if self._current_id is None:
            return False, "先从左侧选一个人。"
        assets = self._current_assets()
        if assets is None:
            return False, "先建人物文件夹。"
        if assets.costume_path is None:
            return False, "先在装配区点一件衣服（衣服行），再点绑定。"
        return (
            True,
            "把当前装配的衣服登记到 GLB 模型（全身部位，登记 + 校验）。"
            "贴图本身不写进 GLB 二进制：Windows 上用 Blender 或 pygltflib 完成最后一步。",
        )

    def _bind_costume(self) -> None:
        """把当前装配的衣服绑定到 GLB 模型（U4 路线 a：接上真实能力）。

        走已有的 service.set_character_assembly(costume=...)：它会把
        共享衣柜的衣服先拷进人物 costume/，再调 binding.bind_costume
        登记进 meta.json model.bindings（部位“全身”），装配摘要随即
        显示“已绑定”。能力边界见 binding.py：只做登记 + 校验，贴图不
        写进 GLB 二进制（本机没有 pygltflib / trimesh，不硬写以免损坏
        模型）；没 GLB 时绑定会跳过并报出原因，不是静默失败。
        适用场景：衣服是先选的、GLB 是后导入/生成的——选衣时绑定跳过
        了，点这个按钮补绑一次。
        """
        cid = self._current_id
        if not cid:
            return
        assets = self._current_assets()
        if assets is None or assets.costume_path is None:
            self.status_label.setText("先在装配区选一件衣服，再点「绑定服装到模型」。")
            return
        try:
            report = self.service.set_character_assembly(cid, costume=str(assets.costume_path))
        except Exception as exc:
            QMessageBox.warning(self, "无法绑定服装", str(exc))
            return
        binding = report.get("binding") or {}
        if not binding.get("applied"):
            reason = binding.get("reason") or "没有 GLB 模型或未登记。"
            QMessageBox.warning(self, "衣服还没绑定", reason)
            self.status_label.setText(f"衣服还没绑定：{reason}")
            return
        self.status_label.setText(
            f"衣服 {Path(str(assets.costume_path)).name} 已绑定到身体模型"
            "（全身，登记在 meta.json；贴图尚未写入 GLB 二进制，"
            "Windows 上用 Blender 或 pygltflib 完成最后一步）。"
        )
        self._refresh_assembly()
        self.reload()

    def _model3d_queue_dialog(self):
        """复用同一个 3D 生成队列对话框（第一波 model3d_queue.py）。

        重复点 [生成 3D 模型] 只往同一队列里加行，不另开对话框。
        _model3d_generate_fn 只给测试注入假后端，正常走
        service.generate_model_3d（对话框的默认 generate_fn）。
        """
        from recast.desktop.model3d_queue import Model3dQueueDialog

        dialog = getattr(self, "_model3d_queue_dialog_obj", None)
        if dialog is None:
            generate_fn = getattr(self, "_model3d_generate_fn", None)
            kwargs: dict[str, Any] = {
                "on_item_finished": self._on_model3d_item_finished,
                "on_close": self._on_model3d_queue_closed,
            }
            if generate_fn is not None:
                dialog = Model3dQueueDialog(self.service, self, generate_fn=generate_fn, **kwargs)
            else:
                dialog = Model3dQueueDialog(self.service, self, **kwargs)
            self._model3d_queue_dialog_obj = dialog
        return dialog

    def _on_model3d_item_finished(self, character: str, status: str, error: str) -> None:
        """N31：3D 生成一行跑完——状态栏轻提示，绝不弹模态框。

        完成时自动刷新人物库（新模型不用手动刷新就看得到）；失败时
        只写失败原因 + 指到队列对话框的 [重试]。reload 在状态之前：
        reload 会触发装配预览任务，任务结果会写状态栏，先写会被盖掉。
        """
        from recast.desktop.model3d_queue import ST_DONE

        if status == ST_DONE:
            self.reload()
            self.status_label.setText(f"「{character}」3D 模型已生成，人物库已刷新。")
        else:
            reason = f"：{error}" if error else ""
            self.status_label.setText(
                f"「{character}」3D 生成失败{reason}，可在队列对话框点 [重试]。"
            )

    def _on_model3d_queue_closed(self, pending: int) -> None:
        """N31：队列对话框关掉时，队列未空就在状态栏留一行“进行中”。

        关窗只是隐藏（对话框被缓存，daemon worker 关窗后继续跑、只在
        进程退出时停，见 model3d_queue.py 模块 docstring），所以这里
        只留提示、不做任何“停止生成”的事。队列已空时不打扰。
        """
        if pending > 0:
            self.status_label.setText(
                f"3D 生成进行中（{pending} 个排队），完成后在人物库刷新可见。"
            )

    def _generate_3d(self) -> None:
        cid = self._current_id
        if not cid:
            return
        if self._generation_blocked():
            return
        name = self._current_character_name() or cid
        dialog = self._model3d_queue_dialog()
        dialog.enqueue(name)
        dialog.show()
        self.status_label.setText(
            f"「{name}」已排进 3D 生成队列（8GB 显存下一次只生成一个，"
            "进度和失败重试都在队列对话框里；生成完成后人物库自动刷新，"
            "关掉对话框后生成继续跑，完成后在人物库刷新可见）。"
        )

    def _generate_variant(self) -> None:
        cid = self._current_id
        if not cid:
            return
        if self._generation_blocked():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择全身照（穿着这身衣服的人物全身，衣服是整身变体，不是独立衣片）",
            "",
            "图片 (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if not path:
            return
        name, ok = QInputDialog.getText(self, "生成换装变体", "变体名字（比如“红裙”）：")
        if not ok or not str(name).strip():
            return
        self._run_task(
            "生成换装变体",
            lambda: self.service.generate_costume_variant(
                cid, image=path, variant_name=str(name).strip()
            ),
        )

    def _switch_variant(self) -> None:
        cid = self._current_id
        if not cid:
            return
        name = self._current_character_name()
        if not name:
            return
        try:
            variants = self.service.list_costume_variants(cid)
        except Exception as exc:
            QMessageBox.warning(self, "无法切换换装变体", str(exc))
            return
        items = ["主模型"] + [str(v.get("name") or "") for v in variants if v.get("name")]
        if len(items) <= 1:
            QMessageBox.information(self, "切换换装变体", "还没有换装变体，先点「生成换装变体…」。")
            return
        chosen, ok = QInputDialog.getItem(
            self,
            "切换换装变体",
            "换上哪身衣服？（衣服是整身变体：切换就是把装配的身体换成对应变体的 GLB）",
            items,
            0,
            False,
        )
        if not ok or not chosen:
            return
        try:
            report = self.service.switch_costume_variant(
                cid, variant_name="" if chosen == "主模型" else str(chosen)
            )
        except Exception as exc:
            QMessageBox.warning(self, "无法切换换装变体", str(exc))
            return
        worn = report.get("variant") or "主模型"
        self.status_label.setText(f"已换成{worn}：{report.get('path')}。")
        self._refresh_assembly()
        self.reload()

    @staticmethod
    def _variant_thumbnail_path(folder, variant: dict) -> Path | None:
        """变体缩略图的真实图片路径，没有时返回 None（调用方显示占位）。

        只认变体登记的 source_image（生成变体时用的全身照）：它在人
        物文件夹内时是相对路径，直接拼；文件夹外的源图登记时只存了
        文件名、拼不出来，就返回 None，不去猜、不用主模型的预览图
        冒充（那是另一身衣服）。
        """
        rel = str(variant.get("source_image") or "").strip()
        if not rel:
            return None
        candidate = Path(folder) / rel
        if candidate.is_file():
            return candidate
        return None

    def _delete_costume_variant(self, name: str) -> None:
        """N26：删除一个换装变体——删 GLB 文件 + 注销 meta 登记，带确认。

        当前穿的变体被删时切回主模型（有主模型才切，没主模型只清
        active_variant，不留指向已删文件的装配）。文件已经不在时只
        清登记，不报错。删完 reload，列表即时更新。
        """
        name = (name or "").strip()
        if not name or self._current_id is None:
            return
        assets = self._current_assets()
        if assets is None:
            return
        folder = Path(assets.folder)
        answer = QMessageBox.question(
            self,
            "删除换装变体",
            f"确定删除换装变体「{name}」吗？\n会删掉变体的 GLB 文件，并从人物信息里注销登记。"
            "当前穿的变体被删时会切回主模型。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        from recast.assets3d.variants import list_costume_variants
        from recast.core.characters import load_meta, save_meta

        variants = list_costume_variants(folder)
        entry = next((v for v in variants if str(v.get("name") or "") == name), None)
        if entry is None:
            self.status_label.setText(f"没有叫「{name}」的换装变体，可能已经被删了。")
            self.reload()
            return
        rel = str(entry.get("path") or "").strip()
        glb = folder / rel if rel else None
        file_was_there = bool(glb is not None and glb.is_file())
        if glb is not None and file_was_there:
            try:
                glb.unlink()
            except OSError as exc:
                QMessageBox.warning(self, "删除变体失败", f"变体文件删不掉：{glb}（{exc}）")
                return
        meta = load_meta(folder) or {}
        model = meta.get("model") if isinstance(meta.get("model"), dict) else {}
        model = {
            **model,
            "variants": [
                v for v in (model.get("variants") or [])
                if isinstance(v, dict) and v.get("name") != name
            ],
        }
        was_active = str(model.get("active_variant") or "") == name
        if was_active:
            model["active_variant"] = ""
        meta["model"] = model
        save_meta(folder, meta)
        switched_back = False
        if was_active:
            main_rel = str(model.get("path") or "").strip()
            main = folder / main_rel if main_rel else None
            if main is not None and main.is_file():
                try:
                    from recast.core.characters import set_assembly

                    set_assembly(folder, body=str(main))
                    switched_back = True
                except Exception as exc:  # noqa: BLE001 - 登记已清，装配切回失败只报状态
                    logging.getLogger("recast").warning("删除变体后切回主模型失败：%s", exc)
        note = "（已切回主模型）" if switched_back else ("（文件已不在，只清了登记）" if not file_was_there else "")
        self.reload()
        # 状态放 reload 之后：reload 会触发装配预览任务，任务结果会写
        # 状态栏，先写会被盖掉。
        self.status_label.setText(f"已删除换装变体「{name}」{note}。")

    def _import_glb(self) -> None:
        cid = self._current_id
        if not cid:
            return
        path, _ = QFileDialog.getOpenFileName(self, "导入 3D 模型（GLB）", "", "GLB 模型 (*.glb)")
        if not path:
            return
        try:
            report = self.service.import_model_glb(cid, path)
        except Exception as exc:
            QMessageBox.warning(self, "无法导入 3D 模型", str(exc))
            return
        self.status_label.setText(f"已导入 GLB 模型：{report.get('path')}。")
        self.reload()

    def _tag_costume(self) -> None:
        cid = self._current_id
        if not cid:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择全身照片（人脸小图没有衣服区域）", "", "图片 (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not paths:
            return
        self._run_task("衣服打标签", lambda: self.service.tag_costume(cid, sources=paths))

    def _audition(self) -> None:
        cid = self._current_id
        if not cid:
            return
        self._run_task(
            "试听声音",
            lambda: self.service.audition_voice(cid),
            after=self._open_audition_file,
        )

    def _open_timbre(self) -> None:
        name = self._current_character_name()
        if not name:
            return
        dlg = TimbreDialog(self.service, name, self)
        if dlg.exec():
            self.status_label.setText(f"「{name}」音色已保存。")
            self.reload()

    def _open_dialogue(self) -> None:
        name = self._current_character_name()
        if not name:
            return
        dlg = DialogueDialog(self.service, name, self)
        dlg.exec()
        self.reload()

    def _save(self) -> None:
        if not self._current_id:
            return
        # N20：改名时把旧名写入别名表。文件夹按旧名建，改名只改数据库
        # 的显示名、不动文件夹，所以文件夹引用要在 confirm 之前取。
        old_name = self._current_character_name() or ""
        assets = self._current_assets()
        folder = Path(assets.folder) if assets is not None else None
        new_name = self.name_edit.text().strip()
        try:
            self.service.confirm_character(
                self._current_id,
                display_name=new_name,
                authorized=self.auth.isChecked(),
                auth_type=self.auth_type.currentData() or None,
            )
        except Exception as exc:
            QMessageBox.warning(self, "无法保存", str(exc))
            return
        if folder is not None and old_name and new_name and old_name != new_name:
            try:
                from recast.core.characters import load_meta, save_meta

                aliases = self._read_aliases(folder)
                if old_name not in aliases:
                    meta = load_meta(folder) or {}
                    meta["aliases"] = [*aliases, old_name]
                    save_meta(folder, meta)
            except Exception as exc:  # noqa: BLE001 - 别名写失败不推翻已保存的改名
                logging.getLogger("recast").warning("改名后旧名记入别名失败：%s", exc)
        self.reload()

    def _pick_face(self) -> None:
        if not self._current_id:
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择参考脸", "", "图片 (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not path:
            return
        try:
            self.service.confirm_character(self._current_id, face_path=path)
        except Exception as exc:
            QMessageBox.warning(self, "无法设置参考脸", str(exc))
            return
        self.reload()

    def _add_manual(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择参考脸（可取消，稍后补充）", "", "图片 (*.png *.jpg *.jpeg *.webp *.bmp)")
        name = Path(path).stem if path else "新人物"
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "手动添加人物", "人物名称：", text=name)
        if not ok or not str(name).strip():
            return
        self.service.add_character_manual(str(name).strip(), path or None)
        self.reload()

    def _build_l2(self) -> None:
        if not self._current_id:
            QMessageBox.information(self, "3D头像资产", "请先选择一个已授权人物。")
            return
        if self.service.settings.is_basic_tier():
            QMessageBox.information(
                self,
                "当前为基础档",
                "无独立 GPU 时仍可尝试生成「3D头像资产」（关键点 + 贴图 + 驱动配置），但 L2 驱动已禁用。",
            )
        try:
            asset = self.service.build_l2(self._current_id)
        except Exception as exc:
            QMessageBox.warning(self, "无法生成 3D头像资产", str(exc))
            return
        QMessageBox.information(
            self,
            "已生成 3D头像资产",
            f"{asset.get('honesty')}\n\n质量分：{asset.get('quality_score')}（{asset.get('quality_flag')}）\n目录：{asset.get('dir')}",
        )
        self.reload()

    def _delete_l2(self) -> None:
        if not self._current_id:
            QMessageBox.information(self, "删除 3D头像资产", "请先选择一个人物。")
            return
        ch = next((c for c in self.service.characters() if c["id"] == self._current_id), None)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or self._current_id
        if not _confirm(
            self,
            "删除 3D头像资产",
            f"确定删除「{name}」的 3D头像资产？\n\n"
            "只会删除该人物的 3D头像文件，人脸照片会保留。"
            "删除后 3D 状态变为「缺失」，之后可以重新生成。",
            accept="删除",
        ):
            return
        try:
            result = self.service.delete_l2(self._current_id)
        except Exception as exc:
            QMessageBox.warning(self, "无法删除 3D头像资产", str(exc))
            return
        if result.get("partial"):
            failed = result.get("failed") or []
            detail = "\n".join(f"{item.get('path')}: {item.get('reason')}" for item in failed)
            QMessageBox.warning(
                self,
                "3D头像资产只删了一部分",
                f"已删除 {len(result.get('removed') or [])} 个目录，"
                f"{len(failed)} 个没删掉：\n{detail}\n请关闭占用文件的程序后重试。",
            )
        else:
            QMessageBox.information(
                self,
                "已删除 3D头像资产",
                f"已删除「{name}」的 3D头像资产。人脸照片仍保留，3D 状态为「缺失」。",
            )
        self.reload()

    def _delete_character(self) -> None:
        if not self._current_id:
            QMessageBox.information(self, "删除人物", "请先选择一个人物。")
            return
        ch = next((c for c in self.service.characters() if c["id"] == self._current_id), None)
        name = (ch or {}).get("display_name") or (ch or {}).get("name") or self._current_id
        if not _confirm(
            self,
            "删除人物",
            f"确定把「{name}」移出人物库？\n\n"
            "这个人会离开人物库，数据库里的人物记录和资料记录也会去掉。\n"
            "若有 3D头像资产，只删除位于 L2 目录内、且通过安全检查的那一份。\n"
            "生成的人脸样本（输出目录中的 face_samples）会删除。\n"
            "从其他位置指定的参考脸不会删除。剧集、任务和成片不会删除。",
            accept="删除",
        ):
            return
        try:
            result = self.service.delete_character(self._current_id)
        except Exception as exc:
            QMessageBox.warning(self, "无法删除人物", str(exc))
            return
        if result.get("partial"):
            failed = result.get("failed") or []
            detail = "\n".join(f"{item.get('path')}: {item.get('reason')}" for item in failed)
            QMessageBox.warning(
                self,
                "人物只删了一部分",
                f"「{name}」仍在人物库中，{len(failed)} 个文件没删掉：\n{detail}\n"
                "请关闭占用文件的程序后重试。",
            )
            self.reload()
            return
        self._current_id = None
        self.reload()
        QMessageBox.information(self, "已移出人物库", f"已将「{name}」移出人物库。")

    def _clean_invalid_samples(self) -> None:
        try:
            plan = self.service.list_invalid_samples()
        except Exception as exc:
            QMessageBox.warning(self, "无法检查失效样本", str(exc))
            return
        if not plan.get("stale_count") and not plan.get("remove_count"):
            QMessageBox.information(self, "清理失效样本", "没有失效样本。")
            return
        if not _confirm(self, "清理失效样本", _cleanup_detail(plan), accept="清理"):
            return
        try:
            result = self.service.clean_invalid_samples()
        except Exception as exc:
            QMessageBox.warning(self, "无法清理失效样本", str(exc))
            return
        self._current_id = None
        self.reload()
        QMessageBox.information(
            self,
            "已清理失效样本",
            f"已清理 {result.get('stale_count', 0)} 条失效资料，"
            f"移出 {result.get('remove_count', 0)} 个人物。\n"
            "仍有人脸文件的人物留在人物库中，只去掉失效资料。",
        )

    def _clean_unused_l2(self) -> None:
        try:
            items = self.service.list_unused_l2()
        except Exception as exc:
            QMessageBox.warning(self, "无法检查 3D头像资产", str(exc))
            return
        if not items:
            QMessageBox.information(self, "清理未使用的 3D头像资产", "没有未使用的 3D头像资产。")
            return
        shown = items[:30]
        lines = []
        for item in shown:
            label = item.get("display_name") or item.get("name") or item.get("character_id") or "未命名"
            cid = item.get("character_id") or ""
            if cid and cid != label:
                label = f"{label}（{cid}）"
            reason = item.get("reason") or ""
            lines.append(f"· {label}：{reason}" if reason else f"· {label}")
        extra = ""
        if len(items) > len(shown):
            extra = f"\n… 另有 {len(items) - len(shown)} 项"
        detail = (
            f"将删除 {len(items)} 个未使用的 3D头像资产：\n\n"
            + "\n".join(lines)
            + extra
            + "\n\n不会删除仍在使用的 3D头像，也不会删除人脸、服装、声音，或 L2 目录以外的文件。\n确定删除吗？"
        )
        if not _confirm(self, "清理未使用的 3D头像资产", detail, accept="删除"):
            return
        try:
            result = self.service.delete_unused_l2(only=[item.get("path") or "" for item in items])
        except Exception as exc:
            QMessageBox.warning(self, "无法清理 3D头像资产", str(exc))
            return
        QMessageBox.information(
            self,
            "已清理 3D头像资产",
            f"已删除 {result.get('count', 0)} 个未使用的 3D头像资产。",
        )
        self.reload()


def _capped_lines(items: list[dict], *, limit: int = 30) -> list[str]:
    lines = [f"· {item.get('label') or item.get('name') or item.get('id')}" for item in items[:limit]]
    extra = len(items) - len(lines)
    if extra > 0:
        lines.append(f"… 另有 {extra} 项")
    return lines


def _cleanup_detail(plan: dict) -> str:
    kind_counts = plan.get("kind_counts") or {}
    kind_text = "、".join(f"{label} {count}" for label, count in kind_counts.items())
    summary = f"失效资料 {plan.get('stale_count', 0)} 条"
    if kind_text:
        summary += f"（{kind_text}）"
    summary += f"，将移出人物库 {plan.get('remove_count', 0)} 人（没有仍有效的人脸文件）。"
    parts = [summary]
    stale = list(plan.get("stale_assets") or [])
    if stale:
        parts.append("失效资料：\n" + "\n".join(_capped_lines(stale)))
    remove = list(plan.get("remove_characters") or [])
    if remove:
        parts.append("将移出人物库（没有仍有效的人脸文件，包括已确认和已授权的人物）：\n" + "\n".join(_capped_lines(remove)))
    keep = list(plan.get("keep_characters") or [])
    if keep:
        parts.append("以下人物仍有人脸文件，会留在人物库，只去掉失效资料：\n" + "\n".join(_capped_lines(keep)))
    parts.append(
        "不会删除仍存在的人脸文件，也不会删除剧集、任务或成片。\n"
        "输出目录以外的参考照片不会删除。仍有效的 3D头像目录不会在这一步删除，"
        "之后可以用「清理未使用的 3D头像资产」处理。\n"
        "确定清理吗？"
    )
    return "\n\n".join(parts)


def _confirm(parent: QWidget, title: str, text: str, *, accept: str, reject: str = "取消") -> bool:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(title)
    box.setText(text)
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    box.setDefaultButton(QMessageBox.StandardButton.No)
    yes = box.button(QMessageBox.StandardButton.Yes)
    no = box.button(QMessageBox.StandardButton.No)
    if yes is not None:
        yes.setText(accept)
    if no is not None:
        no.setText(reject)
    return box.exec() == QMessageBox.StandardButton.Yes
