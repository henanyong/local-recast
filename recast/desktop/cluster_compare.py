"""聚类对比视图（PM §三 3.2）：审核队列行双击打开的对照视图。

待审核的聚类一次判完：代表脸 + 全部样本脸并排，每张标一个质量分，
里面直接 [确认是同一人] [拆分] [丢弃]，复用已有的审核 service 方法
（confirm_character / split_character / review_assign /
review_create_character / review_discard），不另起一套确认逻辑。

诚实边界：聚类内两两相似度分数**不编**。提取时算的聚类距离 /
confidence 没有存进人物表（见 recast/core/review.py
high_confidence_items 的自注），队列里拿不到真正的两两分数。
这里只显示每张样本的文件质量分（score_sample_file：大小 × 清晰度，
是“这张脸清不清楚”，不是“这两张脸像不像”），并在界面上写清这一点，
判同一人靠肉眼并排对照。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QInputDialog,
    QVBoxLayout,
    QWidget,
)


class ClusterCompareDialog(QDialog):
    """One review row, side by side: representative face + all its samples.

    ``service`` is the RecastService (duck-typed in tests); ``item`` is one
    review-queue dict (kind / key / character_id / character_name / thumb /
    score / source). Actions call the existing review service methods and
    close with Accepted on success so the caller can reload the library.
    """

    def __init__(self, service: Any, item: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.item = dict(item)
        self.setWindowTitle(f"聚类对比：{self.item.get('character_name') or ''}")
        self.resize(860, 560)

        kind_text = "聚类" if self.item.get("kind") == "cluster" else "样本"
        title = QLabel(f"{kind_text} · {self.item.get('character_name') or ''} · {self.item.get('source') or ''}")
        title.setStyleSheet("font-weight:bold;")
        title.setWordWrap(True)

        # 代表脸：队列行自带的 thumb，没有就用第一张样本。
        self.rep_label = QLabel("没有代表脸。")
        self.rep_label.setFixedSize(180, 180)
        self.rep_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rep_label.setStyleSheet("border:1px solid #3a3f4d;")
        self.rep_label.setWordWrap(True)
        rep_info = QLabel("代表脸（队列行的缩略图）")
        rep_info.setStyleSheet("color:#8b90a0; font-size:11px;")

        self.samples_list = QListWidget()
        self.samples_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.samples_list.setFlow(QListWidget.Flow.LeftToRight)
        self.samples_list.setWrapping(True)
        self.samples_list.setIconSize(QPixmap(96, 96).size())
        self.samples_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.samples_list.setMinimumHeight(220)
        self.samples_list.setToolTip("全部样本脸并排对照；拆分时先点选要拆出去的样本（可多选）")

        self.note_label = QLabel(
            "质量分是单张脸的文件清晰度分（大小 × 清晰度），不是两两相似度："
            "聚类内两两相似度提取时没有保存，这里不编分数，判同一人靠肉眼并排对照。"
        )
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet("color:#8b90a0; font-size:11px;")

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        self.confirm_btn = QPushButton("确认是同一人")
        self.confirm_btn.setObjectName("primary")
        self.confirm_btn.clicked.connect(self._confirm)
        self.split_btn = QPushButton("拆分")
        self.split_btn.clicked.connect(self._split)
        self.discard_btn = QPushButton("丢弃")
        self.discard_btn.setObjectName("danger")
        self.discard_btn.clicked.connect(self._discard)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.reject)

        top_row = QHBoxLayout()
        rep_box = QVBoxLayout()
        rep_box.addWidget(self.rep_label)
        rep_box.addWidget(rep_info)
        top_row.addLayout(rep_box)
        top_row.addWidget(self.samples_list, 1)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.confirm_btn)
        btn_row.addWidget(self.split_btn)
        btn_row.addWidget(self.discard_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(top_row, 1)
        layout.addWidget(self.note_label)
        layout.addWidget(self.status_label)
        layout.addLayout(btn_row)

        self._sample_paths: list[str] = []
        self._fill()

    # ----- 数据：代表脸 + 全部样本脸 ----------------------------------------

    def _collect_samples(self) -> list[str]:
        """All face files for this review row, deduped, existing files only.

        Sources: the queue row's thumb, the character's store face assets,
        and the character folder's face/ files. Every source is guarded —
        a fake service in tests may not have settings / store at all.
        """
        out: list[str] = []
        seen: set[str] = set()

        def _add(raw: Any) -> None:
            if not raw:
                return
            try:
                p = Path(str(raw))
            except (TypeError, ValueError):
                return
            if not p.is_file() or p.is_symlink():
                return
            try:
                key = str(p.resolve())
            except OSError:
                key = str(p)
            if key in seen:
                return
            seen.add(key)
            out.append(str(p))

        _add(self.item.get("thumb"))
        key = str(self.item.get("key") or "")
        if self.item.get("kind") == "sample" and key:
            _add(key)
        character_id = self.item.get("character_id")
        character_name = str(self.item.get("character_name") or "")
        try:
            characters = self.service.characters()
        except Exception:
            characters = []
        ch = None
        for row in characters or []:
            if character_id and row.get("id") == character_id:
                ch = row
                break
            if character_name and (row.get("display_name") or row.get("name")) == character_name:
                ch = row
                break
        if ch is not None:
            for asset in ch.get("assets") or []:
                if asset.get("kind") == "face" and asset.get("path"):
                    _add(asset["path"])
        try:
            output_root = getattr(getattr(self.service, "settings", None), "output_root", None)
            store = getattr(self.service, "store", None)
            if output_root and character_name:
                from recast.core.character_assets import resolve_character_assets

                assets = resolve_character_assets(output_root, character_name, store=store)
                if assets is not None:
                    for path in assets.face_files:
                        _add(path)
        except Exception:
            pass
        return out

    def _fill(self) -> None:
        """Representative face on the left, every sample face side by side."""
        self._sample_paths = self._collect_samples()
        rep = self.item.get("thumb") or (self._sample_paths[0] if self._sample_paths else None)
        if rep:
            pix = QPixmap(str(rep))
            if not pix.isNull():
                self.rep_label.setPixmap(
                    pix.scaled(180, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                )
                self.rep_label.setText("")
        self.samples_list.clear()
        for path in self._sample_paths:
            score = self._quality_score(path)
            text = Path(path).name + (f"\n质量分 {score:.2f}" if score is not None else "")
            li = QListWidgetItem(text)
            li.setData(Qt.ItemDataRole.UserRole, path)
            pix = QPixmap(path)
            if not pix.isNull():
                from PySide6.QtGui import QIcon

                li.setIcon(QIcon(pix))
            self.samples_list.addItem(li)
        if not self._sample_paths:
            self.status_label.setText("这个聚类/样本没有找到人脸图片文件。")

    @staticmethod
    def _quality_score(path: str) -> float | None:
        """One face file's quality score (size x sharpness), or None.

        This is deliberately NOT a pairwise similarity: it only says how
        clear this one crop is (see module docstring).
        """
        try:
            from recast.core.review import score_sample_file

            return float(score_sample_file(path))
        except Exception:
            return None

    # ----- 操作：复用已有的审核 service 方法 ---------------------------------

    def _selected_sample_paths(self) -> list[str]:
        out: list[str] = []
        for item in self.samples_list.selectedItems():
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                out.append(str(path))
        return out

    def _confirm(self) -> None:
        """确认是同一人：聚类行确认整个聚类，样本行认领回它原来的人物。"""
        key = str(self.item.get("key") or "")
        try:
            if self.item.get("kind") == "cluster":
                self.service.confirm_character(key)
            else:
                owner = self.item.get("character_id")
                if not owner:
                    self.status_label.setText("这个样本没有归属人物，无法确认是同一人：请先在审核队列把它分给一个人物。")
                    return
                self.service.review_assign(key, owner)
        except Exception as exc:
            self.status_label.setText(f"确认失败：{exc}")
            return
        self.accept()

    def _split(self) -> None:
        """拆分：点选的样本拆出去成一个新人物（聚类行调 split_character）。

        样本行只有一张脸：拆分就是把它自己新建成一个人（review_create_character）。
        """
        key = str(self.item.get("key") or "")
        if self.item.get("kind") != "cluster":
            name, ok = QInputDialog.getText(self, "拆分为新人物", "新人物名字：")
            if not ok or not str(name).strip():
                return
            try:
                self.service.review_create_character(key, str(name).strip(), owner_id=self.item.get("character_id"))
            except Exception as exc:
                self.status_label.setText(f"拆分失败：{exc}")
                return
            self.accept()
            return
        paths = self._selected_sample_paths()
        if not paths:
            self.status_label.setText("先点选要拆出去的样本（可多选），再点「拆分」。")
            return
        name, ok = QInputDialog.getText(self, "拆分为新人物", "新人物名字：")
        if not ok or not str(name).strip():
            return
        try:
            self.service.split_character(key, paths, str(name).strip())
        except Exception as exc:
            self.status_label.setText(f"拆分失败：{exc}")
            return
        self.accept()

    def _discard(self) -> None:
        """丢弃：复用 review_discard（样本进 review_trash，聚类只删行）。"""
        key = str(self.item.get("key") or "")
        try:
            self.service.review_discard(key, owner_id=self.item.get("character_id"))
        except Exception as exc:
            self.status_label.setText(f"丢弃失败：{exc}")
            return
        self.accept()
