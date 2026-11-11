"""Real four-level template editor: edit, validate, import/export, bind a pack to a drama."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
)

from recast.core.service import RecastService
from recast.templates.pack import TemplatePack, export_pack, import_pack, load_pack, save_pack, upsert_pack_row
from recast.templates.validate import validate_pack, validate_shot, primary_actions_of


class TemplateEditor(QDialog):
    def __init__(self, service: RecastService, parent: QWidget | None = None, drama_id: str | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.drama_id = drama_id
        self.pack: TemplatePack | None = None
        self.setWindowTitle("模板编辑器")
        self.resize(980, 680)

        self.path_label = QLabel("尚未打开模板包")
        self.path_label.setWordWrap(True)
        self.version_edit = QLineEdit()
        self.name_edit = QLineEdit()

        open_btn = QPushButton("打开模板包…")
        open_btn.clicked.connect(self._open)
        new_btn = QPushButton("新建…")
        new_btn.clicked.connect(self._new)
        save_btn = QPushButton("保存")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self._save)
        imp = QPushButton("导入…")
        imp.clicked.connect(self._import)
        exp = QPushButton("导出…")
        exp.clicked.connect(self._export)
        bind = QPushButton("把当前包套到所选剧目")
        bind.clicked.connect(self._bind)

        top = QHBoxLayout()
        top.addWidget(open_btn)
        top.addWidget(new_btn)
        top.addWidget(save_btn)
        top.addWidget(imp)
        top.addWidget(exp)
        top.addWidget(bind)
        top.addStretch(1)

        self.tabs = QTabWidget()
        self._script_tab()
        self._list_tab("角色", "characters")
        self._list_tab("场景", "scenes")
        self._shot_tab()
        self._validate_tab()

        layout = QVBoxLayout(self)
        layout.addWidget(self.path_label)
        meta = QHBoxLayout()
        meta.addWidget(QLabel("名称"))
        meta.addWidget(self.name_edit, 1)
        meta.addWidget(QLabel("版本"))
        meta.addWidget(self.version_edit)
        layout.addLayout(meta)
        layout.addLayout(top)
        layout.addWidget(self.tabs, 1)
        hint = QLabel(
            "全局风格锁只写在剧本模板里，保存时会自动拼到每一镜。"
            "一镜两个主动作、多人镜头缺参考图、打架镜超过两人，都会在保存和执行前拦截。"
            "先静音、后贴轨的顺序改不了。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8b90a0;")
        layout.addWidget(hint)
        self._try_default()

    def _script_tab(self) -> None:
        w = QWidget()
        form = QFormLayout(w)
        self.sec_edit = QLineEdit()
        self.narr_edit = QLineEdit("F1")
        self.narr_rule = QPlainTextEdit()
        self.narr_rule.setPlaceholderText("旁白规则，例如：旁白只在开头段落 F1，其后只贴对白。")
        self.style_lock = QPlainTextEdit()
        self.style_lock.setPlaceholderText("全局风格锁（画风、色调）。只在这里维护一份。")
        self.negative = QPlainTextEdit()
        self.negative.setPlaceholderText("负面词，每镜自动追加。")
        self.width = QSpinBox(); self.width.setRange(16, 4096); self.width.setValue(720)
        self.height = QSpinBox(); self.height.setRange(16, 4096); self.height.setValue(1280)
        self.duration = QDoubleSpinBox(); self.duration.setRange(1, 3600); self.duration.setValue(90)
        self.fight_two = QCheckBox("打架类镜头最多两人同框（硬约束）")
        self.fight_two.setChecked(True)
        self.bgm = QCheckBox("无 BGM（推荐勾选）")
        self.bgm.setChecked(True)
        form.addRow("分段结构（逗号分隔）", self.sec_edit)
        form.addRow("旁白所在段落", self.narr_edit)
        form.addRow("旁白规则", self.narr_rule)
        form.addRow("风格锁", self.style_lock)
        form.addRow("负面词", self.negative)
        form.addRow("导出宽度", self.width)
        form.addRow("导出高度", self.height)
        form.addRow("目标时长（秒）", self.duration)
        form.addRow("", self.fight_two)
        form.addRow("", self.bgm)
        self.tabs.addTab(w, "剧本")

    def _list_tab(self, title: str, attr: str) -> None:
        w = QWidget()
        col = QVBoxLayout(w)
        lst = QListWidget()
        editor = QPlainTextEdit()
        editor.setPlaceholderText("选中一项后编辑 JSON")
        row = QHBoxLayout()
        add = QPushButton("新增")
        rm = QPushButton("删除")
        row.addWidget(add)
        row.addWidget(rm)
        row.addStretch(1)
        col.addWidget(lst, 1)
        col.addLayout(row)
        col.addWidget(editor, 2)
        setattr(self, f"{attr}_list", lst)
        setattr(self, f"{attr}_edit", editor)

        def load_item() -> None:
            item = lst.currentItem()
            if not item:
                return
            editor.setPlainText(json.dumps(item.data(Qt.ItemDataRole.UserRole), ensure_ascii=False, indent=2))

        def save_item() -> None:
            item = lst.currentItem()
            if not item:
                return
            try:
                data = json.loads(editor.toPlainText() or "{}")
            except Exception as exc:
                QMessageBox.warning(self, "JSON 无效", str(exc))
                return
            item.setData(Qt.ItemDataRole.UserRole, data)
            item.setText(str(data.get("name") or data.get("id") or data.get("shot_code") or "未命名"))

        def add_item() -> None:
            blank = {"id": "", "name": "新" + title, "level": "character" if attr == "characters" else "scene"}
            item = QListWidgetItem(blank["name"])
            item.setData(Qt.ItemDataRole.UserRole, blank)
            lst.addItem(item)
            lst.setCurrentItem(item)

        def rm_item() -> None:
            row_i = lst.currentRow()
            if row_i >= 0:
                lst.takeItem(row_i)

        lst.currentItemChanged.connect(lambda *_: load_item())
        editor.textChanged.connect(save_item)
        add.clicked.connect(add_item)
        rm.clicked.connect(rm_item)
        self.tabs.addTab(w, title)

    def _shot_tab(self) -> None:
        w = QWidget()
        col = QVBoxLayout(w)
        self.shot_list = QListWidget()
        self.shot_edit = QPlainTextEdit()
        self.shot_edit.setPlaceholderText("分镜 JSON：shot_code、prompt、primary_action（只能一个）、ref_assets、dialogue、duration_sec、people、is_fight")
        row = QHBoxLayout()
        add = QPushButton("新增镜头")
        rm = QPushButton("删除镜头")
        row.addWidget(add)
        row.addWidget(rm)
        row.addStretch(1)
        col.addWidget(self.shot_list, 1)
        col.addLayout(row)
        col.addWidget(self.shot_edit, 2)

        def load_item() -> None:
            item = self.shot_list.currentItem()
            if item:
                self.shot_edit.blockSignals(True)
                self.shot_edit.setPlainText(json.dumps(item.data(Qt.ItemDataRole.UserRole), ensure_ascii=False, indent=2))
                self.shot_edit.blockSignals(False)

        def save_item() -> None:
            item = self.shot_list.currentItem()
            if not item:
                return
            try:
                data = json.loads(self.shot_edit.toPlainText() or "{}")
            except Exception:
                return
            item.setData(Qt.ItemDataRole.UserRole, data)
            item.setText(str(data.get("shot_code") or data.get("code") or "镜头"))

        def add_item() -> None:
            blank = {
                "shot_code": "F1-L1",
                "prompt": "",
                "primary_action": "",
                "ref_assets": [],
                "people": [],
                "dialogue": "",
                "duration_sec": 5,
                "is_fight": False,
            }
            item = QListWidgetItem("F1-L1")
            item.setData(Qt.ItemDataRole.UserRole, blank)
            self.shot_list.addItem(item)
            self.shot_list.setCurrentItem(item)

        def rm_item() -> None:
            r = self.shot_list.currentRow()
            if r >= 0:
                self.shot_list.takeItem(r)

        self.shot_list.currentItemChanged.connect(lambda *_: load_item())
        self.shot_edit.textChanged.connect(save_item)
        add.clicked.connect(add_item)
        rm.clicked.connect(rm_item)
        self.tabs.addTab(w, "分镜")

    def _validate_tab(self) -> None:
        w = QWidget()
        col = QVBoxLayout(w)
        run = QPushButton("立即校验")
        run.clicked.connect(self._validate)
        self.issues = QPlainTextEdit()
        self.issues.setReadOnly(True)
        col.addWidget(run)
        col.addWidget(self.issues, 1)
        self.tabs.addTab(w, "校验")

    def _try_default(self) -> None:
        from recast.config import PROJECT_ROOT

        default = PROJECT_ROOT / "examples" / "开局即打-第1集模板"
        if default.is_dir():
            try:
                self._load(default)
            except Exception:
                pass

    def _collect(self) -> TemplatePack | None:
        if self.pack is None:
            return None
        script = dict(self.pack.script or {})
        script.update(
            {
                "name": self.name_edit.text().strip() or self.pack.name,
                "version": self.version_edit.text().strip() or self.pack.version,
                "sections": [s.strip() for s in self.sec_edit.text().split(",") if s.strip()],
                "narration_segments": [s.strip() for s in self.narr_edit.text().split(",") if s.strip()],
                "narration_rule": self.narr_rule.toPlainText().strip(),
                "style_lock": self.style_lock.toPlainText().strip(),
                "negative": self.negative.toPlainText().strip(),
                "export": {"width": int(self.width.value()), "height": int(self.height.value())},
                "target_duration_sec": float(self.duration.value()),
                "fight_max_two": self.fight_two.isChecked(),
                "bgm": not self.bgm.isChecked(),
            }
        )
        def items(lst: QListWidget) -> list:
            return [lst.item(i).data(Qt.ItemDataRole.UserRole) for i in range(lst.count())]

        pack = TemplatePack(
            root=self.pack.root,
            name=self.name_edit.text().strip() or self.pack.name,
            version=self.version_edit.text().strip() or self.pack.version,
            script=script,
            characters=items(self.characters_list),
            scenes=items(self.scenes_list),
            shots=items(self.shot_list),
            workflows=self.pack.workflows,
            meta=self.pack.meta,
        )
        return pack

    def _populate(self, pack: TemplatePack) -> None:
        self.pack = pack
        self.path_label.setText(f"模板包：{pack.root}")
        self.name_edit.setText(pack.name)
        self.version_edit.setText(pack.version)
        script = pack.script or {}
        secs = script.get("sections") or []
        self.sec_edit.setText(",".join(secs) if isinstance(secs, list) else str(secs))
        narr = script.get("narration_segments") or ["F1"]
        self.narr_edit.setText(",".join(narr) if isinstance(narr, list) else str(narr))
        self.narr_rule.setPlainText(script.get("narration_rule") or "旁白只在开头段落 F1，其后只贴对白。")
        self.style_lock.setPlainText(script.get("style_lock") or "")
        self.negative.setPlainText(script.get("negative") or "")
        export = script.get("export") or {}
        self.width.setValue(int(export.get("width") or 720))
        self.height.setValue(int(export.get("height") or 1280))
        self.duration.setValue(float(script.get("target_duration_sec") or 90))
        self.fight_two.setChecked(bool(script.get("fight_max_two", True)))
        self.bgm.setChecked(not bool(script.get("bgm")))

        def fill(lst: QListWidget, rows: list, key: str) -> None:
            lst.clear()
            for row in rows:
                item = QListWidgetItem(str(row.get(key) or row.get("name") or row.get("id") or ""))
                item.setData(Qt.ItemDataRole.UserRole, row)
                lst.addItem(item)
            if lst.count():
                lst.setCurrentRow(0)

        fill(self.characters_list, pack.characters, "name")
        fill(self.scenes_list, pack.scenes, "name")
        fill(self.shot_list, pack.shots, "shot_code")
        self._validate()

    def _load(self, root: Path) -> None:
        self._populate(load_pack(root))

    def _open(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "打开模板包目录", str(self.pack.root if self.pack else Path.home()))
        if path:
            try:
                self._load(Path(path))
            except Exception as exc:
                QMessageBox.warning(self, "无法打开", str(exc))

    def _new(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择空目录保存新模板包", str(Path.home()))
        if not path:
            return
        pack = TemplatePack(
            root=Path(path),
            name="新模板",
            version="1.0.0",
            script={
                "name": "新模板",
                "version": "1.0.0",
                "sections": ["F1"],
                "narration_segments": ["F1"],
                "narration_rule": "旁白只在开头段落 F1，其后只贴对白。",
                "style_lock": "",
                "negative": "",
                "export": {"width": 720, "height": 1280},
                "target_duration_sec": 90,
                "fight_max_two": True,
                "bgm": False,
            },
            shots=[{"shot_code": "F1-L1", "prompt": "", "primary_action": "", "ref_assets": [], "people": [], "duration_sec": 5}],
        )
        save_pack(pack)
        self._populate(pack)

    def _save(self) -> None:
        pack = self._collect()
        if pack is None:
            QMessageBox.information(self, "保存", "请先打开或新建一个模板包。")
            return
        issues = pack.validate(require_files=True)
        errors = [i for i in issues if i.is_error]
        self._show_issues(issues)
        if errors:
            QMessageBox.warning(self, "校验未通过，未保存", "\n".join(i.message for i in errors[:12]))
            return
        save_pack(pack)
        upsert_pack_row(self.service.store, pack)
        self.pack = pack
        QMessageBox.information(self, "已保存", f"已保存模板包 {pack.root}（版本 {pack.version}）。")

    def _validate(self) -> None:
        pack = self._collect()
        if pack is None:
            self.issues.setPlainText("尚未打开模板包。")
            return
        self._show_issues(pack.validate(require_files=True))

    def _show_issues(self, issues) -> None:
        if not issues:
            self.issues.setPlainText("校验通过。")
            return
        lines = [f"[{i.level}] {i.message}" for i in issues]
        self.issues.setPlainText("\n".join(lines))

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入模板包（目录请用「打开」）", str(Path.home()), "Zip (*.zip)")
        if not path:
            dest = QFileDialog.getExistingDirectory(self, "或选择一个模板包目录", str(Path.home()))
            if not dest:
                return
            path = dest
        dest = QFileDialog.getExistingDirectory(self, "导入到哪个目录", str(Path.home()))
        if not dest:
            return
        try:
            pack = import_pack(path, dest)
            upsert_pack_row(self.service.store, pack)
            self._populate(pack)
        except Exception as exc:
            QMessageBox.warning(self, "导入失败", str(exc))

    def _export(self) -> None:
        pack = self._collect()
        if pack is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出模板包", str(Path.home() / f"{pack.name}.zip"), "Zip (*.zip)")
        if not path:
            return
        try:
            save_pack(pack)
            out = export_pack(pack, path)
            QMessageBox.information(self, "已导出", str(out))
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def _bind(self) -> None:
        pack = self._collect()
        if pack is None:
            return
        did = self.drama_id
        if not did:
            QMessageBox.information(self, "套用模板", "请先在主窗口勾选一部剧，再打开模板编辑器并点套用。")
            return
        try:
            save_pack(pack)
            self.service.assign_pack(did, pack.root)
            QMessageBox.information(self, "已套用", f"已把模板「{pack.name}」版本 {pack.version} 绑定到该剧。任务会记录这个版本，重跑时不会被新模板污染。")
        except Exception as exc:
            QMessageBox.warning(self, "套用失败", str(exc))
