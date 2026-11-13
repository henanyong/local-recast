"""P2 第二波：人物库 N22 进度条、搜索+筛选、聚类对比视图、绑定服装、3D 生成队列接线。

每项 >= 2 个测试（正向 + 边界/失败）。全部用离屏 Qt + 真实 RecastService
（tmp_path）或假 service，不真跑 TripoSG、不真改用户库。
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ----- 通用夹具 -------------------------------------------------------------


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _service(tmp_path: Path):
    from recast.core.service import RecastService

    svc = RecastService(tmp_path / "db.db")
    svc.update_settings(output_root=str(tmp_path / "outputs"))
    return svc


def _window(tmp_path: Path):
    from recast.desktop.library_window import LibraryWindow

    _app()
    svc = _service(tmp_path)
    window = LibraryWindow(svc)
    window._sync_tasks = True
    return svc, window


def _select(window, row: int = 0) -> None:
    window.list.setCurrentRow(row)
    window._on_select(window.list.currentItem())


def _solid_image(path: Path, size: int = 120, color=(200, 60, 60)) -> Path:
    """一张纯色图：质量分低（清晰度 0），做低分样本用。"""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)
    return path


def _low_face(path: Path) -> Path:
    """40x40 纯色脸：质量分约 0.13（< 0.5），必进低分样本队列。"""
    return _solid_image(path, size=40, color=(180, 140, 120))


def _insert(svc, name: str, *, confirmed: int = 1, authorized: int = 0) -> dict:
    return svc.store.insert(
        "characters",
        name=name,
        display_name=name,
        confirmed=confirmed,
        authorized=authorized,
        cluster_key=f"cluster:{name}",
    )


def _visible_rows(window) -> list[int]:
    return [
        row
        for row in range(window.list.count())
        if not window.list.item(row).isHidden()
    ]


def _first_review_data_row(window):
    """审核队列里第一行数据行（跳过组头），没有返回 None。"""
    for row in range(window.review_list.count()):
        item = window.review_list.item(row)
        if not window._is_review_header(item):
            return item
    return None


def _minimal_glb(path: Path) -> Path:
    """最小合法 GLB（glTF 2.0 文件头 + JSON 块含 asset），validate_glb 能过。"""
    payload = json.dumps({"asset": {"version": "2.0"}}).encode()
    total = 12 + 8 + len(payload)
    data = struct.pack("<4sII", b"glTF", 2, total)
    data += struct.pack("<II", len(payload), 0x4E4F534A)
    data += payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ----- 1. N22：人物库审核进度条 ---------------------------------------------


def test_n22_progress_counts_low_samples_in_denominator(tmp_path: Path):
    """分母含低分样本：已确认 1 + 聚类 1 + 低分样本 1 = total 3，33%。

    老分母（已确认 + 聚类）会算成 total 2、50%，队列只剩低分样本时
    甚至报 100%。这里断言修复后的分母。
    """
    svc, window = _window(tmp_path)
    ch_a = _insert(svc, "温念", confirmed=1, authorized=1)
    low = _low_face(Path(svc.settings.output_root) / "faces" / "low.png")
    svc.store.set_character_asset(ch_a["id"], "face", str(low), complete=True)
    _insert(svc, "温雪", confirmed=0, authorized=0)
    window.reload()

    stats = window._review_progress_stats()
    assert stats == {
        "confirmed": 1,
        "pending": 2,
        "clusters": 1,
        "low": 1,
        "total": 3,
        "percent": 33,
    }
    assert stats["total"] == stats["confirmed"] + stats["pending"]  # 分母含全部待审核项
    assert window.review_progress.value() == 33
    assert not window.review_progress.isHidden()
    assert window.review_progress_label.text() == "审核进度：已确认 1 / 待审核 2（低分 1）"
    window.close()


def test_n22_zero_confirmed_only_low_samples_shows_zero_percent(tmp_path: Path):
    """零确认 + 仅低分样本：条 0% 显示，文案“待审核 N 项，未确认人物”。

    低分样本放在 review_trash/（纠错移出的样本）：不依赖任何人物行，
    所以 clusters=0、confirmed=0，只有 1 个 sample 项。
    """
    svc, window = _window(tmp_path)
    trash = Path(svc.settings.output_root) / "review_trash"
    _low_face(trash / "moved.png")
    window.reload()

    stats = window._review_progress_stats()
    assert stats["confirmed"] == 0
    assert stats["pending"] == 1
    assert stats["clusters"] == 0
    assert stats["low"] == 1
    assert stats["percent"] == 0
    assert not window.review_progress.isHidden()
    assert window.review_progress.value() == 0
    assert window.review_progress_label.text() == "待审核 1 项，未确认人物"
    window.close()


def test_n22_empty_queue_hides_bar(tmp_path: Path):
    """既没人物也没待审核项：条隐藏，文案“审核进度：还没有人物。”"""
    _svc, window = _window(tmp_path)
    window.reload()
    assert window._review_progress_stats()["total"] == 0
    assert window.review_progress.isHidden()
    assert window.review_progress_label.text() == "审核进度：还没有人物。"
    window.close()


# ----- 2. 搜索 + 状态筛选 ----------------------------------------------------


def test_search_filters_by_name_and_clears_back(tmp_path: Path):
    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    _insert(svc, "温雪", confirmed=1, authorized=0)
    _insert(svc, "陆寒征", confirmed=0, authorized=0)
    window.reload()
    assert len(_visible_rows(window)) == 3

    window.character_search.setText("温")
    assert len(_visible_rows(window)) == 2
    window.character_search.setText("陆寒征")
    assert len(_visible_rows(window)) == 1
    window.character_search.setText("不存在的名字")
    assert len(_visible_rows(window)) == 0
    assert window.character_empty_label.isVisible() or not window.character_empty_label.isHidden()
    assert "去审核队列" in window.character_empty_label.text()
    window.character_search.clear()
    assert len(_visible_rows(window)) == 3
    assert window.character_empty_label.isHidden()
    window.close()


def test_status_filter_unconfirmed_and_unauthorized(tmp_path: Path):
    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    _insert(svc, "温雪", confirmed=1, authorized=0)
    _insert(svc, "陆寒征", confirmed=0, authorized=0)
    window.reload()

    def _pick(data: str) -> None:
        idx = window.character_status_filter.findData(data)
        assert idx >= 0
        window.character_status_filter.setCurrentIndex(idx)

    _pick("unconfirmed")
    visible = _visible_rows(window)
    assert len(visible) == 1
    assert "陆寒征" in window.list.item(visible[0]).text()

    _pick("unauthorized")
    visible = _visible_rows(window)
    assert len(visible) == 2
    names = [window.list.item(r).text() for r in visible]
    assert any("温雪" in t for t in names)
    assert any("陆寒征" in t for t in names)

    _pick("no_model")
    assert len(_visible_rows(window)) == 3  # 三个人都没有人物文件夹/模型
    _pick("all")
    assert len(_visible_rows(window)) == 3
    window.close()


def test_status_filter_no_voice_after_folder_and_voice(tmp_path: Path):
    """无声音筛：有声音样本的人物被筛掉，没声音的留下来。"""
    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    _insert(svc, "温雪", confirmed=1, authorized=0)
    svc.ensure_character_folder("温念")
    voice = Path(svc.settings.output_root) / "characters" / "温念" / "voice" / "s1.wav"
    voice.parent.mkdir(parents=True, exist_ok=True)
    voice.write_bytes(b"RIFF" + b"\x00" * 40)  # 只占位：解析只看文件存在
    from recast.core.characters import load_meta, save_meta

    folder = Path(svc.settings.output_root) / "characters" / "温念"
    meta = load_meta(folder) or {}
    meta["voice"] = {"files": [str(voice)]}
    save_meta(folder, meta)
    window.reload()

    idx = window.character_status_filter.findData("no_voice")
    window.character_status_filter.setCurrentIndex(idx)
    visible = _visible_rows(window)
    assert len(visible) == 1
    assert "温雪" in window.list.item(visible[0]).text()
    window.close()


# ----- 3. 聚类对比视图 -------------------------------------------------------


class _FakeService:
    """对比视图用的假 service：只记调用，不碰数据库/文件。"""

    def __init__(self, tmp_path: Path, face_path: Path):
        self.calls: list[tuple] = []
        self.face_path = face_path
        self.settings = SimpleNamespace(output_root=str(tmp_path / "outputs"))
        self.store = None

    def characters(self):
        return [
            {
                "id": "c1",
                "display_name": "温念",
                "name": "温念",
                "assets": [{"kind": "face", "path": str(self.face_path)}],
            }
        ]

    def confirm_character(self, character_id, **kwargs):
        self.calls.append(("confirm_character", character_id))
        return {"id": character_id}

    def split_character(self, character_id, sample_paths, new_name):
        self.calls.append(("split_character", character_id, list(sample_paths), new_name))
        return {"character": {"id": "c2"}}

    def review_assign(self, item_key, character_id, **kwargs):
        self.calls.append(("review_assign", item_key, character_id))
        return {"kind": "sample"}

    def review_create_character(self, item_key, name, **kwargs):
        self.calls.append(("review_create_character", item_key, name))
        return {"kind": "sample"}

    def review_discard(self, item_key, **kwargs):
        self.calls.append(("review_discard", item_key))
        return {"kind": "cluster"}


def _cluster_item(face_path: Path) -> dict:
    return {
        "kind": "cluster",
        "key": "c1",
        "character_id": "c1",
        "character_name": "温念",
        "thumb": str(face_path),
        "score": None,
        "source": "聚类「温念」未确认",
    }


def _compare_dialog(tmp_path: Path, item: dict):
    from PySide6.QtWidgets import QApplication
    from recast.desktop.cluster_compare import ClusterCompareDialog

    _app()
    face = _solid_image(tmp_path / "faces" / "face.png", size=120)
    fake = _FakeService(tmp_path, face)
    dlg = ClusterCompareDialog(fake, item if item is not None else _cluster_item(face))
    return fake, dlg


def test_compare_view_shows_samples_side_by_side(tmp_path: Path):
    fake, dlg = _compare_dialog(tmp_path, None)
    assert dlg.samples_list.count() >= 1
    paths = [
        dlg.samples_list.item(r).data(0x0100)  # Qt.ItemDataRole.UserRole
        for r in range(dlg.samples_list.count())
    ]
    assert str(fake.face_path) in paths
    assert not dlg.rep_label.pixmap().isNull()
    assert "不是两两相似度" in dlg.note_label.text()  # 不编相似度分数，写清边界
    dlg.close()


def test_compare_confirm_calls_service(tmp_path: Path):
    from PySide6.QtWidgets import QDialog

    fake, dlg = _compare_dialog(tmp_path, None)
    dlg.confirm_btn.click()
    assert ("confirm_character", "c1") in fake.calls
    assert dlg.result() == QDialog.DialogCode.Accepted
    dlg.close()


def test_compare_split_calls_service_with_selected_samples(tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QDialog
    from PySide6.QtWidgets import QInputDialog

    fake, dlg = _compare_dialog(tmp_path, None)
    dlg.samples_list.item(0).setSelected(True)
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        staticmethod(lambda *args, **kwargs: ("阿新", True)),
    )
    dlg.split_btn.click()
    assert ("split_character", "c1", [str(fake.face_path)], "阿新") in fake.calls
    assert dlg.result() == QDialog.DialogCode.Accepted
    dlg.close()


def test_compare_split_without_selection_stays_open(tmp_path: Path):
    from PySide6.QtWidgets import QDialog

    fake, dlg = _compare_dialog(tmp_path, None)
    dlg.split_btn.click()  # 没点选样本：不调 service、不关对话框
    assert fake.calls == []
    assert dlg.result() != QDialog.DialogCode.Accepted
    assert "先点选" in dlg.status_label.text()
    dlg.close()


def test_compare_discard_calls_service(tmp_path: Path):
    from PySide6.QtWidgets import QDialog

    fake, dlg = _compare_dialog(tmp_path, None)
    dlg.discard_btn.click()
    assert ("review_discard", "c1") in fake.calls
    assert dlg.result() == QDialog.DialogCode.Accepted
    dlg.close()


def test_double_click_review_row_opens_compare_view(tmp_path: Path, monkeypatch):
    """双击队列行 → 打开对比视图（用桩对话框验接线，不真 exec 模态框）。"""
    from PySide6.QtWidgets import QDialog

    import recast.desktop.library_window as lw

    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=0, authorized=0)
    _insert(svc, "温雪", confirmed=0, authorized=0)
    window.reload()
    opened: dict = {}

    class _StubDialog:
        def __init__(self, service, item, parent=None):
            opened["item"] = item

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(lw, "ClusterCompareDialog", _StubDialog)
    row_item = _first_review_data_row(window)
    assert row_item is not None
    window.review_list.itemDoubleClicked.emit(row_item)
    assert opened["item"]["character_name"] in ("温念", "温雪")
    window.close()


def test_double_click_review_header_does_not_open(tmp_path: Path, monkeypatch):
    """双击组头行不打开对比视图（组头是折叠开关，不是数据行）。"""
    import recast.desktop.library_window as lw

    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=0, authorized=0)
    _insert(svc, "温雪", confirmed=0, authorized=0)
    window.reload()
    opened: dict = {}

    class _StubDialog:
        def __init__(self, service, item, parent=None):
            opened["item"] = item

        def exec(self):
            return 1

    monkeypatch.setattr(lw, "ClusterCompareDialog", _StubDialog)
    header = next(
        (
            window.review_list.item(r)
            for r in range(window.review_list.count())
            if window._is_review_header(window.review_list.item(r))
        ),
        None,
    )
    assert header is not None  # 两组 → 有组头
    window.review_list.itemDoubleClicked.emit(header)
    assert opened == {}
    window.close()


# ----- 4. U4：绑定服装二选一（路线 a：实现） ---------------------------------


def _bind_setup(tmp_path: Path, *, with_glb: bool):
    """温念：文件夹 + 共享衣柜红衣（先选衣，此时无 GLB，绑定跳过）。

    返回 (svc, window, folder)。with_glb=True 时再导入一个最小 GLB。
    """
    svc, window = _window(tmp_path)
    ch = _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    folder = Path(svc.settings.output_root) / "characters" / "温念"
    red = _solid_image(Path(svc.settings.output_root) / "wardrobe" / "red.png", size=128)
    svc.set_character_assembly(ch["id"], costume=str(red))  # 无 GLB：绑定跳过
    if with_glb:
        glb = _minimal_glb(tmp_path / "models" / "model.glb")
        svc.import_model_glb(ch["id"], glb)
    window.reload()
    _select(window, 0)
    return svc, window, folder


def test_bind_button_binds_assembled_costume(tmp_path: Path):
    """补绑：衣服先选、GLB 后导入 → 点 [绑定服装到模型] 后 meta.json 有登记，
    装配摘要显示“已绑定”，左栏步骤行变 ✓ 已绑定。"""
    from recast.assets3d.binding import list_bindings

    svc, window, folder = _bind_setup(tmp_path, with_glb=True)
    assert list_bindings(folder) == []  # 选衣时无 GLB，确实还没登记
    assert window.bind_btn.isEnabled()

    window.bind_btn.click()

    bindings = list_bindings(folder)
    assert len(bindings) == 1
    assert bindings[0]["costume"] == "costume/red.png"
    assert bindings[0]["part"] == "全身"
    assert bindings[0]["applied_in_glb"] is False  # 诚实：贴图没写进 GLB 二进制
    assert "已绑定" in window.assembly_summary.text()
    assert "✓ 已绑定" in window.bind_step_label.text()
    assert "已绑定" in window.status_label.text()
    window.close()


def test_bind_button_without_glb_warns_instead_of_silently_failing(tmp_path: Path, monkeypatch):
    """没有 GLB 时点绑定：弹警告说明原因、不登记，按钮不是死按钮。"""
    import recast.desktop.library_window as lw
    from recast.assets3d.binding import list_bindings

    svc, window, folder = _bind_setup(tmp_path, with_glb=False)
    warnings: list[tuple] = []
    monkeypatch.setattr(
        lw.QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args)
    )
    assert window.bind_btn.isEnabled()  # 有衣服可选，按钮是亮的

    window.bind_btn.click()

    assert list_bindings(folder) == []
    assert warnings, "没有 GLB 时应弹警告说明原因"
    assert "还没绑定" in warnings[0][1]
    assert "GLB" in warnings[0][2]
    assert "还没绑定" in window.status_label.text()
    window.close()


def test_bind_button_disabled_without_costume(tmp_path: Path):
    """没选衣服时绑定按钮灰掉，tooltip 写清缺什么。"""
    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    window.reload()
    _select(window, 0)
    assert not window.bind_btn.isEnabled()
    assert "衣服" in window.bind_btn.toolTip()
    assert window.bind_btn.text() == "绑定服装到模型"  # 不再是永久灰的“（未实现）”
    assert "未实现" not in window.bind_btn.text()
    window.close()


# ----- 5. 3D 生成队列接线 ----------------------------------------------------


def test_generate_3d_enqueues_and_shows_queue_dialog(tmp_path: Path, monkeypatch):
    """点 [生成 3D 模型] → 队列对话框出现，队列里有一行该人物（假后端，不真跑 TripoSG）。"""
    svc, window = _window(tmp_path)
    ch = _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    _solid_image(
        Path(svc.settings.output_root) / "characters" / "温念" / "cutout" / "cut.png",
        size=120,
    )
    window._model3d_generate_fn = lambda character, image_ref: {"path": "model/model.glb"}
    monkeypatch.setattr(window, "_generation_blocked", lambda: False)
    window.reload()
    _select(window, 0)

    window._generate_3d()

    dialog = window._model3d_queue_dialog()
    assert dialog is not None
    assert not dialog.isHidden()
    items = dialog.queue_items()
    assert len(items) == 1
    assert items[0]["character"] == "温念"
    assert dialog.wait_until_idle(timeout=10)
    assert dialog.queue_items()[0]["status"] == "完成"
    # 重复点同一个人物不重复排队：还是 1 行。
    window._generate_3d()
    assert dialog.wait_until_idle(timeout=10)
    assert len(dialog.queue_items()) == 1
    dialog.close()
    window.close()


def test_generate_3d_does_nothing_without_selection(tmp_path: Path):
    """没选人物时点生成：不建对话框、不入队。"""
    _svc, window = _window(tmp_path)
    window._generate_3d()
    assert window._model3d_queue_dialog_obj is None
    window.close()


def test_generate_3d_blocked_by_preflight(tmp_path: Path, monkeypatch):
    """前置检查不过（缺 TripoSG）时：不建对话框、不入队。"""
    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    window.reload()
    _select(window, 0)
    monkeypatch.setattr(window, "_generation_blocked", lambda: True)

    window._generate_3d()

    assert window._model3d_queue_dialog_obj is None
    window.close()
