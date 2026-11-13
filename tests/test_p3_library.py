"""P3 人物库包：N26 变体缩略图/删除、N27 详情卡直达按钮、N37 docstring、
N31 3D 生成完成通知 + 库刷新、N20 别名表（人物库 UI 侧）。

每项 >= 2 个测试（正向 + 边界/失败）。全部用离屏 Qt + 真实
RecastService（tmp_path），不真跑 TripoSG、不真改用户库。
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

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
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)
    return path


def _insert(svc, name: str, *, confirmed: int = 1, authorized: int = 0) -> dict:
    return svc.store.insert(
        "characters",
        name=name,
        display_name=name,
        confirmed=confirmed,
        authorized=authorized,
        cluster_key=f"cluster:{name}",
    )


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


def _folder(svc, name: str) -> Path:
    return Path(svc.settings.output_root) / "characters" / name


def _register_variant(svc, name: str, variant: str, *, with_source: bool = True):
    """给人物登记一个换装变体。源图放人物文件夹内，缩略图才能解析到。"""
    from recast.assets3d.variants import register_costume_variant

    folder = _folder(svc, name)
    glb = _minimal_glb(Path(svc.settings.output_root) / "tmp-models" / f"{variant}.glb")
    source = None
    if with_source:
        source = _solid_image(folder / "costume" / f"{variant}.png", size=120)
    return register_costume_variant(
        folder, name=variant, glb_path=glb, source_image=source
    )


def _variant_delete_buttons(window):
    """详情区里全部 [删除变体] 按钮。"""
    from PySide6.QtWidgets import QPushButton

    out = []
    for i in range(window.detail_images_layout.count()):
        wrap = window.detail_images_layout.itemAt(i).widget()
        if wrap is None:
            continue
        for btn in wrap.findChildren(QPushButton):
            if btn.text() == "删除变体":
                out.append(btn)
    return out


def _variant_thumb_labels(window):
    """详情区里变体缩略图 QLabel（变体行独有的 140x160 图）。"""
    from PySide6.QtWidgets import QLabel

    out = []
    for i in range(window.detail_images_layout.count()):
        wrap = window.detail_images_layout.itemAt(i).widget()
        if wrap is None:
            continue
        has_delete = any(
            b.text() == "删除变体" for b in wrap.findChildren(QPushButton)
        )
        if not has_delete:
            continue
        for lab in wrap.findChildren(QLabel):
            if lab.minimumSize() == lab.maximumSize() or lab.size() == lab.maximumSize():
                out.append(lab)
    return out


def _detail_variant_labels(window):
    """详情区里变体行的全部 QLabel 文本（缩略图占位文字也在内）。"""
    from PySide6.QtWidgets import QLabel, QPushButton

    texts = []
    for i in range(window.detail_images_layout.count()):
        wrap = window.detail_images_layout.itemAt(i).widget()
        if wrap is None:
            continue
        if not any(b.text() == "删除变体" for b in wrap.findChildren(QPushButton)):
            continue
        texts.extend(lab.text() for lab in wrap.findChildren(QLabel))
    return texts


# ----- 1. N26：变体缩略图 / 删除 --------------------------------------------


def test_n26_variant_row_shows_thumbnail_and_delete(tmp_path: Path, monkeypatch):
    """变体行有真实缩略图（源图）+ [删除变体]；点删除要确认，删完 GLB
    文件和 meta 登记都没了，列表即时更新。"""
    import recast.desktop.library_window as lw
    from recast.assets3d.variants import list_costume_variants
    from recast.core.characters import load_meta

    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    report = _register_variant(svc, "温念", "红裙", with_source=True)
    variant_glb = Path(report["path"])
    assert variant_glb.is_file()
    window.reload()
    _select(window, 0)

    assert len(list_costume_variants(_folder(svc, "温念"))) == 1
    assert "红裙" in window.detail_label.text()
    delete_btns = _variant_delete_buttons(window)
    assert len(delete_btns) == 1
    # 缩略图是源图的真实像素，不是占位文字。
    thumb_texts = _detail_variant_labels(window)
    assert "无预览图" not in thumb_texts
    assert any("变体：红裙" in t for t in thumb_texts)

    monkeypatch.setattr(
        lw.QMessageBox, "question",
        lambda *args, **kwargs: lw.QMessageBox.StandardButton.Yes,
    )
    delete_btns[0].click()

    assert not variant_glb.exists()
    assert list_costume_variants(_folder(svc, "温念")) == []
    assert (load_meta(_folder(svc, "温念")) or {}).get("model", {}).get("variants") == []
    assert "已删除换装变体「红裙」" in window.status_label.text()
    assert _variant_delete_buttons(window) == []  # 删完列表即时更新
    window.close()


def test_n26_variant_without_source_image_shows_placeholder(tmp_path: Path):
    """源图登记不到（文件夹外选图只存了文件名，或根本没源图）时，
    变体行显示“无预览图”占位，不拿主模型预览图冒充。"""
    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    _register_variant(svc, "温念", "蓝衫", with_source=False)
    window.reload()
    _select(window, 0)

    assert len(_variant_delete_buttons(window)) == 1
    assert "无预览图" in _detail_variant_labels(window)
    assert "变体：蓝衫" in " ".join(_detail_variant_labels(window))
    window.close()


def test_n26_delete_cancel_keeps_variant(tmp_path: Path, monkeypatch):
    """确认框点“否”：文件和登记都留着，什么都不删。"""
    import recast.desktop.library_window as lw
    from recast.assets3d.variants import list_costume_variants

    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    report = _register_variant(svc, "温念", "红裙", with_source=True)
    window.reload()
    _select(window, 0)

    monkeypatch.setattr(
        lw.QMessageBox, "question",
        lambda *args, **kwargs: lw.QMessageBox.StandardButton.No,
    )
    _variant_delete_buttons(window)[0].click()

    assert Path(report["path"]).is_file()
    assert len(list_costume_variants(_folder(svc, "温念"))) == 1
    assert "已删除" not in window.status_label.text()
    window.close()


def test_n26_delete_active_variant_switches_back_to_main(tmp_path: Path, monkeypatch):
    """删掉当前穿的变体：active_variant 清掉，装配身体切回主模型，
    不留指向已删文件的装配。"""
    import recast.desktop.library_window as lw
    from recast.assets3d.variants import list_costume_variants
    from recast.core.characters import get_assembly, load_meta

    svc, window = _window(tmp_path)
    ch = _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    glb = _minimal_glb(tmp_path / "models" / "model.glb")
    svc.import_model_glb(ch["id"], glb)
    _register_variant(svc, "温念", "红裙", with_source=True)
    svc.switch_costume_variant(ch["id"], variant_name="红裙")
    folder = _folder(svc, "温念")
    assert (load_meta(folder) or {}).get("model", {}).get("active_variant") == "红裙"
    window.reload()
    _select(window, 0)
    assert "当前穿" in " ".join(_detail_variant_labels(window))

    monkeypatch.setattr(
        lw.QMessageBox, "question",
        lambda *args, **kwargs: lw.QMessageBox.StandardButton.Yes,
    )
    _variant_delete_buttons(window)[0].click()

    meta = load_meta(folder) or {}
    assert meta.get("model", {}).get("variants") == []
    assert meta.get("model", {}).get("active_variant") in ("", None)
    assert get_assembly(meta).get("body") == "model/model.glb"
    assert list_costume_variants(folder) == []
    assert "已删除换装变体「红裙」" in window.status_label.text()
    assert "切回主模型" in window.status_label.text()
    window.close()


def test_n26_delete_missing_file_only_clears_registration(tmp_path: Path, monkeypatch):
    """变体 GLB 文件已经不在磁盘上：只清登记，不报错、不弹警告。"""
    import recast.desktop.library_window as lw
    from recast.assets3d.variants import list_costume_variants

    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    report = _register_variant(svc, "温念", "红裙", with_source=True)
    Path(report["path"]).unlink()  # 文件先丢了
    window.reload()
    _select(window, 0)

    warnings: list = []
    monkeypatch.setattr(lw.QMessageBox, "question",
                        lambda *args, **kwargs: lw.QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(lw.QMessageBox, "warning",
                        lambda *args, **kwargs: warnings.append(args))
    _variant_delete_buttons(window)[0].click()

    assert list_costume_variants(_folder(svc, "温念")) == []
    assert warnings == []
    assert "只清了登记" in window.status_label.text()
    window.close()


# ----- 2. N27：详情卡缺项直达按钮 ------------------------------------------


def _detail_buttons(window):
    return [(b.text(), b.isEnabled()) for b in window.detail_action_buttons]


def _complete_character(svc, name: str = "温念"):
    """什么都不缺的人物：文件夹 + 主模型 + 声音样本 + 抠像 + 已授权。"""
    from recast.core.characters import load_meta, save_meta
    from recast.core.constants import AUTH_TYPES

    ch = _insert(svc, name, confirmed=1, authorized=0)
    svc.ensure_character_folder(name)
    folder = _folder(svc, name)
    glb = _minimal_glb(Path(svc.settings.output_root) / "tmp-models" / "model.glb")
    svc.import_model_glb(ch["id"], glb)
    _solid_image(folder / "cutout" / "cut.png", size=120)
    voice = folder / "voice" / "s1.wav"
    voice.parent.mkdir(parents=True, exist_ok=True)
    voice.write_bytes(b"RIFF" + b"\x00" * 40)
    meta = load_meta(folder) or {}
    meta["voice"] = {"files": [str(voice)]}
    save_meta(folder, meta)
    svc.confirm_character(ch["id"], authorized=True, auth_type=AUTH_TYPES[0])
    return ch


def test_n27_detail_buttons_for_every_gap(tmp_path: Path):
    """缺模型/缺声音/缺抠像/缺授权 → 四个直达按钮都在；没抠像时
    [生成 3D 模型] 是灰的（tooltip 写清先抠像），不是点了才失败的按钮。"""
    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=0)
    svc.ensure_character_folder("温念")
    window.reload()
    _select(window, 0)

    buttons = _detail_buttons(window)
    assert [t for t, _ in buttons] == ["生成 3D 模型", "试听/导入声音", "抠像", "去补授权"]
    assert buttons[0][1] is False  # 没抠像：生成按钮灰掉
    assert "先抠像" in window.detail_action_buttons[0].toolTip()
    assert all(enabled for _, enabled in buttons[1:])
    window.close()


def test_n27_no_buttons_when_nothing_missing(tmp_path: Path):
    """什么都不缺：详情卡一个直达按钮都不出（纯展示，不堆按钮）。"""
    svc, window = _window(tmp_path)
    _complete_character(svc)
    window.reload()
    _select(window, 0)

    assert _detail_buttons(window) == []
    assert "无 3D 模型" not in window.detail_label.text()
    window.close()


def test_n27_no_folder_only_build_folder_button(tmp_path: Path):
    """还没建文件夹：只出 [建人物文件夹]，点它就建出来（走已有入口）。"""
    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=0)
    window.reload()
    _select(window, 0)

    assert _detail_buttons(window) == [("建人物文件夹", True)]
    window.detail_action_buttons[0].click()
    assert _folder(svc, "温念").is_dir()
    assert "已建人物文件夹" in window.status_label.text()
    window.close()


def test_n27_go_authorize_focuses_auth_checkbox(tmp_path: Path):
    """点 [去补授权]：焦点落到右侧「已授权」勾选框，状态栏写清下一步；
    不替用户勾选（授权是用户自己的决定）。"""
    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=0)
    svc.ensure_character_folder("温念")
    window.reload()
    _select(window, 0)

    assert not window.auth.isChecked()
    btn = next(b for b in window.detail_action_buttons if b.text() == "去补授权")
    btn.click()
    assert "勾选「已授权」" in window.status_label.text()
    assert "保存确认" in window.status_label.text()
    assert not window.auth.isChecked()  # 没替用户勾
    window.close()


def test_n27_generate_model_button_enqueues(tmp_path: Path, monkeypatch):
    """详情卡 [生成 3D 模型] 点下去就是已有的生成入口：进 3D 队列。"""
    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    _solid_image(_folder(svc, "温念") / "cutout" / "cut.png", size=120)
    window._model3d_generate_fn = lambda character, image_ref: {"path": "model/model.glb"}
    monkeypatch.setattr(window, "_generation_blocked", lambda: False)
    window.reload()
    _select(window, 0)

    btn = next(b for b in window.detail_action_buttons if b.text() == "生成 3D 模型")
    assert btn.isEnabled()
    btn.click()

    dialog = window._model3d_queue_dialog()
    assert [it["character"] for it in dialog.queue_items()] == ["温念"]
    assert dialog.wait_until_idle(timeout=10)
    assert dialog.queue_items()[0]["status"] == "完成"
    dialog.close()
    window.close()


# ----- 3. N37：进度 docstring 过时注释 --------------------------------------


def test_n37_review_progress_docstring_no_stale_note(tmp_path: Path):
    """docstring 里不再写主窗口是老分母；分母逻辑本身没动。"""
    from recast.desktop.library_window import LibraryWindow

    doc = LibraryWindow._review_progress_stats.__doc__ or ""
    assert "仍是老分母" not in doc
    assert "另一波" not in doc
    assert "total = 已确认 + 待审核全部项" in doc

    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    _insert(svc, "温雪", confirmed=0, authorized=0)
    window.reload()
    stats = window._review_progress_stats()
    assert stats["total"] == stats["confirmed"] + stats["pending"]
    assert stats == {
        "confirmed": 1,
        "pending": 1,
        "clusters": 1,
        "low": 0,
        "total": 2,
        "percent": 50,
    }
    window.close()


# ----- 4. N31：3D 生成完成通知 + 库刷新 -------------------------------------


def _queue_dialog(*, generate_fn=None, on_item_finished=None, on_close=None):
    from recast.desktop.model3d_queue import Model3dQueueDialog

    _app()
    return Model3dQueueDialog(
        service=None,
        generate_fn=generate_fn,
        on_item_finished=on_item_finished,
        on_close=on_close,
    )


def _wait_until(pred, timeout: float = 10.0) -> bool:
    import time

    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        QApplication.processEvents()
        time.sleep(0.02)
    QApplication.processEvents()
    return pred()


def test_n31_finished_callback_fires_on_done():
    """一行生成完：on_item_finished 被调一次（人物， 完成， 空原因）。"""
    import time

    from PySide6.QtWidgets import QApplication

    calls: list[tuple] = []
    dialog = _queue_dialog(
        generate_fn=lambda c, i: {"path": "model/model.glb"},
        on_item_finished=lambda c, s, e: calls.append((c, s, e)),
    )
    dialog.enqueue("温念")
    assert dialog.wait_until_idle(timeout=10.0)
    QApplication.processEvents()
    time.sleep(0.05)
    QApplication.processEvents()
    assert calls == [("温念", "完成", "")]
    assert dialog.pending_count() == 0
    dialog.close()


def test_n31_finished_callback_fires_on_failure_with_error():
    """一行生成失败：回调带失败状态 + 真实原因，不弹框（回调里没地方弹）。"""
    calls: list[tuple] = []
    dialog = _queue_dialog(
        generate_fn=lambda c, i: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
        on_item_finished=lambda c, s, e: calls.append((c, s, e)),
    )
    dialog.enqueue("温念")
    assert dialog.wait_until_idle(timeout=10.0)
    assert calls == [("温念", "失败", "CUDA out of memory")]
    assert dialog.queue_items()[0]["status"] == "失败"
    dialog.close()


def test_n31_library_status_on_done_and_auto_reload(tmp_path: Path, monkeypatch):
    """人物库侧：生成完成 → 状态栏“已生成，人物库已刷新”，列表还在。"""
    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    _solid_image(_folder(svc, "温念") / "cutout" / "cut.png", size=120)
    window._model3d_generate_fn = lambda character, image_ref: {"path": "model/model.glb"}
    monkeypatch.setattr(window, "_generation_blocked", lambda: False)
    window.reload()
    _select(window, 0)

    window._generate_3d()
    dialog = window._model3d_queue_dialog()
    assert dialog.wait_until_idle(timeout=10.0)
    assert _wait_until(lambda: "3D 模型已生成" in window.status_label.text())
    assert "人物库已刷新" in window.status_label.text()
    assert window.list.count() == 1  # reload 后列表还在
    dialog.close()
    window.close()


def test_n31_library_status_on_failure_points_to_retry(tmp_path: Path, monkeypatch):
    """人物库侧：生成失败 → 状态栏写失败原因 + 指到队列对话框 [重试]。"""
    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    _solid_image(_folder(svc, "温念") / "cutout" / "cut.png", size=120)

    def _boom(character, image_ref):
        raise RuntimeError("TripoSG 生成失败（退出码 1）")

    window._model3d_generate_fn = _boom
    monkeypatch.setattr(window, "_generation_blocked", lambda: False)
    window.reload()
    _select(window, 0)

    window._generate_3d()
    dialog = window._model3d_queue_dialog()
    assert dialog.wait_until_idle(timeout=10.0)
    assert _wait_until(lambda: "3D 生成失败" in window.status_label.text())
    assert "退出码 1" in window.status_label.text()
    assert "[重试]" in window.status_label.text()
    dialog.close()
    window.close()


def test_n31_close_with_pending_leaves_status_and_worker_continues(tmp_path: Path, monkeypatch):
    """关窗时队列未空：状态栏留“进行中（n 个排队），完成后在人物库刷新可见”；
    且关窗只是隐藏——放行后 worker 继续跑完（P2“关窗不继续”的说法是错的）。"""
    import threading

    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    _solid_image(_folder(svc, "温念") / "cutout" / "cut.png", size=120)
    release = threading.Event()
    window._model3d_generate_fn = lambda c, i: (
        release.wait(timeout=10), {"path": "model/model.glb"}
    )[1]
    monkeypatch.setattr(window, "_generation_blocked", lambda: False)
    window.reload()
    _select(window, 0)

    window._generate_3d()
    dialog = window._model3d_queue_dialog()
    assert _wait_until(lambda: dialog.queue_items()[0]["status"] == "生成中")
    assert dialog.pending_count() == 1

    dialog.close()
    assert "3D 生成进行中（1 个排队），完成后在人物库刷新可见" in window.status_label.text()
    assert window._model3d_queue_dialog_obj is dialog  # 缓存还在，没销毁

    release.set()
    assert dialog.wait_until_idle(timeout=10.0)
    assert dialog.queue_items()[0]["status"] == "完成"
    assert "3D 模型已生成" in window.status_label.text()
    window.close()


def test_n31_close_empty_queue_no_status_line(tmp_path: Path):
    """队列已空时关窗：不留“进行中”提示，不打扰。"""
    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    window.reload()
    _select(window, 0)
    window.status_label.setText("就绪。")

    dialog = window._model3d_queue_dialog()
    assert dialog.pending_count() == 0
    dialog.close()
    assert "进行中" not in window.status_label.text()
    assert window.status_label.text() == "就绪。"
    window.close()


# ----- 5. N20：别名表（人物库 UI 侧） ---------------------------------------


def _set_aliases(svc, name: str, aliases: list[str]) -> None:
    from recast.core.characters import load_meta, save_meta

    folder = _folder(svc, name)
    meta = load_meta(folder) or {}
    meta["aliases"] = list(aliases)
    save_meta(folder, meta)


def test_n20_aliases_shown_and_add_alias(tmp_path: Path, monkeypatch):
    """别名可查看（基本信息区列出来）+ 可手动加（写进 meta.json aliases）。"""
    from PySide6.QtWidgets import QInputDialog

    from recast.core.characters import load_meta

    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    _set_aliases(svc, "温念", ["阿念"])
    window.reload()
    _select(window, 0)

    assert window.aliases_label.text() == "别名：阿念"

    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(lambda *args, **kwargs: ("念念", True)),
    )
    window.add_alias_btn.click()

    assert (load_meta(_folder(svc, "温念")) or {}).get("aliases") == ["阿念", "念念"]
    assert window.aliases_label.text() == "别名：阿念、念念"
    assert "加别名「念念」" in window.status_label.text()
    window.close()


def test_n20_add_duplicate_alias_ignored(tmp_path: Path, monkeypatch):
    """重复加同一个别名：meta 不变，状态栏说已经在别名里了。"""
    from PySide6.QtWidgets import QInputDialog

    from recast.core.characters import load_meta

    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    _set_aliases(svc, "温念", ["阿念"])
    window.reload()
    _select(window, 0)

    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(lambda *args, **kwargs: ("阿念", True)),
    )
    window.add_alias_btn.click()

    assert (load_meta(_folder(svc, "温念")) or {}).get("aliases") == ["阿念"]
    assert "已经在别名里了" in window.status_label.text()
    window.close()


def test_n20_add_alias_equal_to_current_name_rejected(tmp_path: Path, monkeypatch):
    """别名就是当前名字：不写，别名表保持干净。"""
    from PySide6.QtWidgets import QInputDialog

    from recast.core.characters import load_meta

    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    window.reload()
    _select(window, 0)
    assert window.aliases_label.text() == "别名：无"

    monkeypatch.setattr(
        QInputDialog, "getText",
        staticmethod(lambda *args, **kwargs: ("温念", True)),
    )
    window.add_alias_btn.click()

    assert (load_meta(_folder(svc, "温念")) or {}).get("aliases") in (None, [])
    assert "就是当前名字" in window.status_label.text()
    window.close()


def test_n20_rename_writes_old_name_to_aliases(tmp_path: Path):
    """改名入口在人物库（_save）：旧名写进该人物的 aliases，新名进数据库。"""
    from recast.core.characters import load_meta
    from recast.core.constants import AUTH_TYPES

    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    window.reload()
    _select(window, 0)
    # 已授权的人物保存时必须有授权类型，否则 confirm_character 会拒存。
    window.auth_type.setCurrentIndex(window.auth_type.findData(AUTH_TYPES[0]))

    window.name_edit.setText("温念新")
    window._save()

    assert (load_meta(_folder(svc, "温念")) or {}).get("aliases") == ["温念"]
    assert window.aliases_label.text() == "别名：温念"
    assert svc.store.get_character(window._current_id)["display_name"] == "温念新"
    window.close()


def test_n20_rename_same_name_writes_no_alias(tmp_path: Path):
    """没改名就点保存：别名表不动，不把当前名字自己记成别名。"""
    from recast.core.characters import load_meta
    from recast.core.constants import AUTH_TYPES

    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    svc.ensure_character_folder("温念")
    window.reload()
    _select(window, 0)
    window.auth_type.setCurrentIndex(window.auth_type.findData(AUTH_TYPES[0]))

    window._save()  # 名字没动

    assert (load_meta(_folder(svc, "温念")) or {}).get("aliases") in (None, [])
    assert window.aliases_label.text() == "别名：无"
    window.close()


def test_n20_add_alias_without_folder(tmp_path: Path):
    """没建文件夹时点 [加别名…]：状态栏说先建文件夹，不崩。"""
    svc, window = _window(tmp_path)
    _insert(svc, "温念", confirmed=1, authorized=1)
    window.reload()
    _select(window, 0)

    window.add_alias_btn.click()

    assert "先建人物文件夹" in window.status_label.text()
    window.close()
