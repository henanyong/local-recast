"""Desktop UI for the P0–P3 character-folder pipeline (offscreen Qt tests)."""

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _face(path: Path) -> Path:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), (180, 140, 120))
    draw = ImageDraw.Draw(img)
    draw.ellipse((16, 10, 48, 50), fill=(210, 170, 150))
    img.save(path)
    return path


def _service(tmp_path: Path):
    from recast.core.service import RecastService

    svc = RecastService(tmp_path / "db.db")
    svc.update_settings(output_root=str(tmp_path / "outputs"))
    return svc


def _unconfirmed(svc, name: str) -> dict:
    return svc.store.insert(
        "characters",
        name=name,
        display_name=name,
        confirmed=0,
        authorized=0,
        cluster_key=f"cluster:{name}",
    )


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _button_texts(window) -> list[str]:
    from PySide6.QtWidgets import QPushButton

    return [b.text() for b in window.findChildren(QPushButton)]


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


def test_pipeline_buttons_present_and_binding_available(tmp_path: Path):
    svc, window = _window(tmp_path)
    labels = _button_texts(window)
    for wanted in (
        "导入视频/照片",
        "自动定义人物",
        "迁移旧资产到人物文件夹",
        "建人物文件夹",
        "抠像",
        "生成 3D 模型",
        "导入 3D 模型",
        "衣服打标签…",
        "试听声音",
        "调节音色",
        "编辑对话",
        "绑定服装到模型",
    ):
        assert wanted in labels, labels
    # P2：绑定按钮不再是永久灰的“（未实现）”死按钮；没选人时灰掉是
    # 因为没衣服可绑（_bind_availability），不是因为后端没实现。
    assert not window.bind_btn.isEnabled()
    assert "未实现" not in window.bind_btn.text()
    window.close()


def test_action_states_follow_selection_and_assets(tmp_path: Path):
    svc, window = _window(tmp_path)
    # No selection: every per-character button is off.
    assert not window.matte_btn.isEnabled()
    assert not window.gen3d_btn.isEnabled()
    assert not window.audition_btn.isEnabled()
    assert not window.auto_define_btn.isEnabled()

    ch = _unconfirmed(svc, "温念")
    window.reload()
    _select(window)
    assert window.build_folder_btn.isEnabled()
    assert not window.matte_btn.isEnabled()

    svc.ensure_character_folder("温念")
    window.reload()
    _select(window)
    assert not window.build_folder_btn.isEnabled()
    assert window.matte_btn.isEnabled()
    assert window.import_glb_btn.isEnabled()
    assert window.audition_btn.isEnabled()
    assert window.timbre_btn.isEnabled()
    assert window.dialogue_btn.isEnabled()
    # No cutout yet: 3D generation stays off.
    assert not window.gen3d_btn.isEnabled()

    _face(Path(svc.settings.output_root) / "characters" / "温念" / "cutout" / "cut.png")
    window.reload()
    _select(window)
    assert window.gen3d_btn.isEnabled()
    window.close()


def test_list_row_shows_folder_status(tmp_path: Path):
    from recast.core.characters import load_meta, save_meta

    svc, window = _window(tmp_path)
    _unconfirmed(svc, "温念")
    folder = Path(svc.ensure_character_folder("温念")["folder"])
    meta = load_meta(folder) or {}
    meta["costume"] = {**meta.get("costume", {}), "description": "红色连衣裙"}
    meta["model"] = {"source": "manual", "format": "glb", "path": "model/model.glb", "note": ""}
    save_meta(folder, meta)
    voice = folder / "voice" / "s1.wav"
    voice.parent.mkdir(parents=True, exist_ok=True)
    voice.write_bytes(b"RIFF....")
    (folder / "model").mkdir(exist_ok=True)
    (folder / "model" / "model.glb").write_bytes(b"glTF")
    window.reload()
    text = window.list.item(0).text()
    assert "衣服：有描述" in text
    assert "1 条声音" in text
    assert "GLB（manual）" in text
    window.close()


def test_matting_button_calls_service_and_reports(tmp_path: Path):
    svc, window = _window(tmp_path)
    _unconfirmed(svc, "温念")
    svc.ensure_character_folder("温念")
    window.reload()
    called: dict = {}

    def fake_matting(cid: str):
        called["cid"] = cid
        return {"outputs": ["a.png", "b.png"], "failed": []}

    svc.run_matting = fake_matting  # type: ignore[method-assign]
    _select(window)
    window.matte_btn.click()
    assert called["cid"] == window._current_id
    assert "抠像完成：成功 2 张" in window.status_label.text()
    window.close()


def test_import_source_enables_auto_define(tmp_path: Path, monkeypatch):
    from recast.desktop import library_window

    svc, window = _window(tmp_path)
    video = tmp_path / "ep1.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(
        library_window.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: (str(video), "")),
    )
    window.import_btn.click()
    assert window._source_path == str(video)
    assert window.auto_define_btn.isEnabled()
    assert "ep1.mp4" in window.source_label.text()
    window.close()


def test_auto_define_runs_diarization_then_reports(tmp_path: Path, monkeypatch):
    from recast.desktop import library_window

    svc, window = _window(tmp_path)
    video = tmp_path / "ep1.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(
        library_window.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: (str(video), "")),
    )
    monkeypatch.setattr(
        library_window.QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *args, **kwargs: ""),
    )
    calls: dict = {}

    def fake_diarization(source, *, character_names=None):
        calls.setdefault("diarization", []).append(character_names)
        return {"segments": 2, "clips": {"人物1": ["a.wav"]}, "skipped": []}

    svc.run_diarization = fake_diarization  # type: ignore[method-assign]
    window.import_btn.click()
    window.auto_define_btn.click()
    assert calls["diarization"]
    assert "说话人分离" in window.status_label.text()
    window.close()


def test_timbre_dialog_saves_values(tmp_path: Path):
    from PySide6.QtWidgets import QDialogButtonBox

    from recast.desktop.timbre_dialog import TimbreDialog

    svc, window = _window(tmp_path)
    _unconfirmed(svc, "温念")
    svc.ensure_character_folder("温念")
    saved: dict = {}

    def fake_set_timbre(character, *, pitch=None, rate=None, style=None):
        saved.update(character=character, pitch=pitch, rate=rate, style=style)
        return {"pitch": pitch, "rate": rate, "style": style}

    svc.set_voice_timbre = fake_set_timbre  # type: ignore[method-assign]
    dlg = TimbreDialog(svc, "温念", window)
    dlg.pitch_slider.setValue(4)
    dlg.rate_slider.setValue(24)
    dlg.style_edit.setText("冷淡")
    box = dlg.findChild(QDialogButtonBox)
    assert box is not None
    save = box.button(QDialogButtonBox.StandardButton.Save)
    assert save is not None
    save.click()
    assert saved == {"character": "温念", "pitch": 2.0, "rate": 1.2, "style": "冷淡"}
    dlg.close()
    window.close()


def _dialogue_workdir(tmp_path: Path) -> Path:
    workdir = tmp_path / "work"
    workdir.mkdir()
    lines = [
        {"shot_code": "F1-L1", "speaker": "温念", "character": "温念", "text": "你好。",
         "start_time": 0.0, "end_time": 3.0, "duration_sec": 3.0, "dirty": False},
        {"shot_code": "F1-L2", "speaker": "陈秋微", "character": "陈秋微", "text": "站住。",
         "start_time": 3.0, "end_time": 6.0, "duration_sec": 3.0, "dirty": False},
    ]
    (workdir / "dialogue.json").write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")
    return workdir


def test_dialogue_dialog_edits_only_character_lines(tmp_path: Path):
    from recast.desktop.dialogue_dialog import DialogueDialog

    svc, window = _window(tmp_path)
    _unconfirmed(svc, "温念")
    svc.ensure_character_folder("温念")
    workdir = _dialogue_workdir(tmp_path)
    dlg = DialogueDialog(svc, "温念", window)
    dlg.workdir_edit.setText(str(workdir))
    dlg._load()
    assert dlg.table.rowCount() == 1
    assert dlg.table.item(0, 0).text() == "F1-L1"
    dlg.table.item(0, 2).setText("你干什么！")
    dlg.save_btn.click()
    saved = json.loads((workdir / "dialogue.json").read_text(encoding="utf-8"))
    changed = next(line for line in saved if line["shot_code"] == "F1-L1")
    assert changed["text"] == "你干什么！"
    assert changed["dirty"] is True
    untouched = next(line for line in saved if line["shot_code"] == "F1-L2")
    assert untouched["text"] == "站住。"
    assert untouched["dirty"] is False
    dlg.close()
    window.close()


def test_dialogue_dialog_redub_reports(tmp_path: Path):
    from recast.desktop.dialogue_dialog import DialogueDialog

    svc, window = _window(tmp_path)
    _unconfirmed(svc, "温念")
    svc.ensure_character_folder("温念")
    workdir = _dialogue_workdir(tmp_path)
    svc.redub_dialogue = lambda wd, **kw: {  # type: ignore[method-assign]
        "redubbed": [{"shot_code": "F1-L1"}],
        "skipped": [],
        "remix": {"mixed": str(workdir / "dialogue_mix.wav")},
    }
    dlg = DialogueDialog(svc, "温念", window)
    dlg._sync_tasks = True
    dlg.workdir_edit.setText(str(workdir))
    dlg._load()
    dlg.redub_btn.click()
    assert "重配 1 句" in dlg.status.text()
    dlg.close()
    window.close()


def test_variant_buttons_follow_folder_and_variants(tmp_path: Path):
    import json as _json
    import struct as _struct

    svc, window = _window(tmp_path)
    assert not window.gen_variant_btn.isEnabled()
    assert not window.switch_variant_btn.isEnabled()
    _unconfirmed(svc, "温念")
    svc.ensure_character_folder("温念")
    window.reload()
    _select(window)
    assert window.gen_variant_btn.isEnabled()
    assert not window.switch_variant_btn.isEnabled()
    assert "不是独立衣片" in window.gen_variant_btn.toolTip()
    # Register one variant (core call): the switch button lights up.
    folder = Path(svc.settings.output_root) / "characters" / "温念"
    payload = _json.dumps({"asset": {"version": "2.0"}}).encode()
    payload += b" " * ((-len(payload)) % 4)
    body = _struct.pack("<II", len(payload), 0x4E4F534A) + payload
    src = tmp_path / "red.glb"
    src.write_bytes(_struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
    from recast.assets3d.variants import register_costume_variant

    register_costume_variant(folder, name="红裙", glb_path=src, backend="stub")
    window.reload()
    _select(window)
    assert window.switch_variant_btn.isEnabled()
    window.close()


def test_switch_variant_button_switches_assembly(tmp_path: Path, monkeypatch):
    import json as _json
    import struct as _struct

    from recast.desktop import library_window

    svc, window = _window(tmp_path)
    _unconfirmed(svc, "温念")
    folder = Path(svc.ensure_character_folder("温念")["folder"])
    payload = _json.dumps({"asset": {"version": "2.0"}}).encode()
    payload += b" " * ((-len(payload)) % 4)
    body = _struct.pack("<II", len(payload), 0x4E4F534A) + payload
    glb = _struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body
    (folder / "model" / "model.glb").write_bytes(glb)
    src = tmp_path / "red.glb"
    src.write_bytes(glb)
    from recast.assets3d.variants import register_costume_variant, switch_costume_variant

    register_costume_variant(folder, name="红裙", glb_path=src, backend="stub")
    window.reload()
    _select(window)
    monkeypatch.setattr(
        library_window.QInputDialog,
        "getItem",
        staticmethod(lambda *args, **kwargs: ("红裙", True)),
    )
    window.switch_variant_btn.click()
    from recast.core.characters import get_assembly, load_meta

    assert get_assembly(load_meta(folder))["body"] == "model/variants/红裙.glb"
    # Status text is overwritten by the background preview render, so assert
    # on the assembly summary instead (it names the worn variant).
    assert "红裙" in window.assembly_summary.text()
    assert "换装变体" in window.assembly_summary.text()
    window.close()


def test_main_window_app_layer_buttons(tmp_path: Path):
    from recast.desktop.main_window import MainWindow

    _app()
    svc = _service(tmp_path)
    svc.update_settings(first_run_done=True)
    window = MainWindow(svc)
    window.timer.stop()  # 停掉 400ms 定时器，避免泄漏的窗口在后续测试里继续刷新。
    labels = _button_texts(window)
    # P1 入口整理：入口文案写清区别（只换脸=跳过前面步骤），旧名“换脸重制”已改。
    assert "只换脸（跳过前面步骤）" in labels
    assert "导出剪映" in labels
    assert "按剧本生成新集…" in labels
    calls: list = []
    svc.start = lambda ids, stages, **kw: calls.append((ids, stages))  # type: ignore[method-assign]
    window._drama_ids_checked = lambda: ["d1"]  # type: ignore[method-assign]
    from PySide6.QtWidgets import QPushButton

    by_text = {b.text(): b for b in window.findChildren(QPushButton)}
    by_text["只换脸（跳过前面步骤）"].click()
    by_text["导出剪映"].click()
    assert calls == [(["d1"], ["replace"]), (["d1"], ["export"])]
    window.close()
