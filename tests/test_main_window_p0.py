"""Main-window P0 batch: one-click, step 5 audio/dialogue, retry, adaptive
stages, unified numbering, import pre-check, audition stand, task history.

Only synthetic data + stub service calls: no real video, no TTS engines, no
network. Follows the fixture pattern of tests/test_guide_review_ui.py.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def svc(tmp_path: Path):
    from recast.core.service import RecastService

    out = tmp_path / "out"
    works = tmp_path / "works"
    out.mkdir()
    works.mkdir()
    service = RecastService(tmp_path / "test.db")
    service.update_settings(output_root=str(out), works_root=str(works), first_run_done=True)
    return service


def _main(svc):
    from recast.desktop.main_window import MainWindow

    _app()
    return MainWindow(svc)


def _library(svc):
    from recast.desktop.library_window import LibraryWindow

    _app()
    return LibraryWindow(svc)


def _paint_main(window, svc):
    """Drive one synchronous paint with a real snapshot (no event loop)."""
    window._paint(svc.snapshot(), full=True)


def _add_drama(svc, name: str = "剧", *, entry_type: str = "import") -> dict:
    return svc.store.insert(
        "dramas", origin_name=name, root_path=f"/tmp/{name}", episode_count=1,
        entry_type=entry_type,
    )


# ----- 第 1 项：一键成片 -----

def _add_task(svc, drama_id, *, status: str = "待执行", workdir=None):
    return svc.store.insert(
        "tasks", drama_id=drama_id, episode_name="第1集",
        video_path="/tmp/ep1.mp4", status=status, workdir=workdir,
    )


def _add_character(svc, name: str, *, confirmed: int = 0):
    return svc.store.insert(
        "characters", name=name, display_name=name, confirmed=confirmed,
        authorized=0, cluster_key=f"test:{name}",
    )


def _add_identity_checkpoint(svc, task_id: str):
    svc.store.insert(
        "checkpoints", task_id=task_id, step="identity",
        input_hashes="[]", output_paths="[]",
    )


def test_one_click_stops_at_review_gate(svc, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    drama = _add_drama(svc, "剧", entry_type="import")
    _add_task(svc, drama["id"])
    _add_character(svc, "人物1", confirmed=0)
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    shown: list = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2] if len(a) > 2 else ""))
    opened: list = []
    monkeypatch.setattr(window, "_open_library", lambda: opened.append(True))
    window.oneclick_btn.click()
    assert calls == []
    assert shown and "审核" in shown[0]
    assert opened == [True]
    assert window._oneclick is None
    window.close()


def test_one_click_starts_phase_a(svc, monkeypatch):
    drama = _add_drama(svc, "剧", entry_type="import")
    _add_task(svc, drama["id"])
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    window.oneclick_btn.click()
    assert len(calls) == 1
    assert calls[0][0][0] == [drama["id"]]
    assert calls[0][0][1] == ["scan", "extract", "library"]
    assert window._oneclick["phase"] == "A"
    assert "前期" in window.oneclick_label.text()
    window.close()


def test_one_click_starts_phase_b_when_phase_a_done(svc, monkeypatch):
    drama = _add_drama(svc, "剧", entry_type="import")
    task = _add_task(svc, drama["id"])
    _add_identity_checkpoint(svc, task["id"])
    _add_character(svc, "温念", confirmed=1)
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    window.oneclick_btn.click()
    assert len(calls) == 1
    assert calls[0][0][1] == ["replace", "compose", "export"]
    assert window._oneclick["phase"] == "B"
    window.close()


def test_one_click_advances_after_phase_a(svc, monkeypatch):
    drama = _add_drama(svc, "剧", entry_type="import")
    task = _add_task(svc, drama["id"])
    _add_identity_checkpoint(svc, task["id"])
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    window._oneclick = {
        "drama_id": drama["id"], "phase": "A", "entry": "import",
        "prompted_review": False, "seen_running": True,
    }
    window._maybe_advance_one_click({"scheduler": {"running": False}})
    assert any(c[0][1] == ["replace", "compose", "export"] for c in calls)
    assert window._oneclick["phase"] == "B"
    window.close()


def test_one_click_stops_at_review_after_phase_a(svc, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    drama = _add_drama(svc, "剧", entry_type="import")
    task = _add_task(svc, drama["id"])
    _add_identity_checkpoint(svc, task["id"])
    _add_character(svc, "人物1", confirmed=0)
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    shown: list = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2] if len(a) > 2 else ""))
    opened: list = []
    monkeypatch.setattr(window, "_open_library", lambda: opened.append(True))
    window._oneclick = {
        "drama_id": drama["id"], "phase": "A", "entry": "import",
        "prompted_review": False, "seen_running": True,
    }
    window._maybe_advance_one_click({"scheduler": {"running": False}})
    assert calls == []
    assert window._oneclick["phase"] == "review"
    assert "审核关口" in window.oneclick_label.text()
    assert shown and "待审核" in shown[0]
    assert opened == [True]
    window.close()


def test_one_click_reports_failure(svc):
    drama = _add_drama(svc, "剧", entry_type="import")
    _add_task(svc, drama["id"], status="失败")
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    window._oneclick = {
        "drama_id": drama["id"], "phase": "B", "entry": "import",
        "prompted_review": False, "seen_running": True,
    }
    window._maybe_advance_one_click({"scheduler": {"running": False}})
    assert window._oneclick["phase"] == "failed"
    assert "失败" in window.oneclick_label.text()
    # P1 入口整理：失败提示改成“从失败步续跑”，与“新建任务重跑”区分。
    assert "从失败步续跑" in window.oneclick_label.text()
    window.close()


def test_one_click_completes(svc):
    drama = _add_drama(svc, "剧", entry_type="import")
    _add_task(svc, drama["id"], status="已完成")
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    window._oneclick = {
        "drama_id": drama["id"], "phase": "B", "entry": "import",
        "prompted_review": False, "seen_running": True,
    }
    window._maybe_advance_one_click({"scheduler": {"running": False}})
    assert window._oneclick is None
    assert "完成" in window.oneclick_label.text()
    window.close()


def test_one_click_resumes_after_retry(svc, monkeypatch):
    drama = _add_drama(svc, "剧", entry_type="import")
    task = _add_task(svc, drama["id"])
    _add_identity_checkpoint(svc, task["id"])
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    window._oneclick = {
        "drama_id": drama["id"], "phase": "failed", "failed_phase": "A",
        "entry": "import", "prompted_review": False, "seen_running": True,
    }
    window._maybe_advance_one_click({"scheduler": {"running": False}})
    assert any(c[0][1] == ["replace", "compose", "export"] for c in calls)
    assert window._oneclick["phase"] == "B"
    window.close()


# ----- 第 2 项：第 ⑤ 步声音与台词 -----

def _make_voice_sample(svc, name: str, filename: str = "SPK_1_0.00_2.50.wav") -> Path:
    from recast.core.characters import ensure_character_folder, load_meta, save_meta

    folder, _meta = ensure_character_folder(svc.settings.output_root, name)
    voice_dir = folder / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    clip = voice_dir / filename
    clip.write_bytes(b"RIFF" + b"\x00" * 40)
    meta = load_meta(folder) or {}
    voice = meta.get("voice") if isinstance(meta.get("voice"), dict) else {}
    samples = [str(s) for s in (voice.get("samples") or [])]
    if str(clip) not in samples:
        samples.append(str(clip))
    meta["voice"] = {**voice, "samples": samples}
    save_meta(folder, meta)
    return clip


def test_guide_list_has_five_steps(svc):
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    assert window.guide_list.count() == 5
    labels = [window.guide_list.item(i).text() for i in range(5)]
    assert any("⑤" in t and "声音与台词" in t for t in labels)
    assert window.audio_panel.isHidden()
    window.close()


def test_audio_step_state_machine(svc):
    # N1: 第 ⑤ 步已进 guide.py 状态机，不再是主窗口外挂——状态走
    # service.guide_state()，且从不锁住。
    _add_drama(svc, "剧")
    _add_character(svc, "温念", confirmed=1)
    window = _main(svc)
    window.timer.stop()

    def _audio():
        return next(s for s in svc.guide_state() if s["id"] == "audio")

    assert _audio()["state"] == "todo"
    assert _audio()["state"] != "locked"
    # 审核队列非空时第 ④ 步锁住，第 ⑤ 步仍是 todo，绝不锁住。
    _add_character(svc, "路人", confirmed=0)
    assert _audio()["state"] == "todo"
    assert _audio()["state"] != "locked"
    # 第 ④ 步完成 → 第 ⑤ 步变 active（没有混音/台词表，不算 done）。
    drama = svc.store.list_dramas()[0]
    _add_task(svc, drama["id"], status="已完成")
    assert _audio()["state"] == "active"
    window.close()


def test_audio_step_done_when_mix_exists(svc, tmp_path: Path):
    drama = _add_drama(svc, "剧", entry_type="import")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "dialogue_mix.wav").write_bytes(b"RIFF" + b"\x00" * 10)
    _add_task(svc, drama["id"], status="已完成", workdir=str(workdir))
    window = _main(svc)
    window.timer.stop()
    assert window._audio_done() is True
    audio = next(s for s in svc.guide_state() if s["id"] == "audio")
    assert audio["state"] == "done"
    window.close()


def test_audio_panel_shows_segments(svc):
    _add_character(svc, "温念", confirmed=1)
    clip = _make_voice_sample(svc, "温念")
    window = _main(svc)
    window.timer.stop()
    window._show_audio_panel()
    assert not window.audio_panel.isHidden()
    assert window.diar_table.rowCount() == 1
    assert window.diar_table.item(0, 0).text() == "0.00s"
    assert window.diar_table.item(0, 1).text() == "2.50s"
    assert window.diar_table.item(0, 2).text() == "SPK_1"
    combo = window.diar_table.cellWidget(0, 3)
    assert combo is not None
    assert combo.currentText() == "温念"
    assert window.audio_status.text().startswith("分轨结果共 1 段")
    # 每个人物一行：试听 / 调音色 / 改台词。
    btns = window._audio_char_btns
    assert len(btns) == 1
    row = next(iter(btns.values()))
    assert row["audition"].text() == "试听 3 秒"
    assert row["timbre"].text() == "调节音色"
    assert row["dialogue"].text() == "改台词"
    window.close()


def test_reassign_segment_moves_file_and_meta(svc):
    from recast.core.characters import character_folder, load_meta

    _add_character(svc, "温念", confirmed=1)
    target = _add_character(svc, "陆寒铮", confirmed=1)
    clip = _make_voice_sample(svc, "温念")
    window = _main(svc)
    window.timer.stop()
    window._show_audio_panel()
    combo = window.diar_table.cellWidget(0, 3)
    combo.setCurrentIndex(combo.findData(target["id"]))
    dest = character_folder(svc.settings.output_root, "陆寒铮") / "voice" / clip.name
    assert dest.is_file()
    assert not clip.exists()
    src_meta = load_meta(character_folder(svc.settings.output_root, "温念")) or {}
    assert str(clip) not in (src_meta.get("voice") or {}).get("samples", [])
    dst_meta = load_meta(character_folder(svc.settings.output_root, "陆寒铮")) or {}
    assert str(dest) in (dst_meta.get("voice") or {}).get("samples", [])
    assert "改派给陆寒铮" in window.audio_status.text()
    window.close()


def test_audition_falls_back_to_sample(svc, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    ch = _add_character(svc, "温念", confirmed=1)
    clip = _make_voice_sample(svc, "温念")
    window = _main(svc)
    window.timer.stop()
    opened: list = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()))
    monkeypatch.setattr(svc, "audition_voice", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("离线")))
    window._apply_audio_result({"kind": "audition", "character": ch["id"], "error": "离线"})
    assert opened == [str(clip)]
    assert "已有的声音样本" in window.audio_status.text()
    window.close()


def test_audition_says_missing_when_nothing(svc, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    ch = _add_character(svc, "温念", confirmed=1)
    window = _main(svc)
    window.timer.stop()
    opened: list = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()))
    window._apply_audio_result({"kind": "audition", "character": ch["id"], "error": "离线"})
    assert opened == []
    assert "没有声音样本" in window.audio_status.text()
    window.close()


def test_edit_subtitles_opens_dialog(svc, tmp_path: Path, monkeypatch):
    import recast.desktop.subtitle_dialog as subtitle_dialog_mod

    drama = _add_drama(svc, "剧", entry_type="import")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    srt = workdir / "subtitles.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\n你好\n", encoding="utf-8")
    task = _add_task(svc, drama["id"], status="已完成", workdir=str(workdir))
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    created: list = []

    class FakeDialog:
        def __init__(self, srt_path, parent=None, shots=None):
            created.append((str(srt_path), shots))

        def exec(self):
            return 0

    monkeypatch.setattr(subtitle_dialog_mod, "SubtitleDialog", FakeDialog)
    window._edit_subtitles()
    assert len(created) == 1
    assert created[0][0] == str(srt)
    window.close()


# ----- 第 6 项：导入预检卡 -----

def test_precheck_blocks_on_probe_error(svc):
    drama = _add_drama(svc, "剧", entry_type="import")
    window = _main(svc)
    window.timer.stop()
    window._apply_precheck_result({
        "drama_id": drama["id"], "video": "/tmp/ep1.mp4",
        "error": "文件没有视频流：/tmp/ep1.mp4",
    })
    assert drama["id"] in window._precheck_blocked
    assert "预检没通过" in window.precheck_label.text()
    assert "没有视频流" in window.precheck_label.text()
    window.close()


def test_precheck_blocks_on_zero_duration(svc):
    drama = _add_drama(svc, "剧", entry_type="import")
    window = _main(svc)
    window.timer.stop()
    window._apply_precheck_result({
        "drama_id": drama["id"], "video": "/tmp/ep1.mp4",
        "duration": 0, "has_audio": True, "burned": False,
    })
    assert drama["id"] in window._precheck_blocked
    assert "时长" in window.precheck_label.text()
    window.close()


def test_precheck_passes_with_warnings(svc):
    drama = _add_drama(svc, "剧", entry_type="import")
    window = _main(svc)
    window.timer.stop()
    window._apply_precheck_result({
        "drama_id": drama["id"], "video": "/tmp/ep1.mp4",
        "duration": 90.0, "has_audio": False, "burned": True,
    })
    assert drama["id"] not in window._precheck_blocked
    label = window.precheck_label.text()
    assert "预检通过" in label
    assert "时长：90.0 秒" in label
    assert "音轨：无" in label
    assert "烧录字幕：有" in label
    assert "没有音轨" in label
    assert "烧录字幕" in label
    window.close()


def test_precheck_unknown_burned_not_blocking(svc):
    drama = _add_drama(svc, "剧", entry_type="import")
    window = _main(svc)
    window.timer.stop()
    window._apply_precheck_result({
        "drama_id": drama["id"], "video": "/tmp/ep1.mp4",
        "duration": 12.5, "has_audio": True, "burned": None,
    })
    assert drama["id"] not in window._precheck_blocked
    label = window.precheck_label.text()
    assert "烧录字幕：未知" in label
    assert "无法判断" in label
    window.close()


def test_precheck_no_video_blocks(svc):
    drama = _add_drama(svc, "剧", entry_type="import")
    window = _main(svc)
    window.timer.stop()
    window._run_precheck(drama["id"])
    assert drama["id"] in window._precheck_blocked
    assert "没有视频文件" in window.precheck_label.text()
    window.close()


def test_start_refuses_blocked_drama(svc, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    drama = _add_drama(svc, "剧", entry_type="import")
    _add_task(svc, drama["id"])
    window = _main(svc)
    window.timer.stop()
    _paint_main(window, svc)
    window.table.cellWidget(0, 0).setChecked(True)
    window._precheck_blocked[drama["id"]] = "文件没有视频流：/tmp/ep1.mp4"
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    shown: list = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2] if len(a) > 2 else ""))
    window._start()
    assert calls == []
    assert shown and "预检没通过" in shown[0]
    window.close()


# ----- 第 8 项：任务历史入口按钮 -----

def test_history_button_opens_dialog(svc, monkeypatch):
    import recast.desktop.task_history_dialog as history_mod

    window = _main(svc)
    window.timer.stop()
    assert window.history_btn.text() == "任务历史"
    created: list = []

    class FakeDialog:
        def __init__(self, service, parent=None):
            created.append((service, parent))

        def exec(self):
            return 0

    monkeypatch.setattr(history_mod, "TaskHistoryDialog", FakeDialog)
    window.history_btn.click()
    assert len(created) == 1
    assert created[0][0] is svc
    window.close()


# ----- 第 3 项：失败任务 + 重试 -----

def _add_failed_drama(svc, tmp_path: Path):
    drama = _add_drama(svc, "失败的剧", entry_type="import")
    svc.store.update("dramas", drama["id"], status="失败", error="整集失败")
    task = svc.store.insert(
        "tasks", drama_id=drama["id"], episode_name="第1集",
        video_path=str(tmp_path / "ep1.mp4"), status="失败", error="字幕炸了",
    )
    svc.store.insert(
        "runs", task_id=task["id"], drama_id=drama["id"],
        step="01 扫描与补资料检查 / 02 解析提取（切镜/字幕）",
        status="failed", error="ASR 炸了",
        started_at="2026-11-12 10:00:00", finished_at="2026-11-12 10:01:00",
    )
    svc.store.insert(
        "checkpoints", task_id=task["id"], step="scenes",
        input_hashes="[]", output_paths="[]",
    )
    return drama, task


def test_failed_task_shows_stage_and_reason(svc, tmp_path: Path):
    from PySide6.QtWidgets import QLabel

    drama, _task = _add_failed_drama(svc, tmp_path)
    window = _main(svc)
    _paint_main(window, svc)
    assert window.failed_empty_label.isHidden()
    assert len(window.failed_cards) == 1
    text = window.failed_cards[0].findChild(QLabel, "failedText").text()
    assert "失败的剧" in text
    assert "解析提取" in text
    assert "01" not in text.split("失败在：")[1].split("\n")[0], text
    assert "ASR 炸了" in text
    assert "已完成到：解析提取（切镜/字幕）" in text
    window.close()


def test_failed_task_retry_calls_start_with_applicable_stages(svc, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QPushButton

    drama, _task = _add_failed_drama(svc, tmp_path)
    window = _main(svc)
    _paint_main(window, svc)
    calls: list = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: calls.append((a, k)))
    retry = window.failed_cards[0].findChild(QPushButton, "failedRetry")
    retry.click()
    assert len(calls) == 1
    assert calls[0][0][0] == [drama["id"]]
    assert set(calls[0][0][1]) == {"scan", "extract", "library", "replace", "compose", "export"}
    window.close()


def test_no_failed_tasks_shows_empty_label(svc):
    _add_drama(svc, "好剧", entry_type="import")
    window = _main(svc)
    _paint_main(window, svc)
    assert not window.failed_empty_label.isHidden()
    assert window.failed_cards == []
    window.close()


# ----- 第 4 项：阶段清单随入口自适应 -----

def test_stage_list_adapts_to_import_entry(svc):
    _add_drama(svc, "剧", entry_type="import")
    window = _main(svc)
    _paint_main(window, svc)
    for sid in ("template", "generate", "dub"):
        btn = window.stage_buttons[sid]
        assert not btn.isEnabled(), sid
        assert "仅剧本生成" in btn.text(), btn.text()
        assert not btn.text().startswith(("✓", "▶", "○", "🔒")), btn.text()
    for sid in ("scan", "extract", "library", "replace", "compose", "export"):
        assert window.stage_buttons[sid].isEnabled(), sid
    assert set(window._selected_stages()) == {"scan", "extract", "library", "replace", "compose", "export"}
    window.close()


def test_stage_list_adapts_to_script_entry(svc):
    _add_drama(svc, "剧", entry_type="script")
    window = _main(svc)
    _paint_main(window, svc)
    for sid in ("scan", "extract", "library", "replace"):
        btn = window.stage_buttons[sid]
        assert not btn.isEnabled(), sid
        assert "仅导入再铸" in btn.text(), btn.text()
    for sid in ("template", "generate", "dub", "compose", "export"):
        assert window.stage_buttons[sid].isEnabled(), sid
    assert set(window._selected_stages()) == {"template", "generate", "dub", "compose", "export"}
    window.close()


def test_stage_list_unfiltered_without_selection(svc):
    window = _main(svc)
    _paint_main(window, svc)
    assert all(btn.isEnabled() for btn in window.stage_buttons.values())
    window.close()


# ----- 第 5 项：统一编号 -----

def test_stage_labels_have_no_numbers(svc):
    window = _main(svc)
    _paint_main(window, svc)
    for sid, btn in window.stage_buttons.items():
        text = btn.text()
        assert not text[:1].isdigit(), f"{sid} 还带着数字编号：{text}"
        assert " 0" not in text.split(" ", 1)[-1][:3], text
    window.close()


def test_library_steps_renamed_to_checklist(svc):
    from PySide6.QtWidgets import QGroupBox, QLabel

    window = _library(svc)
    titles = [g.title() for g in window.findChildren(QGroupBox)]
    assert "操作清单" in titles
    assert "人物定义管线" not in titles
    group = next(g for g in window.findChildren(QGroupBox) if g.title() == "操作清单")
    texts = [lab.text() for lab in group.findChildren(QLabel)]
    assert texts, "操作清单里没有步骤行"
    for text in texts:
        assert text[:1] not in "①②③④⑤⑥", f"操作清单还带着编号：{text}"
    window.close()
