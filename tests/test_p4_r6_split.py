"""P4 R6 拆分：主窗口第 ⑤ 步面板 / 预检卡 / 失败卡片抽出的新模块。

每个新模块 ≥2 个测试：模块能独立 import、关键函数行为与抽取前
一致；主窗口的薄委托也要测到（旧测试继续覆盖界面行为，这里只
补委托一致性）。只用合成数据 + stub service 调用，不跑真视频、
不跑 TTS、不联网。
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


def _add_drama(svc, name: str = "剧", *, entry_type: str = "import") -> dict:
    return svc.store.insert(
        "dramas", origin_name=name, root_path=f"/tmp/{name}", episode_count=1,
        entry_type=entry_type,
    )


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


# ----- recast/core/failed_tasks.py -----


def test_failed_tasks_imports_standalone():
    from recast.core import failed_tasks

    assert failed_tasks.CHECKPOINT_STAGE["scenes"] == "extract"
    assert callable(failed_tasks.collect_failed_entries)
    assert callable(failed_tasks.last_checkpoint_stage)


def test_collect_failed_entries_matches_old_shape(svc, tmp_path: Path):
    from recast.core.failed_tasks import collect_failed_entries

    drama, _task = _add_failed_drama(svc, tmp_path)
    entries = collect_failed_entries(svc.store)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["drama_id"] == drama["id"]
    assert entry["name"] == "失败的剧"
    assert entry["stage"] == "扫描与补资料检查 / 解析提取（切镜/字幕）"
    assert "ASR 炸了" in entry["reason"]
    assert "已完成到：解析提取（切镜/字幕）" in entry["reason"]


def test_collect_failed_entries_empty_when_nothing_failed(svc):
    from recast.core.failed_tasks import collect_failed_entries

    _add_drama(svc, "好剧")
    assert collect_failed_entries(svc.store) == []


def test_last_checkpoint_stage_unknown_and_missing(svc):
    from recast.core.failed_tasks import last_checkpoint_stage

    drama = _add_drama(svc, "剧")
    task = _add_task(svc, drama["id"])
    assert last_checkpoint_stage(svc.store, task["id"]) is None
    svc.store.insert(
        "checkpoints", task_id=task["id"], step="bogus",
        input_hashes="[]", output_paths="[]",
    )
    assert last_checkpoint_stage(svc.store, task["id"]) is None


def test_window_failed_entries_delegates_to_core(svc, tmp_path: Path):
    from recast.core.failed_tasks import collect_failed_entries

    _add_failed_drama(svc, tmp_path)
    window = _main(svc)
    window.timer.stop()
    assert window._failed_entries() == collect_failed_entries(svc.store)
    assert window._last_checkpoint_stage("no-such-task") is None
    window.close()


# ----- recast/desktop/failed_cards.py -----


def test_failed_cards_imports_standalone():
    from recast.desktop import failed_cards

    assert callable(failed_cards.build_failed_card)
    assert callable(failed_cards.refresh_failed_cards)
    assert callable(failed_cards.build_failed_group)


def test_build_failed_card_text_and_retry():
    from PySide6.QtWidgets import QLabel, QPushButton

    from recast.desktop.failed_cards import build_failed_card

    _app()
    called: list = []
    card = build_failed_card(
        {"drama_id": "d1", "name": "失败的剧", "stage": "解析提取", "reason": "ASR 炸了"},
        on_retry=lambda did: called.append(did),
    )
    assert card.findChild(QLabel, "failedText").text() == "「失败的剧」失败在：解析提取\nASR 炸了"
    retry = card.findChild(QPushButton, "failedRetry")
    assert retry.text() == "从失败步续跑"
    assert retry.property("drama_id") == "d1"
    retry.click()
    assert called == ["d1"]


def test_refresh_failed_cards_skips_rebuild_when_unchanged():
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    from recast.desktop.failed_cards import refresh_failed_cards

    _app()
    host = QWidget()
    layout = QVBoxLayout(host)
    from PySide6.QtWidgets import QLabel

    empty = QLabel("没有失败的任务。")
    entries = [{"drama_id": "d1", "name": "剧", "stage": "解析提取", "reason": "炸了"}]
    sig, cards = refresh_failed_cards(
        layout=layout, empty_label=empty, entries=entries, prev_sig=None,
        on_retry=lambda did: None,
    )
    assert cards is not None and len(cards) == 1
    assert layout.count() == 1
    assert empty.isHidden()
    sig2, cards2 = refresh_failed_cards(
        layout=layout, empty_label=empty, entries=entries, prev_sig=sig,
        on_retry=lambda did: None,
    )
    assert cards2 is None
    assert sig2 == sig
    assert layout.count() == 1


def test_refresh_failed_cards_rebuilds_on_change_and_empty():
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    from recast.desktop.failed_cards import refresh_failed_cards

    _app()
    host = QWidget()
    layout = QVBoxLayout(host)
    empty = QLabel("没有失败的任务。")
    sig, cards = refresh_failed_cards(
        layout=layout, empty_label=empty, entries=[], prev_sig=None,
        on_retry=lambda did: None,
    )
    assert cards == []
    assert not empty.isHidden()
    entries = [{"drama_id": "d1", "name": "剧", "stage": "解析提取", "reason": "炸了"}]
    sig, cards = refresh_failed_cards(
        layout=layout, empty_label=empty, entries=entries, prev_sig=sig,
        on_retry=lambda did: None,
    )
    assert len(cards) == 1
    assert empty.isHidden()


# ----- recast/core/precheck.py -----


def test_precheck_imports_standalone():
    from recast.core import precheck

    assert callable(precheck.precheck_dot)
    assert callable(precheck.format_precheck_text)
    assert callable(precheck.summarize_precheck)
    assert callable(precheck.probe_precheck_payload)


def test_precheck_dot_variants():
    from recast.core.precheck import precheck_dot

    assert precheck_dot(None) == ("", "")
    assert precheck_dot({"blocked": None, "has_audio": True, "burned": False}) == ("✓ ", "预检通过")
    assert precheck_dot({"blocked": None, "has_audio": False, "burned": False}) == (
        "⚠ ", "预检警告：无音轨")
    assert precheck_dot({"blocked": None, "has_audio": True, "burned": None}) == (
        "⚠ ", "预检警告：烧录字幕未知")
    assert precheck_dot({"blocked": "视频时长读不到（0 秒）。", "video": "/tmp/e.mp4"}) == (
        "✗ ", "预检阻止：视频时长读不到（0 秒）。")


def test_format_precheck_text_warnings():
    from recast.core.precheck import format_precheck_text

    text = format_precheck_text(90.0, False, True)
    assert text.startswith("预检通过。\n时长：90.0 秒\n音轨：无\n烧录字幕：有")
    assert "没有音轨" in text
    assert "检测到烧录字幕" in text
    assert "未知" in format_precheck_text(12.5, True, None)
    assert "⚠" not in format_precheck_text(10.0, True, False)


def test_summarize_precheck_error_and_zero_duration():
    from recast.core.precheck import ZERO_DURATION_REASON, summarize_precheck

    blocked = summarize_precheck({
        "drama_id": "d1", "video": "/tmp/ep1.mp4",
        "error": "文件没有视频流：/tmp/ep1.mp4",
    })
    assert blocked["blocked"] == "文件没有视频流：/tmp/ep1.mp4"
    assert blocked["record"] == {"video": "/tmp/ep1.mp4", "blocked": blocked["blocked"]}
    assert blocked["label_text"].startswith("预检没通过：文件没有视频流")
    zero = summarize_precheck({
        "drama_id": "d1", "video": "/tmp/ep1.mp4",
        "duration": 0, "has_audio": True, "burned": False,
    })
    assert zero["blocked"] == ZERO_DURATION_REASON
    assert "视频时长读不到（0 秒）" in zero["label_text"]


def test_summarize_precheck_pass_record_shape():
    from recast.core.precheck import summarize_precheck

    passed = summarize_precheck({
        "drama_id": "d1", "video": "/tmp/ep1.mp4",
        "duration": 90.0, "has_audio": False, "burned": True,
    })
    assert passed["blocked"] is None
    assert passed["record"] == {
        "video": "/tmp/ep1.mp4", "blocked": None,
        "duration": 90.0, "has_audio": False, "burned": True,
    }
    assert "预检通过" in passed["label_text"]
    assert "时长：90.0 秒" in passed["label_text"]


def test_probe_precheck_payload_probe_error(monkeypatch):
    import recast.ffmpeg as ffmpeg_mod

    from recast.core.precheck import probe_precheck_payload

    monkeypatch.setattr(
        ffmpeg_mod, "probe_video",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("没有视频流")),
    )
    payload = probe_precheck_payload("/tmp/ep1.mp4", "d1", with_burned=True)
    assert payload["error"] == "没有视频流"
    assert "duration" not in payload


def test_probe_precheck_payload_success_without_burned(monkeypatch):
    import recast.ffmpeg as ffmpeg_mod

    from recast.core.precheck import probe_precheck_payload

    class _Info:
        duration = 12.5
        has_audio = True

    monkeypatch.setattr(ffmpeg_mod, "probe_video", lambda *a, **k: _Info())
    payload = probe_precheck_payload("/tmp/ep1.mp4", "d1", row_only=True, with_burned=False)
    assert payload == {
        "drama_id": "d1", "video": "/tmp/ep1.mp4", "row_only": True,
        "duration": 12.5, "has_audio": True,
    }


def test_window_apply_precheck_delegates_to_core(svc):
    drama = _add_drama(svc, "剧", entry_type="import")
    window = _main(svc)
    window.timer.stop()
    window._apply_precheck_result({
        "drama_id": drama["id"], "video": "/tmp/ep1.mp4",
        "duration": 90.0, "has_audio": False, "burned": True,
    })
    assert window._precheck[drama["id"]]["duration"] == 90.0
    assert drama["id"] not in window._precheck_blocked
    assert "预检通过" in window.precheck_label.text()
    assert window._precheck_dot(drama["id"])[0] == "⚠ "
    window.close()


# ----- recast/desktop/precheck_card.py -----


def test_precheck_card_imports_standalone():
    from recast.desktop import precheck_card

    assert callable(precheck_card.build_precheck_panel)
    assert callable(precheck_card.paint_precheck_dot)


def test_build_precheck_panel_hidden_with_default_text():
    from recast.desktop.precheck_card import build_precheck_panel

    _app()
    group, label = build_precheck_panel()
    assert group.isHidden()
    assert label.text() == "先选一部剧。"
    assert group.title() == "导入预检"


def test_paint_precheck_dot_updates_status_cell():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

    from recast.desktop.precheck_card import paint_precheck_dot

    _app()
    table = QTableWidget(1, 5)
    origin = QTableWidgetItem("剧")
    origin.setData(Qt.ItemDataRole.UserRole, "d1")
    table.setItem(0, 1, origin)
    table.setItem(0, 4, QTableWidgetItem("待执行"))
    paint_precheck_dot(table, "d1", "✗ ", "预检阻止：没有视频流")
    assert table.item(0, 4).text() == "✗ 待执行"
    assert table.item(0, 4).toolTip() == "预检阻止：没有视频流"
    paint_precheck_dot(table, "other", "✓ ", "预检通过")
    assert table.item(0, 4).text() == "✗ 待执行"


# ----- recast/core/audio_segments.py -----


def test_audio_segments_imports_standalone():
    from recast.core import audio_segments

    assert callable(audio_segments.collect_diar_segments)
    assert callable(audio_segments.character_voice_samples)
    assert callable(audio_segments.update_voice_samples)
    assert callable(audio_segments.reassign_segment)


def test_collect_diar_segments_sorted_and_filtered(svc):
    from recast.core.audio_segments import collect_diar_segments

    _add_character(svc, "温念", confirmed=1)
    _add_character(svc, "陆寒铮", confirmed=1)
    _make_voice_sample(svc, "温念", "SPK_1_5.00_7.00.wav")
    _make_voice_sample(svc, "陆寒铮", "SPK_2_0.00_2.50.wav")
    # 非分轨格式的文件不算分轨记录。
    _make_voice_sample(svc, "温念", "note.wav")
    segments = collect_diar_segments(svc)
    assert [s["start"] for s in segments] == [0.00, 5.00]
    assert segments[0]["speaker"] == "SPK_2"
    assert segments[0]["owner_name"] == "陆寒铮"
    assert segments[1]["speaker"] == "SPK_1"
    assert segments[1]["owner_name"] == "温念"
    assert collect_diar_segments(svc) == segments  # 幂等：两次结果一致


def test_character_voice_samples_and_update(svc, tmp_path: Path):
    from recast.core.audio_segments import character_voice_samples, update_voice_samples

    _add_character(svc, "温念", confirmed=1)
    clip = _make_voice_sample(svc, "温念")
    samples = character_voice_samples(svc, "温念")
    assert samples == [clip]
    extra = tmp_path / "extra.wav"
    extra.write_bytes(b"RIFF")
    update_voice_samples(svc, "温念", remove_path=None, add_path=extra)
    assert extra in character_voice_samples(svc, "温念")
    update_voice_samples(svc, "温念", remove_path=extra, add_path=None)
    assert extra not in character_voice_samples(svc, "温念")
    update_voice_samples(svc, None, remove_path=clip, add_path=None)  # 空 id 不炸


def test_reassign_segment_moves_file_and_meta(svc):
    from recast.core.audio_segments import collect_diar_segments, reassign_segment
    from recast.core.characters import character_folder, load_meta

    _add_character(svc, "温念", confirmed=1)
    target = _add_character(svc, "陆寒铮", confirmed=1)
    clip = _make_voice_sample(svc, "温念")
    seg = collect_diar_segments(svc)[0]
    message = reassign_segment(svc, seg, target["id"])
    dest = character_folder(svc.settings.output_root, "陆寒铮") / "voice" / clip.name
    assert dest.is_file()
    assert not clip.exists()
    assert seg["owner_id"] == target["id"]
    assert seg["owner_name"] == "陆寒铮"
    assert seg["path"] == str(dest)
    assert "改派给陆寒铮" in message
    src_meta = load_meta(character_folder(svc.settings.output_root, "温念")) or {}
    assert str(clip) not in (src_meta.get("voice") or {}).get("samples", [])
    dst_meta = load_meta(character_folder(svc.settings.output_root, "陆寒铮")) or {}
    assert str(dest) in (dst_meta.get("voice") or {}).get("samples", [])


def test_reassign_segment_missing_file_raises(svc):
    from recast.core.audio_segments import collect_diar_segments, reassign_segment

    _add_character(svc, "温念", confirmed=1)
    target = _add_character(svc, "陆寒铮", confirmed=1)
    clip = _make_voice_sample(svc, "温念")
    seg = collect_diar_segments(svc)[0]
    clip.unlink()
    with pytest.raises(FileNotFoundError):
        reassign_segment(svc, seg, target["id"])
    assert seg["owner_id"] != target["id"]


def test_window_diar_segments_delegates_to_core(svc):
    from recast.core.audio_segments import collect_diar_segments

    _add_character(svc, "温念", confirmed=1)
    _make_voice_sample(svc, "温念")
    window = _main(svc)
    window.timer.stop()
    assert window._diar_segments() == collect_diar_segments(svc)
    window.close()


# ----- recast/desktop/audio_panel.py -----


def test_audio_panel_imports_standalone():
    from recast.desktop import audio_panel

    assert callable(audio_panel.build_audio_panel)
    assert callable(audio_panel.refresh_diar_table)
    assert callable(audio_panel.rebuild_character_rows)
    assert callable(audio_panel.audio_status_text)
    assert callable(audio_panel.apply_audio_result)
    assert callable(audio_panel.find_subtitle_target)


def test_build_audio_panel_hidden_and_buttons():
    from recast.desktop.audio_panel import build_audio_panel

    _app()
    called: list = []
    widgets = build_audio_panel(
        on_rediarize=lambda: called.append("rediarize"),
        on_redub=lambda: called.append("redub"),
        on_edit_subtitles=lambda: called.append("subs"),
    )
    assert widgets.group.isHidden()
    assert widgets.status.text() == "先选一部剧，再点第 ⑤ 步。"
    assert widgets.rediar_btn.text() == "重新分轨"
    assert widgets.redub_btn.text() == "重配音"
    assert widgets.edit_subs_btn.text() == "编辑字幕"
    widgets.rediar_btn.click()
    widgets.redub_btn.click()
    widgets.edit_subs_btn.click()
    assert called == ["rediarize", "redub", "subs"]


def test_audio_status_text_both():
    from recast.desktop.audio_panel import audio_status_text

    assert audio_status_text([]).startswith("这部剧还没有分轨结果")
    assert audio_status_text([{"start": 0.0}]).startswith("分轨结果共 1 段")


def test_refresh_diar_table_and_character_rows():
    from recast.desktop.audio_panel import (
        build_audio_panel,
        rebuild_character_rows,
        refresh_diar_table,
    )

    _app()
    widgets = build_audio_panel(
        on_rediarize=lambda: None, on_redub=lambda: None, on_edit_subtitles=lambda: None,
    )
    segments = [
        {"start": 0.0, "end": 2.5, "speaker": "SPK_1", "owner_id": "c1", "path": "/tmp/a.wav"},
    ]
    characters = [
        {"id": "c1", "display_name": "温念", "name": "温念"},
        {"id": "c2", "display_name": "陆寒铮", "name": "陆寒铮"},
    ]
    reassigned: list = []
    refresh_diar_table(widgets.diar_table, segments, characters, reassigned.append)
    assert widgets.diar_table.rowCount() == 1
    assert widgets.diar_table.item(0, 0).text() == "0.00s"
    assert widgets.diar_table.item(0, 2).text() == "SPK_1"
    combo = widgets.diar_table.cellWidget(0, 3)
    assert combo.currentText() == "温念"
    combo.setCurrentIndex(combo.findData("c2"))
    assert reassigned == [0]
    clicked: list = []
    btns = rebuild_character_rows(
        widgets.char_layout, characters,
        on_audition=lambda cid: clicked.append(("audition", cid)),
        on_timbre=lambda cid: clicked.append(("timbre", cid)),
        on_dialogue=lambda cid: clicked.append(("dialogue", cid)),
    )
    assert set(btns) == {"c1", "c2"}
    assert btns["c1"]["audition"].text() == "试听 3 秒"
    assert btns["c1"]["audition"].objectName() == "audition:c1"
    assert btns["c2"]["timbre"].text() == "调节音色"
    assert btns["c2"]["dialogue"].text() == "改台词"
    btns["c2"]["timbre"].click()
    assert clicked == [("timbre", "c2")]


def test_find_subtitle_target(svc, tmp_path: Path):
    from recast.desktop.audio_panel import find_subtitle_target

    drama = _add_drama(svc, "剧", entry_type="import")
    srt, task_id, error = find_subtitle_target(svc, drama["id"])
    assert srt is None and task_id is None
    assert "还没有分集工作目录" in error
    workdir = tmp_path / "wd"
    workdir.mkdir()
    _add_task(svc, drama["id"], status="已完成", workdir=str(workdir))
    srt, task_id, error = find_subtitle_target(svc, drama["id"])
    assert srt is None and error is not None
    assert "还没有字幕文件" in error
    srt_file = workdir / "subtitles.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:02,000\n你好\n", encoding="utf-8")
    srt, task_id, error = find_subtitle_target(svc, drama["id"])
    assert error is None
    assert srt == srt_file
    assert task_id is not None


def test_apply_audio_result_redub_and_rediarize_via_window(svc):
    window = _main(svc)
    window.timer.stop()
    window._apply_audio_result({"kind": "redub", "report": {"redubbed": 3}})
    assert "重配了 3 行" in window.audio_status.text()
    window._apply_audio_result({"kind": "redub", "report": {}})
    assert window.audio_status.text() == "重配音完成。"
    window._apply_audio_result({"kind": "redub", "error": "炸了"})
    assert "重配音失败：炸了" in window.audio_status.text()
    # 分轨成功：先写“分轨完成：2 段。”，随后 _refresh_audio_panel 按磁盘
    # 真实分轨重建面板（这里没有真实分轨，状态栏被刷回“还没有分轨结果”——
    # 这就是抽取前的原行为）；只断言按钮恢复可用与失败分支文案。
    window._apply_audio_result({"kind": "rediarize", "report": {"segments": [1, 2]}})
    assert window.rediar_btn.isEnabled()
    assert "还没有分轨结果" in window.audio_status.text()
    window._apply_audio_result({"kind": "rediarize", "error": "炸了"})
    assert "分轨失败：炸了" in window.audio_status.text()
    window.close()


def test_apply_audio_result_audition_falls_back_to_sample(svc, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    from recast.desktop.audio_panel import apply_audio_result

    ch = _add_character(svc, "温念", confirmed=1)
    clip = _make_voice_sample(svc, "温念")
    window = _main(svc)
    window.timer.stop()
    opened: list = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()))
    apply_audio_result(
        window, {"kind": "audition", "character": ch["id"], "error": "离线"})
    assert opened == [str(clip)]
    assert "已有的声音样本" in window.audio_status.text()
    window.close()


def test_window_audio_panel_refresh_uses_new_modules(svc):
    _add_character(svc, "温念", confirmed=1)
    _make_voice_sample(svc, "温念")
    window = _main(svc)
    window.timer.stop()
    window._show_audio_panel()
    assert not window.audio_panel.isHidden()
    assert window.diar_table.rowCount() == 1
    assert window.diar_table.item(0, 0).text() == "0.00s"
    assert window.audio_status.text().startswith("分轨结果共 1 段")
    assert len(window._audio_char_btns) == 1
    window.close()
