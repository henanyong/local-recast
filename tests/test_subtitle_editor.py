"""Subtitle editor backend: parse / validate / edit ops / timeline check / detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recast.stages.subtitles import (
    check_subtitle_timeline,
    delete_segment,
    detect_burned_subtitles,
    edit_segment_text,
    edit_segment_times,
    insert_segment,
    merge_segments,
    parse_srt,
    save_segments,
    split_segment,
    validate_segments,
    write_srt,
)

SRT_TEXT = """1
00:00:01,000 --> 00:00:03,000
你好世界

2
00:00:04,000 --> 00:00:06,500
第二行字幕

3
00:00:08,000 --> 00:00:10,000
第三行
"""


def segs():
    return [
        {"start": 1.0, "end": 3.0, "text": "你好世界"},
        {"start": 4.0, "end": 6.5, "text": "第二行字幕"},
        {"start": 8.0, "end": 10.0, "text": "第三行"},
    ]


# --- parse / round-trip -------------------------------------------------------


def test_parse_round_trip(tmp_path):
    path = tmp_path / "a.srt"
    write_srt(segs(), path)
    parsed = parse_srt(path)
    assert [(s["start"], s["end"], s["text"]) for s in parsed] == [
        (1.0, 3.0, "你好世界"),
        (4.0, 6.5, "第二行字幕"),
        (8.0, 10.0, "第三行"),
    ]
    assert parse_srt(SRT_TEXT) == parsed


def test_parse_malformed():
    with pytest.raises(ValueError, match="没有时间戳行"):
        parse_srt("1\n只有文字没有时间\n")
    with pytest.raises(ValueError, match="时间戳不对"):
        parse_srt("1\n99:99:99,999 --> 00:00:02,000\n文字\n")
    with pytest.raises(FileNotFoundError, match="字幕文件不存在"):
        parse_srt("no-such-file.srt")


def test_parse_multi_line_text():
    parsed = parse_srt("1\n00:00:01,000 --> 00:00:03,000\n第一行\n第二行\n")
    assert parsed[0]["text"] == "第一行\n第二行"


# --- validation ---------------------------------------------------------------


def test_validate_ok():
    assert validate_segments(segs()) == []


def test_validate_problems():
    bad = [
        {"start": 1.0, "end": 3.0, "text": "  "},
        {"start": 2.5, "end": 2.0, "text": "结束早于开始"},
        {"start": -1.0, "end": 0.5, "text": "负时间"},
        {"start": 5.0, "end": 6.0, "text": "和第二行重叠"},
        {"start": 2.0, "end": 4.0, "text": "顺序也乱了"},
    ]
    errors = validate_segments(bad)
    assert any("第 1 行文字为空" in e for e in errors)
    assert any("第 2 行结束时间必须晚于开始时间" in e for e in errors)
    assert any("第 3 行开始时间不能为负" in e for e in errors)
    assert any("重叠" in e for e in errors)
    assert any("顺序不对" in e for e in errors)


def test_save_rejects_and_writes_nothing(tmp_path):
    srt = tmp_path / "out.srt"
    bad = [{"start": 3.0, "end": 1.0, "text": "坏时间"}]
    with pytest.raises(ValueError, match="字幕不合法"):
        save_segments(bad, srt)
    assert not srt.exists()


def test_save_writes_srt_and_json(tmp_path):
    srt = tmp_path / "out.srt"
    js = tmp_path / "subtitles.json"
    js.write_text(json.dumps({"language": "zh", "segments": []}, ensure_ascii=False), encoding="utf-8")
    save_segments(segs(), srt, js)
    assert "你好世界" in srt.read_text(encoding="utf-8")
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["language"] == "zh"
    assert len(payload["segments"]) == 3


# --- edit ops -----------------------------------------------------------------


def test_edit_text_and_times():
    out = edit_segment_text(segs(), 1, "改过的第二行")
    assert out[1]["text"] == "改过的第二行"
    assert segs()[1]["text"] == "第二行字幕"  # original untouched
    with pytest.raises(ValueError, match="文字不能为空"):
        edit_segment_text(segs(), 0, "   ")
    out = edit_segment_times(segs(), 0, 1.5, 2.5)
    assert (out[0]["start"], out[0]["end"]) == (1.5, 2.5)
    with pytest.raises(ValueError, match="结束时间必须晚于开始时间"):
        edit_segment_times(segs(), 0, 2.0, 2.0)
    with pytest.raises(IndexError, match="没有第 9 行"):
        edit_segment_text(segs(), 8, "x")


def test_split_mid_and_at_time():
    out = split_segment(segs(), 0)
    assert len(out) == 4
    assert out[0]["text"] == "你好" and out[1]["text"] == "世界"
    assert out[0]["start"] == 1.0 and out[1]["end"] == 3.0
    assert out[0]["end"] == out[1]["start"] == 2.0
    out = split_segment(segs(), 0, split_at=2.5)
    assert out[0]["text"] == "你好世" and out[1]["text"] == "界"
    assert out[0]["end"] == 2.5
    with pytest.raises(ValueError, match="只有一个字"):
        split_segment([{"start": 0.0, "end": 1.0, "text": "啊"}], 0)
    with pytest.raises(ValueError, match="拆分点必须在"):
        split_segment(segs(), 0, split_at=9.0)


def test_merge():
    out = merge_segments(segs(), 0)
    assert len(out) == 2
    assert out[0]["text"] == "你好世界第二行字幕"
    assert (out[0]["start"], out[0]["end"]) == (1.0, 6.5)
    with pytest.raises(IndexError, match="最后一行"):
        merge_segments(segs(), 2)


def test_delete_and_insert():
    out = delete_segment(segs(), 1)
    assert [s["text"] for s in out] == ["你好世界", "第三行"]
    with pytest.raises(IndexError, match="没有第 5 行"):
        delete_segment(segs(), 4)
    out = insert_segment(segs(), 1, 3.2, 3.8, "插进来的")
    assert [s["text"] for s in out] == ["你好世界", "插进来的", "第二行字幕", "第三行"]
    out = insert_segment(segs(), 3, 11.0, 12.0, "末尾")
    assert out[-1]["text"] == "末尾"
    with pytest.raises(ValueError, match="文字不能为空"):
        insert_segment(segs(), 0, 0.0, 1.0, "  ")
    with pytest.raises(ValueError, match="结束时间必须晚于开始时间"):
        insert_segment(segs(), 0, 1.0, 1.0, "x")


# --- timeline check -----------------------------------------------------------


def test_timeline_clean():
    assert check_subtitle_timeline(segs()) == []


def test_timeline_overlap_and_too_long():
    rows = [
        {"start": 1.0, "end": 3.0, "text": "一二三四五六七八九十"},  # 10 chars / 2s = 5/s
        {"start": 2.5, "end": 4.0, "text": "重叠行"},
    ]
    issues = check_subtitle_timeline(rows)
    kinds = {i["kind"] for i in issues}
    assert kinds == {"too_long", "overlap"}
    assert any("第 1 行语速过快" in i["reason"] for i in issues)
    assert any("第 1 行与第 2 行时间重叠" in i["reason"] for i in issues)


def test_timeline_shot_boundary():
    shots = [{"start_time": 0.0, "end_time": 5.0}, {"start_time": 5.0, "end_time": 10.0}]
    rows = [{"start": 4.0, "end": 6.0, "text": "横跨切镜"}]
    issues = check_subtitle_timeline(rows, shots)
    assert len(issues) == 1 and issues[0]["kind"] == "shot_boundary"
    assert "5.00 秒" in issues[0]["reason"]
    aligned = [{"start": 4.0, "end": 5.0, "text": "对齐了"}]
    assert check_subtitle_timeline(aligned, shots) == []


# --- burned-in detection ------------------------------------------------------


def _make_video(path: Path, *, seconds: float = 3.0) -> Path:
    import subprocess

    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=10:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return path


def test_detect_with_stub_ocr(tmp_path):
    video = _make_video(tmp_path / "v.mp4")
    calls = {"n": 0}

    def stub(band):
        calls["n"] += 1
        return "烧录字幕" if calls["n"] == 2 else None

    assert detect_burned_subtitles(video, ocr_fn=stub) is True
    assert detect_burned_subtitles(video, ocr_fn=lambda band: None) is False


def test_detect_no_ocr_available_returns_none(tmp_path):
    # RapidOCR is not installed on this VM: the real default path must
    # degrade to "unknown" (None) instead of crashing or guessing.
    video = _make_video(tmp_path / "v.mp4")
    assert detect_burned_subtitles(video) is None


def test_detect_missing_video_returns_none(tmp_path):
    assert detect_burned_subtitles(tmp_path / "nope.mp4", ocr_fn=lambda band: "x") is None


# --- dialog -------------------------------------------------------------------

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def _qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _open_dialog(_qapp, tmp_path, srt_text=SRT_TEXT, shots=None):
    from recast.desktop.subtitle_dialog import SubtitleDialog

    srt = tmp_path / "subtitles.srt"
    srt.write_text(srt_text, encoding="utf-8")
    dlg = SubtitleDialog(srt, shots=shots)
    return dlg, srt


def test_dialog_loads_and_saves_text_edit(_qapp, tmp_path):
    from PySide6.QtWidgets import QDialog

    dlg, srt = _open_dialog(_qapp, tmp_path)
    assert dlg.table.rowCount() == 3
    dlg.table.item(1, 2).setText("改过的第二行")
    dlg._save()
    assert dlg.result() == QDialog.DialogCode.Accepted
    assert "改过的第二行" in srt.read_text(encoding="utf-8")


def test_dialog_split_merge_delete_insert(_qapp, tmp_path):
    dlg, srt = _open_dialog(_qapp, tmp_path)
    dlg.table.setCurrentCell(0, 2)
    dlg.split_btn.click()
    assert dlg.table.rowCount() == 4
    assert dlg.table.item(0, 2).text() == "你好"
    dlg.table.setCurrentCell(0, 2)
    dlg.merge_btn.click()
    assert dlg.table.rowCount() == 3
    assert dlg.table.item(0, 2).text() == "你好世界"
    dlg.table.setCurrentCell(1, 2)
    dlg.delete_btn.click()
    assert dlg.table.rowCount() == 2
    dlg.insert_btn.click()
    assert dlg.table.rowCount() == 3
    assert dlg.table.item(2, 2).text() == "新字幕"


def test_dialog_illegal_save_rejected(_qapp, tmp_path):
    from PySide6.QtWidgets import QDialog

    dlg, srt = _open_dialog(_qapp, tmp_path)
    before = srt.read_text(encoding="utf-8")
    dlg.table.item(0, 1).setText("00:00:00,500")  # end before start
    dlg._save()
    assert dlg.result() != QDialog.DialogCode.Accepted
    assert "结束时间必须晚于开始时间" in dlg.status.text()
    assert srt.read_text(encoding="utf-8") == before


def test_dialog_highlights_timeline_issues(_qapp, tmp_path):
    from PySide6.QtGui import QColor

    fast = "1\n00:00:01,000 --> 00:00:02,000\n一二三四五六七八九十\n"
    dlg, _srt = _open_dialog(_qapp, tmp_path, srt_text=fast)
    assert dlg.table.item(0, 0).background().color() == QColor(122, 30, 30)
    assert "语速过快" in dlg.status.text()
    dlg.table.item(0, 2).setText("短句")
    assert dlg.table.item(0, 0).background().color() != QColor(122, 30, 30)
