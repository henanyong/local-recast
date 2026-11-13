"""Jianying export: validate before building, self-check after writing (G9)."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from recast.compose.jianying import (
    JianyingDraftError,
    export_jianying_draft,
    validate_for_jianying,
    verify_jianying_draft,
)


def _write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _shots(tmp_path: Path, *, with_outputs: bool = True) -> tuple[list[dict], Path, Path]:
    shot = _write(tmp_path / "S01.mp4", b"s")
    mix = _write(tmp_path / "mix.wav", b"w")
    shots = [
        {
            "shot_code": "S01",
            "start_time": 0.0,
            "duration_sec": 4.0,
            "output_path": str(shot) if with_outputs else None,
            "dialogue": "你好",
            "audio_duration": 1.5,
            "audio_path": str(mix),
        },
        {
            "shot_code": "S02",
            "start_time": 4.0,
            "duration_sec": 4.0,
            "output_path": str(shot) if with_outputs else None,
            "dialogue": "再见",
            "audio_duration": 2.0,
            "audio_path": str(mix),
        },
    ]
    return shots, shot, mix


def _srt(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "subs.srt"
    p.write_text(text, encoding="utf-8")
    return p


GOOD_SRT = """1
00:00:00,000 --> 00:00:04,000
你好

2
00:00:04,000 --> 00:00:08,000
再见
"""


def _dialogue(tmp_path: Path, lines: list[dict]) -> Path:
    p = tmp_path / "dialogue.json"
    p.write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")
    return p


def _aligned_dialogue() -> list[dict]:
    return [
        {"shot_code": "S01", "speaker": "温念", "text": "你好", "start_time": 0.0, "end_time": 4.0},
        {"shot_code": "S02", "speaker": "路人甲", "text": "再见", "start_time": 4.0, "end_time": 8.0},
    ]


def _minimal_glb(path: Path) -> Path:
    payload = json.dumps({"asset": {"version": "2.0"}}).encode()
    payload += b" " * ((4 - len(payload) % 4) % 4)
    total = 12 + 8 + len(payload)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(payload), 0x4E4F534A)
        + payload
    )
    return path


# --- validate: missing / unreadable inputs ---------------------------------


def test_validate_missing_video_names_the_file(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    missing = tmp_path / "nope.mp4"
    report = validate_for_jianying(tmp_path / "draft", shots=shots, output_video=missing)
    assert not report.ok
    assert any("成片视频不存在" in e and "nope.mp4" in e for e in report.errors)


def test_validate_empty_media_file_errors(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    report = validate_for_jianying(tmp_path / "draft", shots=shots, output_video=empty)
    assert not report.ok
    assert any("空文件" in e for e in report.errors)


def test_export_refuses_missing_video_and_writes_nothing(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    dest = tmp_path / "draft"
    with pytest.raises(JianyingDraftError) as excinfo:
        export_jianying_draft(dest, shots=shots, output_video=tmp_path / "nope.mp4")
    assert any("成片视频不存在" in issue for issue in excinfo.value.issues)
    assert not (dest / "project.json").exists()


def test_validate_shot_without_output_is_warning_only(tmp_path: Path):
    shots, _, _ = _shots(tmp_path, with_outputs=False)
    report = validate_for_jianying(tmp_path / "draft", shots=shots)
    assert report.ok
    assert any("S01" in w and "不会出现在剪映草稿里" in w for w in report.warnings)
    assert any("S02" in w for w in report.warnings)


def test_validate_bad_shot_times(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    shots[1]["duration_sec"] = -1
    report = validate_for_jianying(tmp_path / "draft", shots=shots)
    assert not report.ok
    assert any("S02" in e and "没有有效时长" in e for e in report.errors)

    shots, _, _ = _shots(tmp_path)
    shots[1]["start_time"] = -2.0
    report = validate_for_jianying(tmp_path / "draft", shots=shots)
    assert not report.ok
    assert any("S02" in e and "时间轴顺序乱了" in e for e in report.errors)


def test_validate_empty_title(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    report = validate_for_jianying(tmp_path / "draft", shots=shots, title="   ")
    assert not report.ok
    assert any("标题是空的" in e for e in report.errors)


def test_validate_nothing_to_export(tmp_path: Path):
    report = validate_for_jianying(tmp_path / "draft", shots=[])
    assert not report.ok
    assert any("没有可导出的镜头" in e for e in report.errors)


def test_validate_output_dir_blocked_by_file(tmp_path: Path):
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"x")
    shots, _, _ = _shots(tmp_path)
    report = validate_for_jianying(blocker / "draft", shots=shots)
    assert not report.ok
    assert any("输出目录" in e for e in report.errors)


# --- validate: subtitles ----------------------------------------------------


def test_validate_srt_end_before_start_names_block(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    srt = _srt(
        tmp_path,
        "1\n00:00:04,000 --> 00:00:00,000\n你好\n",
    )
    report = validate_for_jianying(tmp_path / "draft", shots=shots, srt=srt)
    assert not report.ok
    assert any("字幕第 1 条" in e and "结束时间早于" in e for e in report.errors)


def test_validate_srt_empty_text_and_bad_timestamp(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    srt = _srt(tmp_path, "1\n00:00:00,000 --> 00:00:04,000\n\n")
    report = validate_for_jianying(tmp_path / "draft", shots=shots, srt=srt)
    assert not report.ok
    assert any("没有文字" in e for e in report.errors)

    srt = _srt(tmp_path, "1\n00:00:00 --> 00:00:04,000\n你好\n")
    report = validate_for_jianying(tmp_path / "draft", shots=shots, srt=srt)
    assert not report.ok
    assert any("时间戳读不懂" in e for e in report.errors)


def test_validate_srt_overlapping_times(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    srt = _srt(
        tmp_path,
        "1\n00:00:00,000 --> 00:00:05,000\n你好\n\n2\n00:00:04,000 --> 00:00:08,000\n再见\n",
    )
    report = validate_for_jianying(tmp_path / "draft", shots=shots, srt=srt)
    assert not report.ok
    assert any("时间重叠" in e for e in report.errors)


def test_validate_good_srt_passes(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    srt = _srt(tmp_path, GOOD_SRT)
    report = validate_for_jianying(tmp_path / "draft", shots=shots, srt=srt)
    assert report.ok


def test_validate_missing_srt_file(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    report = validate_for_jianying(tmp_path / "draft", shots=shots, srt=tmp_path / "no.srt")
    assert not report.ok
    assert any("字幕文件不存在" in e for e in report.errors)


# --- validate: dialogue alignment -------------------------------------------


def test_validate_dialogue_count_mismatch(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    srt = _srt(tmp_path, GOOD_SRT)
    dlg = _dialogue(tmp_path, _aligned_dialogue()[:1])
    report = validate_for_jianying(tmp_path / "draft", shots=shots, srt=srt, dialogue=dlg)
    assert not report.ok
    assert any("台词表有 1 句，字幕有 2 条" in e for e in report.errors)


def test_validate_dialogue_time_mismatch(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    srt = _srt(tmp_path, GOOD_SRT)
    lines = _aligned_dialogue()
    lines[1]["start_time"] = 20.0
    lines[1]["end_time"] = 24.0
    dlg = _dialogue(tmp_path, lines)
    report = validate_for_jianying(tmp_path / "draft", shots=shots, srt=srt, dialogue=dlg)
    assert not report.ok
    assert any("S02" in e and "时间对不上" in e for e in report.errors)


def test_validate_dialogue_aligned_passes(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    srt = _srt(tmp_path, GOOD_SRT)
    dlg = _dialogue(tmp_path, _aligned_dialogue())
    report = validate_for_jianying(tmp_path / "draft", shots=shots, srt=srt, dialogue=dlg)
    assert report.ok


def test_validate_dialogue_bad_json_and_empty_line(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    bad = tmp_path / "dialogue.json"
    bad.write_text("{not json", encoding="utf-8")
    report = validate_for_jianying(tmp_path / "draft", shots=shots, dialogue=bad)
    assert not report.ok
    assert any("台词表读不懂" in e for e in report.errors)

    lines = _aligned_dialogue()
    lines[0]["text"] = "   "
    dlg = _dialogue(tmp_path, lines)
    report = validate_for_jianying(tmp_path / "draft", shots=shots, dialogue=dlg)
    assert not report.ok
    assert any("S01" in e and "没有文字" in e for e in report.errors)


# --- validate: reference assets ---------------------------------------------


def test_validate_missing_reference_asset(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    report = validate_for_jianying(
        tmp_path / "draft", shots=shots, reference_assets=[tmp_path / "face.jpg"]
    )
    assert not report.ok
    assert any("替换参考资产不存在" in e for e in report.errors)


def test_validate_bad_and_good_glb(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    bad = _write(tmp_path / "bad.glb", b"not a glb at all")
    report = validate_for_jianying(tmp_path / "draft", shots=shots, reference_assets=[bad])
    assert not report.ok
    assert any("不是有效的 GLB" in e for e in report.errors)

    good = _minimal_glb(tmp_path / "good.glb")
    face = _write(tmp_path / "face.jpg", b"jpg")
    report = validate_for_jianying(
        tmp_path / "draft", shots=shots, reference_assets=[good, face]
    )
    assert report.ok


def test_validate_shot_reference_fields(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    shots[0]["body_model"] = str(tmp_path / "missing.glb")
    report = validate_for_jianying(tmp_path / "draft", shots=shots)
    assert not report.ok
    assert any("S01" in e and "身体模型" in e for e in report.errors)


# --- validate: duration consistency (real media) ----------------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="需要 ffmpeg 生成测试视频")
def test_validate_duration_mismatch_with_real_video(tmp_path: Path):
    video = tmp_path / "real.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=duration=2:size=320x240:rate=24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True, capture_output=True,
    )
    shots, _, _ = _shots(tmp_path)  # 8 秒分镜，对 2 秒成片
    report = validate_for_jianying(tmp_path / "draft", shots=shots, output_video=video)
    assert not report.ok
    assert any("相差" in e and "静默截断" in e for e in report.errors)

    short_shots = [dict(shots[0], duration_sec=2.0)]
    report = validate_for_jianying(tmp_path / "draft", shots=short_shots, output_video=video)
    assert report.ok


# --- verify: written draft ---------------------------------------------------


def _good_draft(tmp_path: Path) -> Path:
    shots, shot, mix = _shots(tmp_path)
    srt = _srt(tmp_path, GOOD_SRT)
    return export_jianying_draft(
        tmp_path / "draft", shots=shots, output_video=shot, audio=mix, srt=srt, title="t"
    )


def test_verify_good_draft_passes(tmp_path: Path):
    dest = _good_draft(tmp_path)
    check = verify_jianying_draft(dest)
    assert check.ok, check.messages


def test_verify_missing_media_file(tmp_path: Path):
    dest = _good_draft(tmp_path)
    media_file = dest / "media" / "shots" / "S01.mp4"
    assert media_file.is_file()
    media_file.unlink()
    check = verify_jianying_draft(dest)
    assert not check.ok
    assert any("不存在" in e and "S01.mp4" in e for e in check.errors)


def test_verify_corrupt_draft_content(tmp_path: Path):
    dest = _good_draft(tmp_path)
    (dest / "draft_content.json").write_text("{broken", encoding="utf-8")
    check = verify_jianying_draft(dest)
    assert not check.ok
    assert any("draft_content.json" in e and "读不懂" in e for e in check.errors)


def test_verify_empty_material_path(tmp_path: Path):
    dest = _good_draft(tmp_path)
    content = json.loads((dest / "draft_content.json").read_text(encoding="utf-8"))
    content["materials"]["videos"][0]["path"] = ""
    (dest / "draft_content.json").write_text(
        json.dumps(content, ensure_ascii=False), encoding="utf-8"
    )
    check = verify_jianying_draft(dest)
    assert not check.ok
    assert any("路径是空的" in e for e in check.errors)


def test_verify_dangling_material_id(tmp_path: Path):
    dest = _good_draft(tmp_path)
    content = json.loads((dest / "draft_content.json").read_text(encoding="utf-8"))
    content["tracks"][0]["segments"][0]["material_id"] = "does-not-exist"
    (dest / "draft_content.json").write_text(
        json.dumps(content, ensure_ascii=False), encoding="utf-8"
    )
    check = verify_jianying_draft(dest)
    assert not check.ok
    assert any("断链" in e and "does-not-exist" in e for e in check.errors)


def test_verify_zero_size_file_and_missing_dir(tmp_path: Path):
    dest = _good_draft(tmp_path)
    (dest / "README.txt").write_bytes(b"")
    check = verify_jianying_draft(dest)
    assert not check.ok
    assert any("README.txt" in e and "空的" in e for e in check.errors)

    check = verify_jianying_draft(tmp_path / "no-such-draft")
    assert not check.ok
    assert any("草稿目录不存在" in e for e in check.errors)


def test_verify_duration_mismatch_between_files(tmp_path: Path):
    dest = _good_draft(tmp_path)
    project = json.loads((dest / "project.json").read_text(encoding="utf-8"))
    project["duration_sec"] = 99.0
    (dest / "project.json").write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
    check = verify_jianying_draft(dest)
    assert not check.ok
    assert any("总时长对不上" in e for e in check.errors)


# --- export wiring ----------------------------------------------------------


def test_export_with_aligned_srt_and_dialogue(tmp_path: Path):
    shots, shot, mix = _shots(tmp_path)
    srt = _srt(tmp_path, GOOD_SRT)
    dlg = _dialogue(tmp_path, _aligned_dialogue())
    dest = export_jianying_draft(
        tmp_path / "draft",
        shots=shots, output_video=shot, audio=mix, srt=srt, dialogue=dlg, title="t",
    )
    assert (dest / "project.json").is_file()
    assert verify_jianying_draft(dest).ok


def test_export_opt_out_flags_respected(tmp_path: Path):
    shots, _, _ = _shots(tmp_path)
    dest = export_jianying_draft(
        tmp_path / "draft",
        shots=shots, output_video=tmp_path / "nope.mp4",
        validate=False, verify=False,
    )
    assert (dest / "project.json").is_file()
