"""Scanner order, nested episodes, frame rate, compact tracks, and stem preference."""

from pathlib import Path

import pytest

from recast.core.checker import _index_images
from recast.core.scanner import list_videos, natural_sort_key, scan_root
from recast.ffmpeg import FFmpegError, interpret_timing
from recast.stages.track import compact_track_payload
from recast.store.db import Store


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"v")
    return path


def test_natural_sort_puts_episode_2_before_10(tmp_path: Path):
    folder = tmp_path / "剧"
    for name in ("第10集.mp4", "第2集.mp4", "第1集.mp4"):
        _touch(folder / name)
    names = [p.name for p in list_videos(folder)]
    assert names == ["第1集.mp4", "第2集.mp4", "第10集.mp4"]
    assert natural_sort_key(Path("第2集.mp4")) < natural_sort_key(Path("第10集.mp4"))


def test_season_subfolders_are_one_drama(tmp_path: Path):
    show = tmp_path / "作品" / "开局"
    _touch(show / "第一季" / "第10集.mp4")
    _touch(show / "第一季" / "第2集.mp4")
    _touch(show / "第二季" / "第1集.mp4")
    _touch(show / "参考图" / "广告.mp4")
    _touch(show / ".git" / "hidden.mp4")
    hits = scan_root(tmp_path / "作品")
    assert [h.origin_name for h in hits] == ["开局"]
    assert [e.name for e in hits[0].episodes] == ["第2集", "第10集", "第1集"]


def test_missing_fps_is_an_error_and_vfr_is_flagged():
    with pytest.raises(FFmpegError):
        interpret_timing({"avg_frame_rate": "0/0", "r_frame_rate": "N/A"}, {})
    fps, count, is_vfr, duration = interpret_timing(
        {"avg_frame_rate": "24000/1001", "r_frame_rate": "30/1", "nb_frames": "100", "duration": "4"},
        {},
    )
    assert is_vfr is True
    assert fps == pytest.approx(24000 / 1001)
    assert count == 100
    fps2, count2, steady, dur2 = interpret_timing(
        {"avg_frame_rate": "25/1", "r_frame_rate": "25/1", "duration": "2"},
        {},
    )
    assert steady is False
    assert fps2 == 25
    assert count2 == 50
    assert dur2 == 2


def test_compact_tracks_omit_the_frame_log():
    payload = {
        "track_ids": [1],
        "tracks": [{"track_id": 1, "frames": [{"frame_idx": 0, "bbox": [0, 0, 1, 1]}]}],
        "frames": [{"frame_idx": 0, "detections": [{"track_id": 1}]}],
    }
    compact = compact_track_payload(payload)
    assert "frames" not in compact
    assert compact["frames_omitted"] is True
    assert compact["tracks"][0]["frames"][0]["bbox"] == [0, 0, 1, 1]


def test_png_wins_over_jpg_with_the_same_stem(tmp_path: Path):
    folder = tmp_path / "参考图"
    folder.mkdir()
    (folder / "5.jpg").write_bytes(b"jpg")
    (folder / "5.png").write_bytes(b"png")
    index = _index_images(folder)
    assert index["5"].suffix == ".png"
    assert index["5.png"].suffix == ".png"
    assert index["5.jpg"].suffix == ".jpg"


def test_sql_identifiers_must_be_known_columns(tmp_path: Path):
    store = Store(tmp_path / "t.db")
    with pytest.raises(ValueError):
        store.insert("dramas;drop", origin_name="x", root_path=str(tmp_path))
    drama = store.insert("dramas", origin_name="剧", root_path=str(tmp_path / "show"))
    with pytest.raises(ValueError):
        store.update("dramas", drama["id"], **{"status;drop": "x"})
    with pytest.raises(ValueError):
        store.list_rows("tasks", "drama_id = ?; DROP TABLE dramas", (drama["id"],))


def test_upload_command_does_not_use_the_shell(tmp_path: Path, monkeypatch):
    from recast.upload import run_upload

    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["shell"] = kwargs.get("shell")
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("recast.upload.subprocess.run", fake_run)
    result = run_upload(src, enabled=True, mode="command", command="echo {name}")
    assert result["ok"] is True
    assert isinstance(seen["argv"], list)
    assert seen["shell"] is False
    assert seen["argv"][-1] == "a.mp4"
