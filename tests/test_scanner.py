from __future__ import annotations

from pathlib import Path

from recast.core.scanner import scan_root


def _touch_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-video")
    return path


def test_scan_single_video(tmp_path: Path) -> None:
    video = _touch_video(tmp_path / "第01集.mp4")
    hits = scan_root(video)
    assert len(hits) == 1
    assert hits[0].episode_count == 1
    assert hits[0].episodes[0].video_path.resolve() == video.resolve()
    assert hits[0].origin_name == "第01集"


def test_scan_root_with_subfolder_dramas(tmp_path: Path) -> None:
    a = tmp_path / "剧目A"
    b = tmp_path / "剧目B" / "视频"
    _touch_video(a / "ep1.mp4")
    _touch_video(a / "ep2.mkv")
    _touch_video(b / "c01.mov")
    (a / "参考图").mkdir()
    (a / "名单.yaml").write_text("5: 沈黛\n6: 男主\n", encoding="utf-8")
    hits = {h.origin_name: h for h in scan_root(tmp_path)}
    assert "剧目A" in hits
    assert "剧目B" in hits
    assert hits["剧目A"].episode_count == 2
    assert hits["剧目B"].episode_count == 1
    assert hits["剧目A"].roster_path is not None
    assert hits["剧目A"].ref_dir is not None
    assert [e.key for e in hits["剧目A"].roster.entries] == ["5", "6"]


def test_scan_skips_venv_and_outputs(tmp_path: Path) -> None:
    _touch_video(tmp_path / "keep" / "ok.mp4")
    _touch_video(tmp_path / ".venv" / "hidden.mp4")
    _touch_video(tmp_path / "outputs" / "old.mp4")
    hits = scan_root(tmp_path)
    names = {h.origin_name for h in hits}
    assert names == {"keep"}
