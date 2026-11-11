from __future__ import annotations

from pathlib import Path

from recast.core.checker import check_drama, check_replace_ready
from recast.core.scanner import scan_root
from recast.core.state_machine import format_missing


def test_format_missing_names_files() -> None:
    assert format_missing(["5.png", "6.png"]) == "缺少 5.png、6.png"
    assert format_missing([]) == ""
    assert "资料不全" not in format_missing(["参考图/"])


def test_missing_roster_faces_are_named(tmp_path: Path) -> None:
    drama = tmp_path / "短剧"
    (drama / "参考图").mkdir(parents=True)
    (drama / "第01集.mp4").write_bytes(b"x")
    (drama / "名单.yaml").write_text("5: 沈黛\n6: 男主\n", encoding="utf-8")
    hit = scan_root(drama)[0]
    result = check_drama(hit, require_faces=True)
    assert result.ok is False
    assert result.message == "缺少 5.png、6.png"
    assert result.missing_names == ["5.png", "6.png"]


def test_present_faces_pass(tmp_path: Path) -> None:
    drama = tmp_path / "短剧"
    refs = drama / "参考图"
    refs.mkdir(parents=True)
    (refs / "5.png").write_bytes(b"img")
    (refs / "6.jpg").write_bytes(b"img")  # 6.png implied, but 6.jpg is also accepted
    (drama / "第01集.mp4").write_bytes(b"x")
    (drama / "名单.yaml").write_text("5: 沈黛\n6: 男主\n", encoding="utf-8")
    hit = scan_root(drama)[0]
    result = check_drama(hit)
    assert result.ok is True
    assert result.message == ""
    assert len(result.found_faces) == 2


def test_explicit_face_filename(tmp_path: Path) -> None:
    drama = tmp_path / "短剧"
    refs = drama / "参考图"
    refs.mkdir(parents=True)
    (drama / "a.mp4").write_bytes(b"x")
    (drama / "名单.yaml").write_text(
        "characters:\n  - name: 女主\n    face: hero.png\n",
        encoding="utf-8",
    )
    hit = scan_root(drama)[0]
    result = check_drama(hit)
    assert result.message == "缺少 hero.png"
    (refs / "hero.png").write_bytes(b"img")
    hit = scan_root(drama)[0]
    assert check_drama(hit).ok is True


def test_extract_without_roster_does_not_require_faces(tmp_path: Path) -> None:
    drama = tmp_path / "短剧"
    drama.mkdir()
    (drama / "a.mp4").write_bytes(b"x")
    hit = scan_root(drama)[0]
    result = check_drama(hit, require_faces=False)
    assert result.ok is True


def test_replace_ready_requires_authorized_face(tmp_path: Path) -> None:
    face = tmp_path / "ok.png"
    face.write_bytes(b"img")
    bad = check_replace_ready(authorized_faces=[], declared_missing=["5.png"])
    assert bad.ok is False
    assert "5.png" in bad.message
    assert "已授权角色的参考脸" in bad.message
    good = check_replace_ready(authorized_faces=[str(face)])
    assert good.ok is True
