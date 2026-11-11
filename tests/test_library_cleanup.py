"""Delete a library character and drop stale face samples without leaving the output tree."""

from pathlib import Path

import pytest

from recast.assets3d.avatar import bind_path_for, legacy_bind_path_for
from recast.core.constants import AUTH_LICENSED, AUTH_OWN_GEN
from recast.core.sample_paths import (
    SamplePathError,
    plan_sample_removal,
    remove_generated_sample_file,
    require_generated_sample_file,
)
from recast.core.service import RecastService, asset_flags


def _face(path: Path) -> Path:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), (180, 140, 120))
    draw = ImageDraw.Draw(img)
    draw.ellipse((16, 10, 48, 50), fill=(210, 170, 150))
    img.save(path)
    return path


def _service(tmp_path: Path) -> RecastService:
    svc = RecastService(tmp_path / "db.db")
    svc.update_settings(output_root=str(tmp_path / "outputs"))
    return svc


def _sample(output: Path, name: str, filename: str = "00_f1.jpg") -> Path:
    return _face(output / "workbench" / "job-a" / "face_samples" / name / filename)


def _unconfirmed(svc: RecastService, name: str) -> dict:
    return svc.store.insert(
        "characters",
        name=name,
        display_name=name,
        confirmed=0,
        authorized=0,
        cluster_key=f"cluster:{name}",
    )


def _add_face(svc: RecastService, character_id: str, path: Path | str) -> None:
    svc.store.insert(
        "character_assets",
        character_id=character_id,
        kind="face",
        path=str(path),
        complete=1,
    )


def test_delete_character_removes_rows_samples_and_keeps_reference(tmp_path: Path):
    svc = _service(tmp_path)
    output = Path(svc.settings.output_root)
    sample = _sample(output, "Person_3")
    extra = _sample(output, "Person_3", "01_f2.jpg")
    index = sample.parents[1] / "index.json"
    index.write_text("{}", encoding="utf-8")
    reference = _face(tmp_path / "Documents" / "ref-face.jpg")
    outside_samples = tmp_path / "Documents" / "face_samples" / "Person_3"
    outside_samples.mkdir(parents=True)
    costume = tmp_path / "wardrobe" / "coat.png"
    costume.parent.mkdir(parents=True)
    costume.write_bytes(b"png")
    voice = tmp_path / "audio" / "line.wav"
    voice.parent.mkdir(parents=True)
    voice.write_bytes(b"wav")
    video = output / "exports" / "ep1" / "final.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    job_dir = output / "workbench" / "job-a"
    (job_dir / "shots" / "ep1.mp4").parent.mkdir(parents=True)
    (job_dir / "shots" / "ep1.mp4").write_bytes(b"shot")

    drama = svc.store.insert(
        "dramas",
        origin_name="旧剧",
        root_path=str(tmp_path / "drama"),
        status="已完成",
        output_dir=str(output / "exports" / "ep1"),
    )
    task = svc.store.insert(
        "tasks",
        drama_id=drama["id"],
        episode_name="第01集",
        video_path=str(video),
        output_file=str(video),
        status="已完成",
    )

    ch = _unconfirmed(svc, "Person_3")
    _add_face(svc, ch["id"], sample)
    _add_face(svc, ch["id"], extra)
    _add_face(svc, ch["id"], reference)
    _add_face(svc, ch["id"], outside_samples / "00.jpg")
    svc.store.set_character_asset(ch["id"], "costume", str(costume), complete=True)
    svc.store.set_character_asset(ch["id"], "voice", str(voice), complete=True)

    result = svc.delete_character(ch["id"])
    assert svc.store.get_character(ch["id"]) is None
    assert svc.store.list_rows("character_assets", "character_id = ?", (ch["id"],)) == []
    assert not sample.exists() and not extra.exists()
    assert not sample.parent.exists()
    assert index.is_file()
    assert job_dir.is_dir()
    assert reference.is_file()
    assert outside_samples.is_dir()
    assert costume.is_file() and voice.is_file()
    assert video.is_file() and (job_dir / "shots" / "ep1.mp4").is_file()
    assert str(reference) in result["kept_files"] or reference.resolve() in {Path(p).resolve() for p in result["kept_files"]}
    assert svc.store.get("dramas", drama["id"])["status"] == "已完成"
    assert svc.store.get("tasks", task["id"])["output_file"] == str(video)
    assert ch["id"] not in {row["id"] for row in svc.characters()}


def test_delete_character_removes_l2_without_escaping(tmp_path: Path):
    svc = _service(tmp_path)
    output = Path(svc.settings.output_root)
    photo = _face(tmp_path / "face.jpg")
    sample = _sample(output, "温念")
    kept_photo = _face(tmp_path / "kept-face.jpg")
    other = svc.add_character_manual("路人", str(kept_photo))
    svc.confirm_character(other["id"], authorized=True, auth_type=AUTH_OWN_GEN)
    other_asset = svc.build_l2(other["id"], [str(kept_photo)])

    ch = svc.add_character_manual("温念", str(photo))
    svc.confirm_character(ch["id"], authorized=True, auth_type=AUTH_OWN_GEN)
    _add_face(svc, ch["id"], sample)
    asset = svc.build_l2(ch["id"], [str(photo)])
    legacy = legacy_bind_path_for(ch["id"], output)
    from recast.assets3d.avatar import build_avatar

    build_avatar(ch["id"], [photo], legacy, display_name="温念旧档")
    svc.store.insert(
        "character_assets",
        character_id=ch["id"],
        kind="3d",
        path=str(legacy / "asset.json"),
        complete=1,
    )
    sentinel = output / "l2" / "KEEP.txt"
    sentinel.write_text("keep", encoding="utf-8")

    result = svc.delete_character(ch["id"])
    assert svc.store.get_character(ch["id"]) is None
    assert not Path(asset["dir"]).exists()
    assert not legacy.exists()
    assert not sample.exists()
    assert photo.is_file()
    assert sentinel.is_file()
    assert (Path(other_asset["dir"]) / "asset.json").is_file()
    assert kept_photo.is_file()
    assert svc.store.get_character(other["id"]) is not None
    assert any(Path(path).resolve() == legacy.resolve() or Path(path).resolve() == Path(asset["dir"]).resolve() for path in result["removed_l2"])
    assert Path(asset["dir"]).resolve() in {Path(path).resolve() for path in result["removed_l2"]}
    assert legacy.resolve() in {Path(path).resolve() for path in result["removed_l2"]}


def test_delete_character_refuses_l2_paths_outside_the_tree(tmp_path: Path):
    svc = _service(tmp_path)
    output = Path(svc.settings.output_root)
    sample = _sample(output, "温念")
    ch = svc.add_character_manual("温念", str(sample))
    weight = tmp_path / "models" / "flame" / "generic_model.pkl"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weight")
    venv_file = tmp_path / ".venv" / "lib" / "stay.py"
    venv_file.parent.mkdir(parents=True)
    venv_file.write_text("x", encoding="utf-8")
    video = output / "exports" / "ep1" / "final.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    l2_root = output / "l2"
    l2_root.mkdir(parents=True)
    (l2_root / "marker.txt").write_text("root", encoding="utf-8")

    svc.store.set_character_asset(ch["id"], "3d", str(weight), complete=True)
    with pytest.raises(PermissionError, match="拒绝删除"):
        svc.delete_character(ch["id"])
    assert weight.is_file()
    assert sample.is_file()
    assert svc.store.get_character(ch["id"]) is not None

    svc.store.set_character_asset(ch["id"], "3d", str(venv_file), complete=True)
    with pytest.raises(PermissionError, match="拒绝删除"):
        svc.delete_character(ch["id"])
    assert venv_file.is_file()

    svc.store.set_character_asset(ch["id"], "3d", str(video), complete=True)
    with pytest.raises(PermissionError, match="拒绝删除"):
        svc.delete_character(ch["id"])
    assert video.is_file()

    svc.store.set_character_asset(ch["id"], "3d", str(l2_root), complete=True)
    with pytest.raises(PermissionError, match="拒绝删除"):
        svc.delete_character(ch["id"])
    assert (l2_root / "marker.txt").is_file()

    link_target = tmp_path / "models" / "linked"
    link_target.mkdir()
    secret = link_target / "secret.bin"
    secret.write_text("keep", encoding="utf-8")
    link = output / "l2" / "escape"
    link.symlink_to(link_target, target_is_directory=True)
    svc.store.set_character_asset(ch["id"], "3d", str(link / "asset.json"), complete=True)
    with pytest.raises(PermissionError, match="拒绝删除"):
        svc.delete_character(ch["id"])
    assert secret.is_file()
    assert link.is_symlink()
    assert sample.is_file()
    assert svc.store.get_character(ch["id"]) is not None


def test_shared_sample_and_nested_models_are_not_deleted(tmp_path: Path):
    svc = _service(tmp_path)
    output = Path(svc.settings.output_root)
    shared = _sample(output, "Person_9")
    first = _unconfirmed(svc, "Person_9")
    second = _unconfirmed(svc, "Person_9b")
    _add_face(svc, first["id"], shared)
    _add_face(svc, second["id"], shared)
    svc.delete_character(first["id"])
    assert shared.is_file()
    assert svc.store.get_character(second["id"]) is not None
    svc.delete_character(second["id"])
    assert not shared.exists()

    planted = _face(output / "models" / "face_samples" / "hidden.jpg")
    outside_video = output / "workbench" / "job-a" / "face_samples" / "clip.mp4"
    outside_video.parent.mkdir(parents=True, exist_ok=True)
    outside_video.write_bytes(b"video")
    plain = output / "workbench" / "job-a" / "frame.jpg"
    plain.write_bytes(b"jpg")
    venv_pic = _face(output / ".venv" / "face_samples" / "x.jpg")
    for unsafe in (planted, outside_video, plain, venv_pic, output):
        with pytest.raises(SamplePathError, match="拒绝删除"):
            require_generated_sample_file(unsafe, output)
        with pytest.raises(SamplePathError, match="拒绝删除"):
            remove_generated_sample_file(unsafe, output)
        assert Path(unsafe).exists()

    link = output / "workbench" / "job-a" / "face_samples" / "link.jpg"
    target = _face(tmp_path / "Documents" / "picked.jpg")
    link.symlink_to(target)
    with pytest.raises(SamplePathError, match="拒绝删除"):
        remove_generated_sample_file(link, output)
    assert target.is_file() and link.is_symlink()
    assert plan_sample_removal(link, output).action == "skip"


def test_clean_invalid_samples_removes_people_without_a_valid_face(tmp_path: Path):
    svc = _service(tmp_path)
    output = Path(svc.settings.output_root)
    good = _sample(output, "Person_2", "00_f1.jpg")
    good_owner = _unconfirmed(svc, "Person_2")
    _add_face(svc, good_owner["id"], good)
    _add_face(svc, good_owner["id"], output / "workbench" / "job-a" / "face_samples" / "Person_2" / "missing.jpg")

    stale_only = _unconfirmed(svc, "Person_1")
    missing = output / "workbench" / "old-job" / "face_samples" / "Person_1" / "00_f1.jpg"
    missing.parent.mkdir(parents=True)
    (missing.parent / "note.txt").write_text("not a sample image", encoding="utf-8")
    _add_face(svc, stale_only["id"], missing)
    empty_person = _unconfirmed(svc, "Person_empty")

    mixed = _unconfirmed(svc, "Person_mix")
    mixed_face = _sample(output, "Person_mix")
    _add_face(svc, mixed["id"], mixed_face)
    _add_face(svc, mixed["id"], output / "workbench" / "job-a" / "face_samples" / "Person_mix" / "gone.jpg")
    costume_keep = tmp_path / "coat.png"
    costume_keep.write_bytes(b"png")
    svc.store.set_character_asset(mixed["id"], "costume", str(costume_keep), complete=True)
    svc.store.insert(
        "character_assets",
        character_id=mixed["id"],
        kind="costume",
        path=str(tmp_path / "missing-coat.png"),
        complete=1,
    )

    confirmed = svc.add_character_manual("已确认的人", None)
    confirmed_missing = output / "workbench" / "old-job" / "face_samples" / "已确认的人" / "00.jpg"
    _add_face(svc, confirmed["id"], confirmed_missing)

    authorized = _unconfirmed(svc, "Person_20")
    svc.store.update("characters", authorized["id"], authorized=1, auth_type=AUTH_LICENSED, confirmed=0)
    _add_face(svc, authorized["id"], output / "workbench" / "old-job" / "face_samples" / "Person_20" / "00.jpg")
    _add_face(svc, authorized["id"], output / "workbench" / "old-job" / "face_samples" / "Person_20" / "01.jpg")

    live_l2_owner = svc.add_character_manual("仍有头像", str(good))
    svc.confirm_character(live_l2_owner["id"], authorized=True, auth_type=AUTH_OWN_GEN)
    live_asset = svc.build_l2(live_l2_owner["id"], [str(good)])
    _add_face(svc, live_l2_owner["id"], output / "workbench" / "old-job" / "face_samples" / "仍有头像" / "gone.jpg")

    gone_l2 = bind_path_for("missing-avatar", output)
    confirmed_l2 = svc.add_character_manual("头像目录没了", None)
    svc.store.set_character_asset(confirmed_l2["id"], "3d", str(gone_l2 / "asset.json"), complete=True)

    video = output / "exports" / "ep1" / "final.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    drama = svc.store.insert(
        "dramas",
        origin_name="旧剧",
        root_path=str(tmp_path / "drama"),
        status="待执行",
    )

    preview = svc.list_invalid_samples()
    remove_ids = {item["id"] for item in preview["remove_characters"]}
    keep_ids = {item["id"] for item in preview["keep_characters"]}
    assert stale_only["id"] in remove_ids
    assert empty_person["id"] in remove_ids
    assert confirmed["id"] in remove_ids
    assert authorized["id"] in remove_ids
    assert confirmed_l2["id"] in remove_ids
    assert good_owner["id"] not in remove_ids
    assert mixed["id"] not in remove_ids
    assert live_l2_owner["id"] not in remove_ids
    assert good_owner["id"] in keep_ids
    assert mixed["id"] in keep_ids
    assert live_l2_owner["id"] in keep_ids
    assert Path(live_asset["dir"]).resolve() not in {Path(item["path"]).resolve() for item in preview["stale_assets"] if item.get("path")}
    assert any(item["character_id"] == confirmed_l2["id"] and item["kind"] == "3d" for item in preview["stale_assets"])
    assert any(item["id"] == authorized["id"] and "Person_20" in item["label"] for item in preview["remove_characters"])

    result = svc.clean_invalid_samples()
    assert result["remove_count"] >= 5
    ids = {row["id"] for row in svc.characters()}
    assert stale_only["id"] not in ids
    assert empty_person["id"] not in ids
    assert confirmed["id"] not in ids
    assert authorized["id"] not in ids
    assert confirmed_l2["id"] not in ids
    assert good_owner["id"] in ids
    assert mixed["id"] in ids
    assert live_l2_owner["id"] in ids

    assert good.is_file() and mixed_face.is_file() and costume_keep.is_file()
    assert video.is_file()
    assert (Path(live_asset["dir"]) / "asset.json").is_file()
    assert (missing.parent / "note.txt").is_file()
    assert svc.store.get("dramas", drama["id"]) is not None

    kept_good = svc.store.get_character(good_owner["id"])
    assert any(row.get("kind") == "face" and Path(row["path"]).is_file() for row in kept_good["assets"])
    assert not any(row.get("path") and not Path(str(row["path"])).is_file() for row in kept_good["assets"] if row.get("kind") == "face")
    assert asset_flags(kept_good)["face"] == "已齐"

    kept_mix = svc.store.get_character(mixed["id"])
    assert asset_flags(kept_mix)["face"] == "已齐"
    assert asset_flags(kept_mix)["costume"] == "已齐"
    assert not any(row.get("path") == str(tmp_path / "missing-coat.png") for row in kept_mix["assets"])

    assert svc.store.get_character(confirmed["id"]) is None
    assert svc.store.get_character(authorized["id"]) is None

    saved_l2 = svc.store.get_character(live_l2_owner["id"])
    assert any(row.get("kind") == "3d" and Path(row["path"]).parent.is_dir() for row in saved_l2["assets"])
    assert asset_flags(saved_l2)["3d"] in {"已齐", "偏低"}
    assert asset_flags(saved_l2)["face"] == "已齐"
    assert not any(row.get("kind") == "face" and row.get("path") and not Path(str(row["path"])).is_file() for row in saved_l2["assets"])
    assert svc.list_invalid_samples()["stale_count"] == 0
    assert svc.list_invalid_samples()["remove_count"] == 0


def test_library_window_delete_button_follows_selection(tmp_path: Path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QPushButton

    from recast.desktop.library_window import LibraryWindow

    app = QApplication.instance() or QApplication([])
    assert app is not None
    svc = _service(tmp_path)
    empty = LibraryWindow(svc)
    labels = [button.text() for button in empty.findChildren(QPushButton)]
    assert "删除人物" in labels
    assert "清理失效样本" in labels
    assert "删除 3D头像资产" in labels
    assert "清理未使用的 3D头像资产" in labels
    assert empty.delete_char_btn.objectName() == "danger"
    assert not empty.delete_char_btn.isEnabled()
    empty.close()

    ch = _unconfirmed(svc, "Person_20")
    _add_face(svc, ch["id"], tmp_path / "missing.jpg")
    window = LibraryWindow(svc)
    assert window.delete_char_btn.isEnabled()
    assert svc.store.get_character(ch["id"]) is not None
    assert svc.store.list_rows("character_assets", "character_id = ?", (ch["id"],))
    window.list.setCurrentRow(-1)
    assert window._current_id is None
    assert not window.delete_char_btn.isEnabled()
    window.close()


def test_clean_invalid_samples_leaves_a_valid_l2_directory(tmp_path: Path):
    svc = _service(tmp_path)
    output = Path(svc.settings.output_root)
    photo = _face(tmp_path / "Documents" / "face.jpg")
    ch = svc.add_character_manual("临时", str(photo))
    svc.confirm_character(ch["id"], authorized=True, auth_type=AUTH_OWN_GEN)
    asset = svc.build_l2(ch["id"], [str(photo)])
    svc.store.set_character_asset(
        ch["id"],
        "face",
        str(output / "workbench" / "job" / "face_samples" / "临时" / "gone.jpg"),
        complete=True,
    )
    still = svc.store.get_character(ch["id"])
    assert still["authorized"] and still["confirmed"]
    assert ch["id"] in {item["id"] for item in svc.list_invalid_samples()["remove_characters"]}

    svc.clean_invalid_samples()
    assert svc.store.get_character(ch["id"]) is None
    assert (Path(asset["dir"]) / "asset.json").is_file()
    assert photo.is_file()
    unused = svc.list_unused_l2()
    assert Path(asset["dir"]).resolve() in {Path(item["path"]).resolve() for item in unused}


def test_cleanup_detail_caps_the_preview_list():
    from recast.desktop.library_window import _cleanup_detail

    plan = {
        "stale_count": 31,
        "remove_count": 2,
        "kind_counts": {"脸": 31},
        "stale_assets": [{"label": f"Person_{i}（id{i}）  脸  00.jpg"} for i in range(31)],
        "remove_characters": [{"label": "Person_1（a）"}, {"label": "Person_2（b）"}],
        "keep_characters": [{"label": "Person_20（c）"}],
    }
    text = _cleanup_detail(plan)
    assert "失效资料 31 条（脸 31）" in text
    assert "将移出人物库 2 人（没有仍有效的人脸文件）" in text
    assert "包括已确认和已授权的人物" in text
    assert "仍有人脸文件，会留在人物库" in text
    assert "Person_0（id0）" in text
    assert "Person_29（id29）" in text
    assert "Person_30（id30）" not in text
    assert "另有 1 项" in text
    assert "Person_20（c）" in text
    assert "清理吗？" in text
