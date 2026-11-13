"""Tests for the review queue service methods + correction + guide steps.

Uses a real Store (temp sqlite) and synthetic images. No real video, no
detection model.
"""

from pathlib import Path

import pytest
from PIL import Image

from recast.core.review import DEFAULT_REVIEW_MIN_SCORE
from recast.core.service import RecastService


def _checker(path: Path, size: int = 140) -> Path:
    img = Image.new("L", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = 255 if (x // 7 + y // 7) % 2 else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path)
    return path


def _solid(path: Path, size: int = 140) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (200, 30, 30)).save(path)
    return path


@pytest.fixture()
def svc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Same construction as the other service tests: temp db, then point
    # the output/works roots at temp dirs. Never touches real settings.
    from recast.core.service import RecastService

    out = tmp_path / "out"
    works = tmp_path / "works"
    out.mkdir()
    works.mkdir()
    service = RecastService(tmp_path / "test.db")
    service.update_settings(output_root=str(out), works_root=str(works))
    return service


def _add_character(svc, name: str, *, confirmed: int = 0, samples: list[Path] | None = None):
    ch = svc.store.insert(
        "characters", name=name, display_name=name, confirmed=confirmed,
        authorized=0, cluster_key=f"test:{name}",
    )
    for p in samples or []:
        svc.store.insert(
            "character_assets", character_id=ch["id"], kind="face",
            path=str(p), complete=1, notes="聚类样本",
        )
    return svc.store.get_character(ch["id"])


def test_guide_steps_empty_library(tmp_path: Path, svc):
    steps = svc.guide_state()
    assert [s["id"] for s in steps] == ["import", "extract", "review", "replace", "audio"]
    assert steps[0]["state"] == "active"  # nothing imported yet
    # No characters at all: nothing to review, so step 4 is merely "todo"
    # (replace itself will refuse with its own "missing faces" message).
    assert steps[3]["state"] == "todo"
    # Step 5 follows step 4 and is never locked.
    assert steps[4]["state"] == "todo"
    assert svc.replace_locked() == (False, "")


def test_guide_step3_done_when_queue_empty(tmp_path: Path, svc):
    good = _checker(tmp_path / "works" / "good.png")
    _add_character(svc, "温念", confirmed=1, samples=[good])
    svc.store.insert("dramas", origin_name="剧", root_path=str(tmp_path / "drama"), episode_count=1)
    steps = {s["id"]: s for s in svc.guide_state()}
    assert steps["import"]["state"] == "done"
    assert steps["extract"]["state"] == "done"
    assert steps["review"]["state"] == "done"
    assert steps["replace"]["state"] == "active"
    assert svc.replace_locked() == (False, "")


def test_guide_step3_locked_when_queue_nonempty(tmp_path: Path, svc):
    svc.store.insert("dramas", origin_name="剧", root_path=str(tmp_path / "drama"), episode_count=1)
    _add_character(svc, "人物1", confirmed=0)
    steps = {s["id"]: s for s in svc.guide_state()}
    assert steps["review"]["state"] == "active"
    assert steps["replace"]["state"] == "locked"
    locked, reason = svc.replace_locked()
    assert locked and "待审核 1 项" in reason


def test_review_assign_sample_moves_file(tmp_path: Path, svc):
    bad = _solid(tmp_path / "works" / "bad.png")
    src = _add_character(svc, "人物1", confirmed=1, samples=[bad])
    target = _add_character(svc, "温念", confirmed=1)
    assert len(svc.list_review_queue()) == 1
    out = svc.review_assign(str(bad), target["id"], owner_id=src["id"])
    assert out["kind"] == "sample"
    dest = Path(out["path"])
    assert dest.is_file() and dest.parent.name == "face"
    assert not bad.exists()
    assert svc.list_review_queue() == []


def test_review_assign_cluster_merges(tmp_path: Path, svc):
    good = _checker(tmp_path / "works" / "good.png")
    src = _add_character(svc, "人物1", confirmed=0, samples=[good])
    target = _add_character(svc, "温念", confirmed=1)
    out = svc.review_assign(src["id"], target["id"])
    assert out["kind"] == "cluster"
    assert svc.store.get_character(src["id"]) is None
    assert svc.list_review_queue() == []


def test_review_create_character_from_sample(tmp_path: Path, svc):
    bad = _solid(tmp_path / "works" / "bad.png")
    src = _add_character(svc, "人物1", confirmed=1, samples=[bad])
    out = svc.review_create_character(str(bad), "陈秋微", owner_id=src["id"])
    assert out["kind"] == "sample"
    assert out["character"]["display_name"] == "陈秋微"
    assert svc.list_review_queue() == []


def test_review_create_character_from_cluster(tmp_path: Path, svc):
    src = _add_character(svc, "人物1", confirmed=0)
    out = svc.review_create_character(src["id"], "温雪")
    assert out["character"]["display_name"] == "温雪"
    assert svc.list_review_queue() == []


def test_review_discard_sample_goes_to_trash_not_deleted(tmp_path: Path, svc):
    bad = _solid(tmp_path / "works" / "bad.png")
    src = _add_character(svc, "人物1", confirmed=1, samples=[bad])
    out = svc.review_discard(str(bad), owner_id=src["id"])
    assert out["kind"] == "sample"
    trash = Path(out["trash"])
    assert trash.is_file() and "review_trash" in trash.parts
    assert not bad.exists()
    assert svc.list_review_queue() == []


def test_review_discard_cluster_deletes_character_row(tmp_path: Path, svc):
    src = _add_character(svc, "人物1", confirmed=0)
    svc.review_discard(src["id"])
    assert svc.store.get_character(src["id"]) is None


def test_move_face_sample_refuses_outside_files(tmp_path: Path, svc):
    outside = _checker(tmp_path / "outside.png")
    src = _add_character(svc, "人物1", confirmed=1, samples=[outside])
    target = _add_character(svc, "温念", confirmed=1)
    with pytest.raises(ValueError, match="作品目录或输出目录"):
        svc.move_face_sample(src["id"], str(outside), target["id"])


def test_move_face_sample_refuses_symlink(tmp_path: Path, svc, monkeypatch: pytest.MonkeyPatch):
    real = _checker(tmp_path / "works" / "real.png")
    link = tmp_path / "works" / "link.png"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("创建符号链接需要管理员权限或开发者模式。")
    src = _add_character(svc, "人物1", confirmed=1, samples=[link])
    target = _add_character(svc, "温念", confirmed=1)
    with pytest.raises(ValueError, match="符号链接"):
        svc.move_face_sample(src["id"], str(link), target["id"])


def test_remove_face_sample_moves_to_trash(tmp_path: Path, svc):
    good = _checker(tmp_path / "works" / "good.png")
    src = _add_character(svc, "人物1", confirmed=1, samples=[good])
    out = svc.remove_face_sample(src["id"], str(good))
    assert Path(out["trash"]).is_file()
    assert not good.exists()


def test_split_character_moves_selected_samples(tmp_path: Path, svc):
    a = _checker(tmp_path / "works" / "a.png")
    b = _checker(tmp_path / "works" / "b.png")
    src = _add_character(svc, "人物1", confirmed=1, samples=[a, b])
    out = svc.split_character(src["id"], [str(b)], "温雪")
    assert out["character"]["display_name"] == "温雪"
    assert len(out["moved"]) == 1
    assert Path(out["moved"][0]).is_file()
    assert a.is_file() and not b.exists()


def test_removed_sample_returns_to_queue(tmp_path: Path, svc):
    """纠错移出的样本要回到待审核清单（review_trash/ 根目录进队列）。"""
    good = _checker(tmp_path / "works" / "good.png")
    src = _add_character(svc, "人物1", confirmed=1, samples=[good])
    assert svc.list_review_queue() == []
    out = svc.remove_face_sample(src["id"], str(good))
    assert Path(out["trash"]).parent.name == "review_trash"
    queue = svc.list_review_queue()
    assert len(queue) == 1
    assert queue[0]["kind"] == "sample"
    assert queue[0]["character_id"] is None
    assert "移出" in queue[0]["source"]
    # 锁跟着回来：第 ③ 步没完成，替换继续锁住。
    assert svc.replace_locked()[0] is True


def test_review_assign_adopts_trash_sample(tmp_path: Path, svc):
    good = _checker(tmp_path / "works" / "good.png")
    src = _add_character(svc, "人物1", confirmed=1, samples=[good])
    target = _add_character(svc, "温念", confirmed=1)
    svc.remove_face_sample(src["id"], str(good))
    trash_row = svc.list_review_queue()[0]
    out = svc.review_assign(trash_row["key"], target["id"])
    assert out["kind"] == "sample"
    assert Path(out["path"]).is_file()
    assert svc.list_review_queue() == []


def test_review_discard_trash_sample_goes_to_discarded(tmp_path: Path, svc):
    good = _checker(tmp_path / "works" / "good.png")
    src = _add_character(svc, "人物1", confirmed=1, samples=[good])
    svc.remove_face_sample(src["id"], str(good))
    trash_row = svc.list_review_queue()[0]
    out = svc.review_discard(trash_row["key"])
    assert out["kind"] == "sample"
    assert Path(out["trash"]).parent.name == "discarded"
    assert Path(out["trash"]).is_file()
    assert svc.list_review_queue() == []


def test_review_discard_sample_goes_to_discarded_not_queue(tmp_path: Path, svc):
    bad = _solid(tmp_path / "works" / "bad.png")
    src = _add_character(svc, "人物1", confirmed=1, samples=[bad])
    out = svc.review_discard(str(bad), owner_id=src["id"])
    assert Path(out["trash"]).parent.name == "discarded"
    assert svc.list_review_queue() == []
