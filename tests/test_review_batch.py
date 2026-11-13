"""Tests for the review-queue batch / filter / group helpers (recast/core/review.py).

Pure functions on synthetic data only: a sharp checkerboard scores high,
a solid colour scores low. No UI, no real video, no detection model.
"""

from pathlib import Path

from PIL import Image

from recast.core.review import (
    ReviewItem,
    filter_review_items,
    group_review_items,
    high_confidence_items,
)


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


def _item(kind: str, name: str, *, score=None, thumb=None, key="k") -> ReviewItem:
    return ReviewItem(
        kind=kind, key=key, character_id=None, character_name=name,
        thumb=thumb, score=score, source="test",
    )


def test_filter_by_kind(tmp_path: Path):
    items = [
        _item("cluster", "人物1", key="c1"),
        _item("sample", "温念", score=0.2, key="s1"),
        _item("sample", "温念", score=0.4, key="s2"),
    ]
    assert [i.key for i in filter_review_items(items, kinds={"cluster"})] == ["c1"]
    assert [i.key for i in filter_review_items(items, kinds={"sample"})] == ["s1", "s2"]
    assert len(filter_review_items(items)) == 3


def test_filter_by_score_band_keeps_clusters(tmp_path: Path):
    # 聚类行没有分数：分数段筛不过滤它们，只筛样本行。
    items = [
        _item("cluster", "人物1", key="c1"),
        _item("sample", "温念", score=0.2, key="s1"),
        _item("sample", "温念", score=0.4, key="s2"),
        _item("sample", "温念", score=0.9, key="s3"),  # 移出的样本分数可以很高
    ]
    assert [i.key for i in filter_review_items(items, score_band="low")] == ["c1", "s1"]
    assert [i.key for i in filter_review_items(items, score_band="mid")] == ["c1", "s2"]
    assert [i.key for i in filter_review_items(items, score_band="all")] == ["c1", "s1", "s2", "s3"]


def test_filter_accepts_dicts_too(tmp_path: Path):
    items = [{"kind": "sample", "key": "s1", "score": 0.2, "character_name": "温念"}]
    assert filter_review_items(items, score_band="low") == items
    assert filter_review_items(items, score_band="mid") == []


def test_group_by_kind_and_character(tmp_path: Path):
    items = [
        _item("cluster", "人物1", key="c1"),
        _item("sample", "温念", score=0.2, key="s1"),
        _item("sample", "温念", score=0.4, key="s2"),
        _item("cluster", "人物2", key="c2"),
        _item("sample", "待审核（已移出）", score=0.9, key="s3"),
    ]
    groups = group_review_items(items)
    assert [(title, [i.key for i in g]) for _, title, g in groups] == [
        ("聚类 · 人物1", ["c1"]),
        ("样本 · 温念", ["s1", "s2"]),
        ("聚类 · 人物2", ["c2"]),
        ("样本 · 待审核（已移出）", ["s3"]),
    ]
    # group_key 是 (kind, name)，同名不同类型不混到一组。
    assert groups[0][0] == ("cluster", "人物1")
    assert groups[1][0] == ("sample", "温念")


def test_high_confidence_only_clear_clusters(tmp_path: Path):
    clear = _checker(tmp_path / "clear.png")
    blurry = _solid(tmp_path / "blurry.png")
    items = [
        _item("cluster", "人物1", thumb=str(clear), key="c1"),   # 有清楚的代表脸
        _item("cluster", "人物2", thumb=str(blurry), key="c2"),  # 代表脸模糊
        _item("cluster", "人物3", thumb=None, key="c3"),          # 没有代表脸
        _item("sample", "温念", score=0.9, thumb=str(clear), key="s1"),  # 样本行永不入选
    ]
    assert [i.key for i in high_confidence_items(items)] == ["c1"]


def test_high_confidence_respects_min_score(tmp_path: Path):
    clear = _checker(tmp_path / "clear.png")
    items = [_item("cluster", "人物1", thumb=str(clear), key="c1")]
    assert high_confidence_items(items, min_score=0.5) == items
    assert high_confidence_items(items, min_score=1.01) == []  # 清晰度满分 1.0
