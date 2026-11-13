"""B1: review queue by GPID + pending stitch pairs (recast/core/review.py).

Memory fake store + tmp folders only: a sharp checkerboard scores high,
a solid colour scores low. No real video, no detection model.
"""

from pathlib import Path

from PIL import Image

from recast.core.review import (
    ReviewItem,
    build_review_queue,
    layer_review_items,
)


def _checker(path: Path, size: int = 140) -> Path:
    """High-frequency image: sharp by any measure."""
    img = Image.new("L", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = 255 if (x // 7 + y // 7) % 2 else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path)
    return path


def _solid(path: Path, size: int = 140, color=(200, 30, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)
    return path


class _Store:
    """Minimal stand-in for the real Store: only list_characters."""

    def __init__(self, characters):
        self._characters = characters

    def list_characters(self):
        return self._characters


def _pair(kind, a, b, *, face_best=None, reason="理由"):
    return {
        "kind": kind,
        "a_track_id": a,
        "b_track_id": b,
        "votes": 1,
        "face_best": face_best,
        "body_sim": 0.8,
        "costume_sim": 0.7,
        "reason": reason,
        "a_frame": None,
        "b_frame": None,
    }


def _item(kind, name, *, key="k", score=None, gpid=None):
    return ReviewItem(
        kind=kind, key=key, character_id=None, character_name=name,
        thumb=None, score=score, source="test", gpid=gpid,
    )


# ----- pending_pairs 产出 stitch 行 -----

def test_pending_near_miss_creates_stitch_row(tmp_path: Path):
    out = tmp_path / "out"
    pair = _pair("near_miss", 3, 5, face_best=0.62, reason="仅一票：脸最优 0.62")
    items = build_review_queue(_Store([]), out, pending_pairs=[pair])
    assert len(items) == 1
    row = items[0]
    assert row.kind == "stitch"
    assert row.key == "stitch:3:5"
    assert row.character_id is None
    assert row.character_name == "待裁决拼接"
    assert row.thumb is None
    assert row.score == 0.62
    assert row.source == "仅一票：脸最优 0.62"
    assert row.gpid is None


def test_pending_faceless_attach_creates_stitch_row(tmp_path: Path):
    out = tmp_path / "out"
    pair = _pair(
        "faceless_attach", 7, 9, face_best=None,
        reason="无脸轨迹靠身体+服装两票挂到该人物，置信度低，需人工确认。",
    )
    items = build_review_queue(_Store([]), out, pending_pairs=[pair])
    assert len(items) == 1
    row = items[0]
    assert row.kind == "stitch"
    assert row.key == "stitch:7:9"
    assert row.score is None  # faceless_attach 没有 face_best
    assert row.source == "无脸轨迹靠身体+服装两票挂到该人物，置信度低，需人工确认。"
    assert row.character_name == "待裁决拼接"


def test_pending_faceless_attach_without_rep_still_keys(tmp_path: Path):
    # b_track_id 为 None（无脸轨迹挂到的组里没有有脸代表）：仍成行，
    # key 按公式 stitch:{a}:{b} 生成，score 为 None。
    out = tmp_path / "out"
    pair = _pair("faceless_attach", 7, None, reason="无代表")
    items = build_review_queue(_Store([]), out, pending_pairs=[pair])
    assert [i.key for i in items] == ["stitch:7:None"]
    assert items[0].score is None


def test_pending_pairs_none_has_no_stitch_rows(tmp_path: Path):
    out = tmp_path / "out"
    ch = {"id": "c1", "name": "人物1", "display_name": "人物1", "confirmed": 0, "assets": []}
    items = build_review_queue(_Store([ch]), out, pending_pairs=None)
    assert [i.kind for i in items] == ["cluster"]


def test_pending_pairs_empty_list_behaves_like_none(tmp_path: Path):
    out = tmp_path / "out"
    items = build_review_queue(_Store([]), out, pending_pairs=[])
    assert items == []
    # 默认参数（不传 pending_pairs）同样无 stitch 行。
    assert build_review_queue(_Store([]), out) == []


# ----- 队列顺序：聚类行 → stitch 行 → 样本行 -----

def test_queue_order_cluster_stitch_sample(tmp_path: Path):
    out = tmp_path / "out"
    cluster_ch = {
        "id": "c1", "name": "人物1", "display_name": "人物1", "confirmed": 0,
        "cluster_key": "ep1:3",
        "assets": [{"kind": "face", "path": str(_checker(tmp_path / "good.png"))}],
    }
    bad = _solid(tmp_path / "bad.png", 140)
    sample_ch = {
        "id": "c2", "name": "温念", "display_name": "温念", "confirmed": 1,
        "assets": [{"kind": "face", "path": str(bad)}],
    }
    pair = _pair("near_miss", 3, 5, face_best=0.62, reason="仅一票")
    items = build_review_queue(_Store([cluster_ch, sample_ch]), out, pending_pairs=[pair])
    assert [i.kind for i in items] == ["cluster", "stitch", "sample"]
    assert items[0].key == "c1"
    assert items[1].key == "stitch:3:5"
    assert items[2].key == str(bad)


def test_queue_order_multiple_stitch_pairs_keep_input_order(tmp_path: Path):
    out = tmp_path / "out"
    pairs = [
        _pair("near_miss", 1, 2, face_best=0.5, reason="第一对"),
        _pair("near_miss", 8, 11, face_best=0.7, reason="第二对"),
        _pair("faceless_attach", 4, 6, reason="第三对"),
    ]
    items = build_review_queue(_Store([]), out, pending_pairs=pairs)
    assert [i.key for i in items] == ["stitch:1:2", "stitch:8:11", "stitch:4:6"]


# ----- layer_review_items 分层 -----

def test_layer_mixed_queue(tmp_path: Path):
    items = [
        _item("sample", "温念", key="s1", score=0.2),
        _item("cluster", "人物1", key="c1", gpid="3"),
        _item("stitch", "待裁决拼接", key="stitch:3:5", score=0.62),
        _item("sample", "温念", key="s2", score=0.4),
        _item("cluster", "人物2", key="c2", gpid="1"),
    ]
    layers = layer_review_items(items)
    assert [name for name, _, _ in layers] == ["decisions", "samples"]
    assert layers[0][1] == "待确认（人物与拼接）"
    assert layers[1][1] == "低分样本"
    # decisions 层：stitch 在前，再按 gpid 排序的聚类行。
    assert [i.key for i in layers[0][2]] == ["stitch:3:5", "c2", "c1"]
    # samples 层：只含样本行，原顺序。
    assert [i.key for i in layers[1][2]] == ["s1", "s2"]


def test_layer_empty_returns_empty():
    assert layer_review_items([]) == []


def test_layer_only_samples_returns_samples_layer_only():
    items = [_item("sample", "温念", key="s1", score=0.2)]
    layers = layer_review_items(items)
    assert len(layers) == 1
    assert layers[0][0] == "samples"
    assert [i.key for i in layers[0][2]] == ["s1"]


def test_layer_only_decisions_returns_decisions_layer_only():
    items = [_item("cluster", "人物1", key="c1", gpid="3")]
    layers = layer_review_items(items)
    assert len(layers) == 1
    assert layers[0][0] == "decisions"
    assert [i.key for i in layers[0][2]] == ["c1"]


def test_layer_decisions_sorted_by_gpid_numerically():
    # gpid 是字符串："10" 应排在 "3" 之后（数值序，不是字典序）；
    # 无 gpid 的聚类行排在同类末尾；stitch 行始终在最前。
    items = [
        _item("cluster", "人物十", key="c10", gpid="10"),
        _item("cluster", "无编号", key="cX"),
        _item("cluster", "人物三", key="c3", gpid="3"),
        _item("stitch", "待裁决拼接", key="stitch:1:2"),
    ]
    layers = layer_review_items(items)
    assert [i.key for i in layers[0][2]] == ["stitch:1:2", "c3", "c10", "cX"]


# ----- gpid 推导 -----

def test_gpid_from_cluster_key_numeric(tmp_path: Path):
    out = tmp_path / "out"
    ch = {
        "id": "c1", "name": "人物1", "display_name": "人物1", "confirmed": 0,
        "cluster_key": "ep1:3", "assets": [],
    }
    items = build_review_queue(_Store([ch]), out)
    assert len(items) == 1
    assert items[0].kind == "cluster"
    assert items[0].gpid == "3"


def test_gpid_missing_cluster_key_is_none(tmp_path: Path):
    out = tmp_path / "out"
    # 根本没有 cluster_key 字段的行。
    ch1 = {"id": "c1", "name": "人物1", "display_name": "人物1", "confirmed": 0, "assets": []}
    # cluster_key 显式为 None 的行。
    ch2 = {
        "id": "c2", "name": "人物2", "display_name": "人物2", "confirmed": 0,
        "cluster_key": None, "assets": [],
    }
    items = build_review_queue(_Store([ch1, ch2]), out)
    assert [i.gpid for i in items] == [None, None]


def test_gpid_non_numeric_suffix_is_none(tmp_path: Path):
    out = tmp_path / "out"
    # cluster_id 缺失时 key 是 "{drama}:{auto_name}"，后缀不是数字。
    ch1 = {
        "id": "c1", "name": "人物1", "display_name": "人物1", "confirmed": 0,
        "cluster_key": "ep1:Person_1", "assets": [],
    }
    # 手动建的人物 cluster_key 是 "manual:{name}"。
    ch2 = {
        "id": "c2", "name": "温念", "display_name": "温念", "confirmed": 0,
        "cluster_key": "manual:温念", "assets": [],
    }
    # 没有冒号的 key。
    ch3 = {
        "id": "c3", "name": "人物3", "display_name": "人物3", "confirmed": 0,
        "cluster_key": "global", "assets": [],
    }
    items = build_review_queue(_Store([ch1, ch2, ch3]), out)
    assert [i.gpid for i in items] == [None, None, None]


def test_review_item_default_gpid_is_none():
    # 位置构造（旧调用方式）不受新字段影响，gpid 默认 None。
    item = ReviewItem(
        kind="cluster", key="c1", character_id="c1",
        character_name="人物1", thumb=None, score=None,
    )
    assert item.gpid is None


# ----- 回归：B0 include_rejected 开关行为不变 -----

def test_include_rejected_default_hides_rejected(tmp_path: Path):
    out = tmp_path / "out"
    bad = _solid(tmp_path / "bad.png", 140)
    ch = {
        "id": "c1", "name": "温念", "display_name": "温念", "confirmed": 1,
        "assets": [{"kind": "face", "path": str(bad), "notes": "待审核：太模糊"}],
    }
    # 被挡样本默认不出队列。
    assert build_review_queue(_Store([ch]), out) == []
    assert build_review_queue(_Store([ch]), out, include_rejected=False) == []


def test_include_rejected_true_shows_rejected(tmp_path: Path):
    out = tmp_path / "out"
    bad = _solid(tmp_path / "bad.png", 140)
    ch = {
        "id": "c1", "name": "温念", "display_name": "温念", "confirmed": 1,
        "assets": [{"kind": "face", "path": str(bad), "notes": "待审核：太模糊"}],
    }
    items = build_review_queue(_Store([ch]), out, include_rejected=True)
    assert len(items) == 1
    assert items[0].kind == "sample"
    assert items[0].key == str(bad)
