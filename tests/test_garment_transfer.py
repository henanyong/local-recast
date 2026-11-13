"""Tests for costume transfer (2D route).

The two invariants that make a transfer look like clothing instead of a
sticker, and that a future refactor could silently break:

- colour comes from the reference costume;
- shading (low frequency) and folds (high frequency) come from the frame.

Everything outside the body mask must stay bit-identical: the whole reason
this route has less show-through than a 3D stand-in is that it touches
nothing else.
"""

import cv2
import numpy as np
import pytest

from recast.stages.garment_transfer import (
    NEUTRAL_MAIN_CHROMA,
    GarmentError,
    _auto_garment,
    skin_mask,
    costume_reference,
    split_body,
    split_by_colour,
    split_by_colour,
    transfer_character_costume,
    transfer_garment,
)

TOP = 40
X0, X1 = 30, 50


def _frame_with_gradient(h=120, w=80):
    """Grey background, a body column whose luminance ramps left to right."""
    frame = np.full((h, w, 3), 120, dtype=np.uint8)
    span = max(1, X1 - X0 - 1)
    for x in range(X0, X1):
        value = int(40 + (x - X0) / span * 160)
        frame[TOP:, x] = (value, value, value)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[TOP:, X0:X1] = 255
    return frame, mask


def _solid(h=40, w=40, bgr=(200, 40, 40)):
    return np.full((h, w, 3), bgr, dtype=np.uint8)


def _two_tone_body(h=120, w=160):
    """A garment with a print-like minority split, as in the real footage:
    the cardigan is the dominant hue mass (2/3), the minority print (1/3) is
    what must survive. A grey body would be neutral, and the neutral guard
    would (correctly) refuse to recolour it — so colour-behaviour tests need
    real chroma. Halves are wide enough to sit outside each other's
    print-suppression neighbourhood.
    """
    span = (X1 - X0) * 3  # 60 px: 40 px dominant hue + 20 px minority
    split = X0 + (span * 2) // 3
    frame = np.full((h, w, 3), 120, dtype=np.uint8)
    frame[TOP:, X0:split] = (40, 60, 190)
    frame[TOP:, split : X0 + span] = (60, 170, 90)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[TOP:, X0 : X0 + span] = 255
    return frame, mask


def _lab_l(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)


def test_colour_comes_from_the_reference():
    """语义已改（相对主色保护）：与区域主色同向的那一半全额换向参考色，
    偏色的另一半被降权但仍与主色半区可区分——这是"开衫换了、里面的花
    衬衫保住"这个真实需求在合成数据上的对应行为。"""
    frame, mask = _two_tone_body()
    out = transfer_garment(frame, mask, TOP, _solid(bgr=(200, 40, 40)), part="full")
    img = out["image"]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    before_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    mid = X0 + ((X1 - X0) * 3 * 2) // 3
    red_half = (slice(TOP, 120), slice(X0 + 8, mid - 8))  # interior, off the seam
    green_half = (slice(TOP, 120), slice(mid + 8, X0 + (X1 - X0) * 3 - 8))
    a_red = float(lab[red_half][:, :, 1].mean())
    a_green = float(lab[green_half][:, :, 1].mean())
    # The main-hue half lands clearly on the reference colour...
    assert a_red > 60.0, f"主色半区没有换到参考色（a={a_red:.1f}）"
    # ...and moved further than the off-hue half, which is ramped down.
    delta_red = a_red - float(before_lab[red_half][:, :, 1].mean())
    delta_green = a_green - float(before_lab[green_half][:, :, 1].mean())
    assert delta_red > delta_green, "偏色半区的移动幅度不该超过主色半区"
    assert a_green < a_red - 30.0, "偏色半区必须仍然与主色半区可区分"


def test_lightness_and_folds_are_untouched():
    """Recolour mode must not touch L: that channel carries every fold, the
    shadow under the chin and the direction of the light. If this drifts the
    garment starts looking painted on."""
    frame, mask = _frame_with_gradient()
    out = transfer_garment(frame, mask, TOP, _solid(bgr=(128, 128, 128)), part="full")
    region = mask > 0
    before = _lab_l(frame)[region]
    after = _lab_l(out["image"])[region]
    assert np.abs(after - before).max() <= 1.0


def test_chroma_moves_toward_the_reference_without_flattening_the_pattern():
    """The gradient stands in for a print: after recolouring its spread
    should survive, because a flat fill reads as a cut-out."""
    frame, mask = _two_tone_body()
    out = transfer_garment(frame, mask, TOP, _solid(bgr=(200, 40, 40)), part="full")
    region = mask > 0
    chroma = cv2.cvtColor(out["image"], cv2.COLOR_BGR2LAB)[:, :, 1:].astype(np.float32)[region]
    assert chroma.std(axis=0).max() > 1.0, "两色衣身的色彩差异必须保留下来"


def test_texture_mode_still_runs_and_stays_in_range():
    frame, mask = _frame_with_gradient()
    out = transfer_garment(
        frame, mask, TOP, _solid(bgr=(200, 40, 40)), part="full", mode="texture"
    )
    assert out["mode"] == "texture"
    assert out["image"].dtype == np.uint8
    assert out["image"].shape == frame.shape


def test_everything_outside_the_mask_is_untouched():
    frame, mask = _frame_with_gradient()
    out = transfer_garment(frame, mask, TOP, _solid(bgr=(200, 40, 40)), part="full")
    img = out["image"]
    assert (img[:, :26] == frame[:, :26]).all()  # feather bleeds ~3px by design
    assert (img[:, 54:] == frame[:, 54:]).all()
    assert (img[: TOP - 4] == frame[: TOP - 4]).all()


def test_transfer_reports_what_it_did():
    frame, mask = _frame_with_gradient()
    out = transfer_garment(frame, mask, TOP, _solid(), part="full")
    assert out["region_pixels"] == int((mask > 0).sum())
    assert out["part"] == "full"
    assert out["box"][0] == X0 and out["box"][2] == X1


def test_split_body_halves_are_disjoint_and_cover_the_body():
    _, mask = _frame_with_gradient()
    parts = split_body(mask, TOP)
    assert not (parts["upper"] & parts["lower"]).any()
    assert TOP < parts["split_y"] < parts["bottom"] + 1
    assert int(parts["upper"].sum() + parts["lower"].sum()) == int((mask > 0)[TOP:].sum())


def test_split_follows_an_explicit_waist_line():
    _, mask = _frame_with_gradient()
    out = transfer_garment(
        np.zeros((120, 80, 3), np.uint8), mask, TOP, _solid(), part="upper", split_y=90
    )
    ys = np.where((mask > 0)[TOP:])[0] + TOP
    assert out["region_pixels"] < int(ys.size)  # only the part above the waist


def test_unknown_part_is_rejected():
    frame, mask = _frame_with_gradient()
    with pytest.raises(GarmentError, match="未知部位"):
        transfer_garment(frame, mask, TOP, _solid(), part="gloves")


def test_empty_reference_is_rejected():
    frame, mask = _frame_with_gradient()
    with pytest.raises(GarmentError, match="空的"):
        transfer_garment(frame, mask, TOP, np.zeros((0, 0, 3), np.uint8), part="full")


def test_empty_body_region_is_rejected():
    frame = np.full((120, 80, 3), 120, np.uint8)
    with pytest.raises(GarmentError, match="没有人物像素"):
        transfer_garment(frame, np.zeros((120, 80), np.uint8), TOP, _solid(), part="full")


def test_costume_reference_ignores_non_images(tmp_path):
    folder = tmp_path / "char"
    (folder / "costume").mkdir(parents=True)
    (folder / "costume" / "notes.txt").write_text("x", encoding="utf-8")
    assert costume_reference(folder) is None
    (folder / "costume" / "01.jpg").write_bytes(b"not really a jpeg")
    assert costume_reference(folder).name == "01.jpg"


def test_character_costume_reports_a_missing_folder(tmp_path):
    folder = tmp_path / "char"
    folder.mkdir()
    frame, mask = _frame_with_gradient()
    with pytest.raises(GarmentError, match="没有服装参考图"):
        transfer_character_costume(frame, mask, TOP, folder)


def _write_png(path, bgr):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    buf.tofile(str(path))


def test_read_image_is_cached_and_reread_when_the_file_changes(tmp_path):
    """A clip run asks for the same reference once per frame; re-decoding it
    12000 times would cost more than the transfer itself."""
    import os
    import time

    from recast.stages.garment_transfer import read_image

    path = tmp_path / "ref.png"
    _write_png(path, np.full((16, 16, 3), 60, np.uint8))
    first = read_image(path)
    assert read_image(path) is first

    time.sleep(0.02)
    _write_png(path, np.full((16, 16, 3), 200, np.uint8))
    os.utime(path, None)
    second = read_image(path)
    assert second is not first
    assert second.mean() > first.mean()


# ----- neutral guard (white collars must not be dyed) -----


def _frame_with_collar(h=120, w=80):
    """A coloured garment block with a white band across it."""
    frame = np.full((h, w, 3), 120, dtype=np.uint8)
    frame[TOP:, X0:X1] = (40, 40, 200)  # saturated red garment
    frame[TOP + 8 : TOP + 20, X0:X1] = (245, 245, 245)  # white collar band
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[TOP:, X0:X1] = 255
    return frame, mask


def test_white_collar_survives_a_recolour():
    frame, mask = _frame_with_collar()
    collar = np.zeros_like(mask, dtype=bool)
    collar[TOP + 10 : TOP + 18, X0 + 2 : X1 - 2] = True

    before = _chroma_hue(frame, collar)
    out = transfer_garment(frame, mask, TOP, _solid(bgr=(200, 200, 40)), part="full")
    after = _chroma_hue(out["image"], collar)
    # The collar was near-neutral and must stay near-neutral.
    assert abs(before) < 4, "测试数据的领子本身就不中性"
    assert abs(after) < 12, f"白领子被染色了（色度 {after:.1f}）"
    assert out["reference_chroma"]["neutral_pixels_protected"] > 100


def test_without_the_guard_the_collar_does_get_dyed():
    """Guard the guard: if this passes only because recolour is weak, the
    previous test would be meaningless."""
    frame, mask = _frame_with_collar()
    collar = np.zeros_like(mask, dtype=bool)
    collar[TOP + 10 : TOP + 18, X0 + 2 : X1 - 2] = True

    guarded = transfer_garment(frame, mask, TOP, _solid(bgr=(200, 200, 40)), part="full")
    unguarded = transfer_garment(
        frame, mask, TOP, _solid(bgr=(200, 200, 40)), part="full", neutral_guard=False
    )
    assert abs(_chroma_hue(unguarded["image"], collar)) > abs(
        _chroma_hue(guarded["image"], collar)
    )
    assert unguarded["reference_chroma"]["neutral_pixels_protected"] == 0


def _chroma_hue(img, region):
    """Signed b-channel chroma: positive means yellow, negative means blue."""
    import cv2

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(lab[:, :, 2][region].mean() - 128.0)


# ----- colour-cluster split (offline stand-in for SCHP) -----


def test_split_by_colour_separates_two_garments():
    frame, mask = _two_tone_body()
    parts = split_by_colour(frame, mask > 0, clusters=3)
    assert len(parts) == 2
    assert sum(p["share"] for p in parts) == pytest.approx(1.0, abs=0.02)
    # Largest first, and the two parts must not overlap.
    assert parts[0]["share"] >= parts[1]["share"]
    assert not (parts[0]["mask"] & parts[1]["mask"]).any()
    for part in parts:
        assert 0 < part["pixels"] < int(mask.sum())


def test_colour_split_is_deterministic():
    frame, mask = _two_tone_body()
    first = [(p["order"], p["pixels"]) for p in split_by_colour(frame, mask > 0, clusters=3)]
    second = [(p["order"], p["pixels"]) for p in split_by_colour(frame, mask > 0, clusters=3)]
    assert first == second, "同一帧的分件结果必须可复现，否则预览无法对比"


def test_recolouring_one_cluster_leaves_the_other_garment_alone():
    """This is the whole point: change the jacket, keep the blouse."""
    frame, mask = _two_tone_body()
    region = mask > 0
    parts = split_by_colour(frame, mask > 0, clusters=3)
    keep = parts[1]["mask"] & region

    out = transfer_garment(
        frame, mask, TOP, _solid(bgr=(200, 40, 40)), part="full", cluster=0
    )
    before = _lab_l(frame)[keep]
    after = _lab_l(out["image"])[keep]
    # The untouched garment keeps its lightness exactly; only chroma of the
    # picked cluster moves.
    assert np.abs(after - before).max() <= 1.0
    assert out["part"].startswith("full:cluster")


def test_cluster_index_out_of_range_is_rejected():
    frame, mask = _two_tone_body()
    with pytest.raises(GarmentError, match="取不到第"):
        transfer_garment(frame, mask, TOP, _solid(), part="full", cluster=9)


def test_split_by_colour_rejects_an_empty_region():
    frame = np.full((60, 60, 3), 120, np.uint8)
    with pytest.raises(GarmentError, match="没有像素"):
        split_by_colour(frame, np.zeros((60, 60), bool))


# ----- colour-cluster split (offline stand-in for SCHP) -----


def test_split_by_colour_separates_two_garments():
    frame, mask = _two_tone_body()
    parts = split_by_colour(frame, mask > 0, clusters=3)
    assert len(parts) == 2
    assert sum(p["share"] for p in parts) == pytest.approx(1.0, abs=0.02)
    # Largest first, and the two parts must not overlap.
    assert parts[0]["share"] >= parts[1]["share"]
    assert not (parts[0]["mask"] & parts[1]["mask"]).any()
    for part in parts:
        assert 0 < part["pixels"] < int(mask.sum())


def test_colour_split_is_deterministic():
    frame, mask = _two_tone_body()
    first = [(p["order"], p["pixels"]) for p in split_by_colour(frame, mask > 0, clusters=3)]
    second = [(p["order"], p["pixels"]) for p in split_by_colour(frame, mask > 0, clusters=3)]
    assert first == second, "同一帧的分件结果必须可复现，否则预览无法对比"


def test_recolouring_one_cluster_leaves_the_other_garment_alone():
    """This is the whole point: change the jacket, keep the blouse."""
    frame, mask = _two_tone_body()
    region = mask > 0
    parts = split_by_colour(frame, mask > 0, clusters=3)
    keep = parts[1]["mask"] & region

    out = transfer_garment(
        frame, mask, TOP, _solid(bgr=(200, 40, 40)), part="full", cluster=0
    )
    before = _lab_l(frame)[keep]
    after = _lab_l(out["image"])[keep]
    # The untouched garment keeps its lightness exactly; only chroma of the
    # picked cluster moves.
    assert np.abs(after - before).max() <= 1.0
    assert out["part"].startswith("full:cluster")


def test_cluster_index_out_of_range_is_rejected():
    frame, mask = _two_tone_body()
    with pytest.raises(GarmentError, match="取不到第"):
        transfer_garment(frame, mask, TOP, _solid(), part="full", cluster=9)


def test_split_by_colour_rejects_an_empty_region():
    frame = np.full((60, 60, 3), 120, np.uint8)
    with pytest.raises(GarmentError, match="没有像素"):
        split_by_colour(frame, np.zeros((60, 60), bool))


# ----- skin guard and auto garment pick -----

SKIN_BGR = (165, 190, 230)  # sRGB 230/190/165: inside the YCrCb skin rule
BLUE_BGR = (200, 120, 60)  # saturated, outside the skin rule


def _frame_with_skin(h=160, w=120):
    """Grey plate, a blue garment block, and a skin-coloured square on it."""
    frame = np.full((h, w, 3), 120, dtype=np.uint8)
    frame[TOP:, X0:X1] = BLUE_BGR
    # Keep the patch inside X0..X1: a patch wider than the garment block would
    # sit on the plate, where the neutral guard (not the skin guard) covers it
    # and the test would pass for the wrong reason.
    frame[TOP + 10 : TOP + 40, X0 + 2 : X0 + 18] = SKIN_BGR
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[TOP:, X0:X1] = 255
    return frame, mask


def test_skin_is_not_dyed():
    frame, mask = _frame_with_skin()
    hand = np.zeros_like(mask, dtype=bool)
    hand[TOP + 14 : TOP + 36, X0 + 4 : X0 + 16] = True

    out = transfer_garment(frame, mask, TOP, _solid(bgr=(200, 200, 40)), part="full")
    # The patch must stay within a rounding error of where it was: a green
    # hand is read as broken instantly, even if the garment is perfect.
    assert np.abs(_lab_b(out["image"])[hand] - _lab_b(frame)[hand]).max() < 3.0
    assert out["reference_chroma"]["skin_pixels_protected"] > 300


def test_without_the_skin_guard_the_patch_is_dyed():
    """Guard the guard: with both guards off the skin patch must actually be
    recoloured, so `moved_guarded < moved_unguarded` means something. (The
    relative neutral guard alone already protects off-hue pixels, so the
    unguarded variant has to switch it off too.)"""
    frame, mask = _frame_with_skin()
    hand = np.zeros_like(mask, dtype=bool)
    hand[TOP + 14 : TOP + 36, X0 + 4 : X0 + 16] = True

    guarded = transfer_garment(frame, mask, TOP, _solid(bgr=(200, 200, 40)), part="full")
    unguarded = transfer_garment(
        frame,
        mask,
        TOP,
        _solid(bgr=(200, 200, 40)),
        part="full",
        skin_guard=False,
        neutral_guard=False,
    )
    assert unguarded["reference_chroma"]["skin_pixels_protected"] == 0
    moved_guarded = float(np.abs(_lab_b(guarded["image"])[hand] - _lab_b(frame)[hand]).mean())
    moved_unguarded = float(np.abs(_lab_b(unguarded["image"])[hand] - _lab_b(frame)[hand]).mean())
    assert moved_unguarded > 0.0, "关闭双保护后肤色块应该被染色"
    assert moved_unguarded > moved_guarded


def test_auto_garment_skips_the_neutral_plate_and_the_skin():
    frame, mask = _frame_with_skin()
    region = mask > 0
    picked = _auto_garment(frame, region)
    assert picked is not None
    # The blue block is the only chromatic non-skin cluster that is big enough.
    touched = picked["mask"] & region
    assert touched.any()
    skin = skin_mask(frame) & region
    assert float((touched & skin).sum()) / max(1, int(touched.sum())) < 0.1


def test_auto_garment_falls_back_when_everything_is_grey():
    frame = np.full((100, 100, 3), 130, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:, 20:80] = 255
    out = transfer_garment(frame, mask, 40, _solid(bgr=(200, 40, 40)), part="full", auto_garment=True)
    assert out["part"] == "full:auto-all"


def test_auto_garment_reports_which_cluster_it_used():
    frame, mask = _frame_with_skin()
    out = transfer_garment(frame, mask, TOP, _solid(bgr=(200, 200, 40)), part="full", auto_garment=True)
    assert out["part"].startswith("full:auto")
    assert out["auto_garment"] is True


# ----- rectangle gate -----

def test_a_rectangular_mask_is_flagged():
    """A person never fills their own bbox; when the mask does, the caller has
    to be told, because the recolour then follows a rectangle."""
    frame = np.full((120, 100, 3), 120, dtype=np.uint8)
    frame[40:, :] = BLUE_BGR
    rect = np.zeros((120, 100), dtype=np.uint8)
    rect[40:, :] = 255
    out = transfer_garment(frame, rect, 40, _solid(bgr=(200, 200, 40)), part="full")
    assert out["region_fill"] == pytest.approx(1.0)
    assert any("矩形" in w for w in out["warnings"])


def test_a_person_shaped_mask_is_not_flagged():
    frame = np.full((160, 120, 3), 120, dtype=np.uint8)
    frame[40:, :] = BLUE_BGR
    mask = np.zeros((160, 120), dtype=np.uint8)
    cv2.ellipse(mask, (60, 110), (34, 50), 0, 0, 360, 255, -1)  # torso-ish
    mask[:40] = 0
    out = transfer_garment(frame, mask, 40, _solid(bgr=(200, 200, 40)), part="full")
    assert out["region_fill"] < 0.95
    assert out["warnings"] == []


def _lab_b(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 2].astype(np.float32)


# ----- relative neutral guard (shadows must follow the garment's hue) -----


def _shaded_wine_body(h=120, w=80):
    """One wine garment: a lit band and a shadow band of the SAME hue.

    Real knitwear measures chroma 22.8 in the light and 5-11 in the folds;
    (40,34,60) stands in for the folds — same hue as the lit band, a fraction
    of its chroma. The old absolute-chroma guard ramped those shadows down,
    so the lit band changed colour while the folds stayed behind: mottling.
    """
    frame = np.full((h, w, 3), 120, dtype=np.uint8)
    frame[TOP : TOP + 40, X0:X1] = (60, 50, 170)  # lit wine
    frame[TOP + 40 :, X0:X1] = (40, 34, 60)  # shadow wine, same hue
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[TOP:, X0:X1] = 255
    return frame, mask


def _lab_a_mean(img, rows, cols):
    return float(
        cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[rows, cols, 1].mean()
    )


def test_relative_guard_keys_on_hue_not_magnitude():
    """The new semantics, pinned directly: a low-chroma pixel whose hue
    matches the garment's main colour is garment body (full weight); a
    near-neutral pixel or an off-hue colour is foreign (protected)."""
    from recast.stages.garment_transfer import _relative_guard

    a = np.array([[150.0, 134.0, 128.0, 130.0, 103.0]], np.float32)  # +22, +6, 0, +2, -25
    b = np.array([[138.0, 131.0, 128.0, 140.0, 118.0]], np.float32)  # +10, +3, 0, +12, -10
    guard, off = _relative_guard(a, b, 22.0, 10.0)
    assert float(guard[0, 0]) > 0.95, "亮部本体必须全额换色"
    assert float(guard[0, 1]) > 0.95, "暗部与主色同向，必须全额换色（斑驳根源）"
    assert float(guard[0, 2]) == 0.0, "真正的中性色（白领子）必须保护"
    assert float(guard[0, 3]) < 0.3, "偏色的印花/奶油底色要大幅降权"
    assert float(guard[0, 4]) == 0.0, "反向色相的像素必须保护"
    # Signed hue offset: aligned pixels ~0, warm foreign positive, cool negative.
    assert abs(float(off[0, 0])) < 10.0
    assert float(off[0, 3]) > 20.0, "奶油底比酒红主色暖得多"
    assert float(off[0, 4]) < -20.0, "蓝花比酒红主色冷得多"


def test_print_blob_in_a_light_base_is_exempt():
    """嵌在奶油底里的小块同色相红花属于衬衫印花，必须整块豁免；
    而主导色相的大块开衫（哪怕紧邻异色物）不受影响。"""
    frame = np.full((220, 340, 3), 120, dtype=np.uint8)
    frame[TOP:, 30:160] = (40, 30, 120)  # wine cardigan, ~23k px blob
    frame[TOP:, 160:260] = (205, 220, 235)  # cream blouse base
    frame[TOP + 20 : TOP + 50, 175:195] = (60, 50, 170)  # a red flower on it
    mask = np.zeros((220, 340), dtype=np.uint8)
    mask[TOP:, 30:260] = 255

    out = transfer_garment(frame, mask, TOP, _solid(bgr=(120, 180, 120)), part="full")
    lab_after = cv2.cvtColor(out["image"], cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_before = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    flower = (slice(TOP + 22, TOP + 48), slice(177, 193))
    cardigan = (slice(TOP + 2, 218), slice(34, 158))
    delta_flower = (
        lab_after[flower][:, :, 1] - lab_before[flower][:, :, 1]
    )
    delta_cardigan = (
        lab_after[cardigan][:, :, 1] - lab_before[cardigan][:, :, 1]
    )
    assert float(np.abs(delta_flower).mean()) < 3.0, (
        f"奶油底里的红花被染色了（|Δa|={np.abs(delta_flower).mean():.1f}）"
    )
    assert float(delta_cardigan.mean()) < -15.0, "开衫主体没有换到参考色"


def test_shadow_and_light_of_one_garment_recolour_together():
    frame, mask = _shaded_wine_body()
    lit = (slice(TOP + 24, TOP + 38), slice(X0 + 6, X1 - 6))
    shadow = (slice(TOP + 44, mask.shape[0] - 2), slice(X0 + 6, X1 - 6))
    ref = _solid(bgr=(120, 180, 120))  # a muted green costume

    before_lit = _lab_a_mean(frame, *lit)
    before_shadow = _lab_a_mean(frame, *shadow)
    out = transfer_garment(frame, mask, TOP, ref, part="full")
    img = out["image"]
    delta_lit = before_lit - _lab_a_mean(img, *lit)
    delta_shadow = before_shadow - _lab_a_mean(img, *shadow)

    assert delta_lit > 8.0, "亮部必须向参考色移动"
    assert delta_shadow > 0.4 * delta_lit, (
        f"暗部没有跟上亮部（暗部 Δa={delta_shadow:.1f}，亮部 Δa={delta_lit:.1f}）：斑驳还在"
    )


def test_skin_rule_does_not_eat_a_dark_red_garment():
    """skin_mask's YCrCb rule matches mid-dark reds: a wine cardigan pixel
    measured Y=83/Cr=162/Cb=126, inside the rule, so the recolour skipped
    garment pixels and the knit went patchy. The skip must only fire on LIT
    skin (Y > SKIN_SKIP_MIN_Y); a dark red "skin" pixel is garment shadow."""
    frame = np.full((120, 80, 3), 120, dtype=np.uint8)
    frame[TOP:, X0:X1] = (40, 30, 120)  # wine cardigan
    # A red the skin rule genuinely matches (verified in the assertion below).
    frame[TOP + 10 : TOP + 30, X0 + 4 : X1 - 4] = (80, 60, 130)
    mask = np.zeros((120, 80), dtype=np.uint8)
    mask[TOP:, X0:X1] = 255

    flagged = skin_mask(frame)[TOP + 12 : TOP + 28, X0 + 6 : X1 - 6]
    assert flagged.all(), "测试数据必须真的触发皮肤误检，否则此测试无效"

    strip = (slice(TOP + 12, TOP + 28), slice(X0 + 6, X1 - 6))
    ref = _solid(bgr=(120, 180, 120))
    before = _lab_a_mean(frame, *strip)
    out = transfer_garment(frame, mask, TOP, ref, part="full")
    after = _lab_a_mean(out["image"], *strip)

    assert before - after > 8.0, "被误判成皮肤的衣服像素仍然被跳过了"
    assert out["reference_chroma"]["skin_pixels_protected"] == 0


def test_relative_guard_reports_its_mode_and_main_chroma():
    frame, mask = _shaded_wine_body()
    out = transfer_garment(frame, mask, TOP, _solid(bgr=(120, 180, 120)), part="full")
    assert out["reference_chroma"]["guard_mode"] == "relative_hue"
    assert out["reference_chroma"]["main_chroma"] > NEUTRAL_MAIN_CHROMA


def test_neutral_region_falls_back_to_the_absolute_guard():
    """A grey body has no dominant hue; the guard must fall back cleanly and
    change nothing (allowing ±2 for the feather blend's float rounding)."""
    frame = np.full((120, 80, 3), 120, dtype=np.uint8)
    frame[TOP:, X0:X1] = (128, 128, 128)
    mask = np.zeros((120, 80), dtype=np.uint8)
    mask[TOP:, X0:X1] = 255
    out = transfer_garment(frame, mask, TOP, _solid(bgr=(200, 40, 40)), part="full")
    assert out["reference_chroma"]["guard_mode"] == "absolute_chroma"
    diff = np.abs(out["image"].astype(np.int16) - frame.astype(np.int16))
    assert diff.max() <= 2, f"中性区域不该被换色（最大差 {diff.max()}）"
