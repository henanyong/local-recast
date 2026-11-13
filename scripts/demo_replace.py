"""Run the 2D replacement on a real frame and write a before/after sheet.

Two inputs:
- a frame (from a video at a timestamp, or a still),
- a costume reference (a full portrait is cropped to its garment region, so
  it can be pointed at the character pack's own 参考图 directly).

The mask comes from YOLO + GrabCut (offline, no download), and the neck cut
from the detected face when there is one. Everything the run could not do is
printed, not swallowed.

    python scripts/demo_replace.py --video "C:/.../第1集.mp4" --time 90 \
        --costume "C:/.../02-陈秋微.jpg" --out outputs/_demo

Pass ``--person 1`` to pick the second-largest person in frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_erase_scope import imread_unicode, imwrite_unicode  # noqa: E402
from recast.stages.costume import crop_costume  # noqa: E402
from recast.stages.garment_transfer import transfer_garment  # noqa: E402
from recast.stages.person_mask import (  # noqa: E402
    _load_yolo,
    estimate_neck_y,
    segment_person,
    segment_person_auto,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "_demo"


def grab_frame(source: Path, seconds: float) -> np.ndarray:
    if source.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi"}:
        image = imread_unicode(source)
        if image is None:
            raise SystemExit(f"读不懂图片：{source}")
        return image
    cap = cv2.VideoCapture(str(source))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise SystemExit(f"取不到 {seconds}s 的帧：{source}")
    return frame


def costume_crop(portrait: np.ndarray) -> np.ndarray:
    """Crop the garment band of a portrait, so the reference is clothing."""
    from recast.stages.character_sheet import _face_box

    box = _face_box(portrait)
    from PIL import Image

    crop = crop_costume(
        Image.fromarray(portrait[:, :, ::-1]),
        face_bbox=tuple(box) if box else None,
    )
    return np.asarray(crop)[:, :, ::-1].copy()


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (24, 24, 24), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 2, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--time", type=float, default=0.0)
    ap.add_argument("--costume", required=True)
    ap.add_argument("--person", type=int, default=0)
    ap.add_argument("--part", default="upper", choices=["upper", "lower", "full"])
    ap.add_argument("--mode", default="recolor", choices=["recolor", "texture"])
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "rmbg+yolo", "birefnet+yolo", "yolo-seg", "mediapipe-tasks", "grabcut"],
        help="分割后端。auto 会选本机最快可用的模型；grabcut 是不需要模型的兜底，慢约 50 倍",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = grab_frame(Path(args.video), args.time)
    print(f"帧 {frame.shape[1]}x{frame.shape[0]}  取景 {args.time}s")

    model = _load_yolo("yolo11n.pt", device="cuda")
    seg = segment_person_auto(
        frame, backend=args.backend, model=model, index=args.person, conf=0.3
    )
    print(
        f"分割：{seg['method']}，画面里 {seg['person_count']} 个人，"
        f"选中第 {args.person + 1} 个（面积最大的开始数）"
        + (f"；警告：{seg['warning']}" if seg["warning"] else "")
    )

    neck = estimate_neck_y(frame)
    print(f"脖子切口 y={neck['neck_y']}（来源：{neck['source']}）" + (f"；{neck['note']}" if neck["note"] else ""))

    portrait = imread_unicode(Path(args.costume))
    if portrait is None:
        raise SystemExit(f"读不懂服装参考图：{args.costume}")
    reference = costume_crop(portrait)
    ref_path = out_dir / "costume_reference.png"
    imwrite_unicode(ref_path, reference)

    transfer = transfer_garment(
        frame,
        seg["mask"],
        neck["neck_y"],
        reference,
        part=args.part,
        mode=args.mode,
    )
    image = transfer["image"]

    from recast.quality.replace_audit import audit_composite

    audit = audit_composite(frame, image, seg["mask"], neck["neck_y"])
    print(
        f"换装（{transfer['mode']}）：{transfer['region_pixels']} 像素  框 {transfer['box']}\n"
        f"  区域：{transfer['part']}  填充率 {transfer['region_fill']:.0%}"
        f"（人物轮廓一般 40–70%，接近 100% 说明分割漏到了背景）\n"
        f"  参考服装色 a={transfer['reference_chroma'].get('reference_a')} "
        f"b={transfer['reference_chroma'].get('reference_b')}  "
        f"保住皮肤 {transfer['reference_chroma'].get('skin_pixels_protected', 0)} px / "
        f"保住中性色 {transfer['reference_chroma'].get('neutral_pixels_protected', 0)} px\n"
        f"审计：背景损伤 {audit['background_damage_frac']:.3%}  接缝 {audit['seam_ratio']:.2f}  "
        f"-> {audit['verdict']}"
    )
    for warning in transfer.get("warnings") or []:
        print(f"  ⚠ {warning}")
    if args.part != "full":
        print(f"  提示：--part {args.part} 只换这一段，下方保持原样是设计如此；整片换用 --part full")
    if args.part != "full":
        print(f"  提示：--part {args.part} 只换这一段，下方保持原样是设计如此；整片换用 --part full")

    strip = np.hstack(
        [
            label(frame, "original"),
            label(image, f"costume transfer ({args.part})"),
        ]
    )
    imwrite_unicode(out_dir / "replace_ab.png", strip)

    # Zoom on the garment area so folds and edges can be judged.
    ys, xs = np.where(seg["mask"] > 0)
    x0 = max(0, int(xs.min()) - 20)
    y0 = max(0, neck["neck_y"] - 20)
    cw = min(image.shape[1] - x0, 520)
    ch = min(image.shape[0] - y0, 520)
    zoom = np.hstack(
        [
            label(frame[y0 : y0 + ch, x0 : x0 + cw], "original"),
            label(image[y0 : y0 + ch, x0 : x0 + cw], "transferred"),
        ]
    )
    imwrite_unicode(out_dir / "replace_zoom.png", zoom)
    print(f"输出：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
