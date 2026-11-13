"""Run the 2D replacement over a real clip and report what it cost.

This is the end-to-end check the UI buttons cannot give: a whole segment
through segmentation -> tracking -> recolour -> audit -> encode, with the
per-frame verdicts summarised so a human knows which frames to look at.

    python scripts/demo_replace_clip.py --video "C:/.../第1集.mp4" \
        --start 88 --frames 60 --costume "examples/.../02-陈秋微.jpg" \
        --out outputs/_demo/clip

The costume reference is cropped to its garment band, so a portrait from the
character pack can be pointed at directly.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_erase_scope import imwrite_unicode  # noqa: E402
from demo_replace import costume_crop, grab_frame  # noqa: E402
from demo_erase_scope import imread_unicode  # noqa: E402
from recast.core.person_replace import replace_clip  # noqa: E402
from recast.stages.person_mask import describe_backends, estimate_neck_y, pick_backend  # noqa: E402
from recast.stages.track_mask import TemporalMasker  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="起始秒")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--costume", default="", help="服装参考图；给了 --character 时可省略")
    ap.add_argument("--part", default="upper", choices=["upper", "lower", "full"])
    ap.add_argument(
        "--auto-garment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "只换最有可能是衣服的那一簇（默认关）。素色单件时有用；"
            "针织/印花上会把明暗分块各成一簇，结果变成花斑，所以整段推荐关掉"
        ),
    )
    ap.add_argument("--redetect", type=int, default=6, help="每隔多少帧重检测一次")
    ap.add_argument("--smooth", type=float, default=0.55)
    ap.add_argument("--delivery", action="store_true", help="交付跑：拒绝非商用模型")
    ap.add_argument(
        "--face",
        action="store_true",
        help="同时换脸。需要 --face-ref 指定参考脸（或用 --character 指向人物文件夹）",
    )
    ap.add_argument("--face-ref", default="", help="参考脸图片路径")
    ap.add_argument(
        "--character",
        default="",
        help="人物文件夹（如 outputs/characters/林知夏）：一次拿它的脸和衣服",
    )
    ap.add_argument("--out", default="outputs/_demo/clip")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend, reason = pick_backend("auto", allow_noncommercial=not args.delivery)
    licences = {k: v.get("license") for k, v in describe_backends().items()}
    print(f"分割后端：{backend}" + (f"（{reason}）" if reason else ""))
    print(f"  许可证：{licences.get(backend, '—')}")

    # A character folder is the unit the pipeline reads from.
    char = out_dir / "char"
    (char / "costume").mkdir(parents=True, exist_ok=True)
    (char / "face").mkdir(parents=True, exist_ok=True)
    IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if args.character:
        src_char = Path(args.character)
        faces = sorted(p for p in (src_char / "face").glob("*") if p.suffix.lower() in IMAGES)
        costumes = sorted(p for p in (src_char / "costume").glob("*") if p.suffix.lower() in IMAGES)
        if not costumes:
            raise SystemExit(f"{src_char / 'costume'} 里没有服装参考图")
        imwrite_unicode(char / "costume" / "01.png", imread_unicode(costumes[0]))
        if faces:
            imwrite_unicode(char / "face" / "01.png", imread_unicode(faces[0]))
        print(f"人物包：{src_char.name}（衣服 {costumes[0].name}，脸 {faces[0].name if faces else '无'}）")
    elif args.costume:
        reference = costume_crop(imread_unicode(Path(args.costume)))
        imwrite_unicode(char / "costume" / "01.png", reference)
    else:
        raise SystemExit("要么给 --costume，要么给 --character")

    if args.face_ref:
        imwrite_unicode(char / "face" / "01.png", imread_unicode(Path(args.face_ref)))

    # Cut the segment so a clip run does not re-read from frame 0.
    import cv2

    source = cv2.VideoCapture(str(args.video))
    fps = float(source.get(cv2.CAP_PROP_FPS) or 25.0)
    source.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    segment = out_dir / "segment.mp4"
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(segment), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for _ in range(args.frames):
        ok, frame = source.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    source.release()
    print(f"片段：{segment.name}  {width}x{height}  {fps:.0f}fps  {args.frames} 帧")

    masker = TemporalMasker(
        backend=backend,
        redetect_every=args.redetect,
        smooth=args.smooth,
        allow_noncommercial=not args.delivery,
    )

    neck_cache: dict[str, int] = {}

    def neck_y_fn(frame, _index):
        # One face estimate per clip: the neck line does not move within a
        # shot, and insightface per frame would cost more than the mask.
        if "y" not in neck_cache:
            neck_cache["y"] = int(estimate_neck_y(frame)["neck_y"])
        return neck_cache["y"]

    started = time.perf_counter()
    result = replace_clip(
        segment,
        out_dir / "replaced.mp4",
        character_folder=char,
        mask_fn=masker,
        neck_y_fn=neck_y_fn,
        parts=(args.part,),
        auto_garment=args.auto_garment,
        do_face=args.face,
        allow_noncommercial=not args.delivery,
    )
    elapsed = time.perf_counter() - started

    print(f"\n输出：{result['out_path']}")
    print(f"帧数 {result['frames']}  用时 {elapsed:.1f}s  =  {elapsed / result['frames'] * 1000:.0f} ms/帧")
    print(
        f"审计：{result['audited']} 帧有数，"
        f"接缝增量均值 {result['seam_ratio_mean']}（原始接缝比），"
        f"峰值 {result['seam_ratio_max']}，需复看 {result['review_frames']} 帧"
    )
    rejected_n = result.get("rejected_frames", 0)
    counts = result.get("reject_reason_counts") or {}
    print(f"拦截了 {rejected_n} 帧，原因分布：{counts if counts else '无'}")
    print(f"掩码：{masker.report()}")
    if result["worst_frames"]:
        print("最差帧：")
        for row in result["worst_frames"][:5]:
            print(f"  #{row['frame']:>4}  接缝 {row['seam_ratio']}  {row['verdict']}")
    for note in result["notes"]:
        print(f"备注：{note}")

    per_frame = elapsed / result["frames"]
    full = 5056 * per_frame  # the 168 s episode in demo/1
    print(f"\n外推：168 秒一集（5056 帧）约 {full / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
