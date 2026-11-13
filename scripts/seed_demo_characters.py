"""Seed the character library with the demo test set.

The demo assets live outside the project (``Desktop/demo/测试素材``) because
they are sample material, not code. This script copies them into the
configured ``output_root`` as proper character folders and registers one row
per character so the UI lists them without any manual import.

Everything it creates is tagged ``source_task_id = demo-seed`` so it can be
removed again with ``--remove``; nothing else in the library is touched.

Usage:
    python scripts/seed_demo_characters.py            # import (idempotent)
    python scripts/seed_demo_characters.py --dry-run   # show what would happen
    python scripts/seed_demo_characters.py --remove    # take the demo set out
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED_TAG = "demo-seed"
SUBDIRS = ("face", "costume", "voice", "cutout", "model")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_ASSETS = Path.home() / "Desktop" / "demo" / "测试素材"


def _person_folders(assets: Path) -> list[Path]:
    root = assets / "人物"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "meta.json").is_file())


def _copy_tree(src: Path, dest: Path, *, dry_run: bool) -> int:
    copied = 0
    for sub in SUBDIRS:
        (dest / sub).mkdir(parents=True, exist_ok=True)
    for path in sorted(src.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = path.relative_to(src)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == path.stat().st_size:
            continue
        if not dry_run:
            shutil.copy2(path, target)
        copied += 1
    return copied


def seed(assets: Path, *, dry_run: bool = False) -> int:
    from recast.core.service import RecastService

    svc = RecastService()
    output_root = Path(svc.settings.output_root)
    folders = _person_folders(assets)
    if not folders:
        print(f"没有找到人物素材：{assets / '人物'}")
        return 2

    existing = {str(c.get("display_name") or c.get("name") or "") for c in svc.store.list_characters()}
    print(f"素材目录：{assets}")
    print(f"人物库：  {output_root}")
    print("")

    for folder in folders:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        name = str(meta.get("name") or folder.name).strip()
        created = svc.ensure_character_folder(name, name_source="manual")
        dest = Path(created["folder"])
        copied = _copy_tree(folder, dest, dry_run=dry_run)

        if not dry_run:
            # Copy the authored meta over the blank one so costume description,
            # assembly and the authorization note survive the import.
            (dest / "meta.json").write_text(
                json.dumps({**meta, "name": name}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if name not in existing:
                svc.store.insert(
                    "characters",
                    name=name,
                    display_name=name,
                    confirmed=1,
                    authorized=1,
                    auth_type="AI 生成虚构角色",
                    cluster_key=f"{SEED_TAG}:{name}",
                    source_task_id=SEED_TAG,
                    source_note="测试素材，非真实人物",
                )
                existing.add(name)
                state = "新建"
            else:
                state = "已存在"
        else:
            state = "预览"
        print(f"  [{state}] {name}: {copied} 张图 -> {dest}")
    print("")
    print("打开人物库即可看到：  .venv\\Scripts\\python.exe -m recast ui")
    return 0


def remove(*, dry_run: bool = False) -> int:
    from recast.core.service import RecastService

    svc = RecastService()
    rows = [c for c in svc.store.list_characters() if str(c.get("source_task_id") or "") == SEED_TAG]
    if not rows:
        print("人物库里没有 demo-seed 记录，无需清理。")
        return 0
    output_root = Path(svc.settings.output_root)
    for row in rows:
        name = str(row.get("display_name") or row.get("name") or "")
        folder = output_root / "characters" / name
        print(f"  [{'预览' if dry_run else '删除'}] {name}  ({folder})")
        if not dry_run:
            svc.store.delete("characters", row["id"])
            if folder.is_dir():
                shutil.rmtree(folder)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="把 demo 测试人物导入人物库")
    ap.add_argument("--assets", default=str(DEFAULT_ASSETS), help="测试素材根目录")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不落盘")
    ap.add_argument("--remove", action="store_true", help="移除之前导入的测试人物")
    args = ap.parse_args()
    if args.remove:
        return remove(dry_run=args.dry_run)
    return seed(Path(args.assets), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
