"""Drive the character-library window end to end, through its real buttons.

This is a smoke test, not a mock: it builds the actual ``LibraryWindow``,
clicks the actual ``QPushButton``s and reads the status bar the user reads.
The only things replaced are the modal dialogs (file picker, message boxes),
which cannot be answered from a script — everything between them runs for real.

Usage:
    python scripts/ui_drive.py --clip D:\\path\\clip_4s.mp4
    python scripts/ui_drive.py --clip ... --character 林知夏 --skip-auto-define
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STEPS: list[str] = []


def log(step: str, ok: bool, detail: str = "") -> None:
    STEPS.append(f"{'PASS' if ok else 'FAIL'}  {step}" + (f"  |  {detail}" if detail else ""))
    print(STEPS[-1], flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="用真实按钮驱动人物库窗口跑一遍")
    ap.add_argument("--clip", required=True, help="几秒钟的测试片段或图片")
    ap.add_argument("--character", default="", help="要驱动的人物名，默认取列表第一个")
    ap.add_argument("--skip-auto-define", action="store_true", help="跳过自动定义人物（说话人分离较慢）")
    ap.add_argument("--skip-matte", action="store_true", help="跳过抠像")
    ap.add_argument("--shots", default="", help="把每一步的窗口截图存到这个目录（做手册用）")
    args = ap.parse_args()

    clip = Path(args.clip)
    if not clip.is_file():
        print(f"找不到片段：{clip}")
        return 2

    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    from recast.core.service import RecastService
    from recast.desktop.library_window import LibraryWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])

    dialogs: list[str] = []

    def remember(kind: str):
        def _stub(*_a, **_k):
            text = f"{kind}: {_a[2] if len(_a) > 2 else ''}"
            dialogs.append(text)
            print(f"  [弹窗] {text}", flush=True)
            return QMessageBox.StandardButton.Ok if kind != "information" else None

        return _stub

    # Modals cannot be clicked from a script; record them instead of blocking.
    QMessageBox.warning = staticmethod(remember("warning"))  # type: ignore[assignment]
    QMessageBox.critical = staticmethod(remember("critical"))  # type: ignore[assignment]
    QMessageBox.information = staticmethod(remember("information"))  # type: ignore[assignment]
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(clip), ""))  # type: ignore[assignment]
    QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "")  # type: ignore[assignment]

    try:
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl = staticmethod(lambda url: True)  # type: ignore[assignment]
    except Exception:
        pass

    svc = RecastService()
    window = LibraryWindow(svc)
    window._sync_tasks = True  # tasks run inline: no threads to wait on
    window.show()
    app.processEvents()
    window.reload()
    app.processEvents()

    shots = Path(args.shots) if args.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)
    shot_no = {"n": 0}

    def snap(tag: str) -> None:
        """Grab the real window, so the manual shows what the user will see."""
        if not shots:
            return
        shot_no["n"] += 1
        app.processEvents()
        pixmap = window.grab()
        target = shots / f"{shot_no['n']:02d}-{tag}.png"
        pixmap.save(str(target))
        print(f"  [截图] {target.name} {pixmap.width()}x{pixmap.height()}", flush=True)

    rows = window.list.count()
    log("窗口加载", rows > 0, f"列表 {rows} 行")
    snap("人物库总览")

    # 1 导入
    window.import_btn.click()
    app.processEvents()
    label = window.source_label.text()
    log("导入视频/照片", str(clip.name) in label, label)
    snap("导入片段")

    # 2 自动定义人物
    if not args.skip_auto_define:
        window.auto_define_btn.click()
        app.processEvents()
        status = window.status_label.text()
        ok = "失败" not in status
        log("自动定义人物", ok, status[:150])
        snap("自动定义人物")

    # 3 选人
    target = args.character
    index = -1
    for i in range(window.list.count()):
        text = window.list.item(i).text()
        if not target or text.startswith(target):
            index = i
            break
    if index < 0:
        log("选中人物", False, f"列表里没有 {target or '任何人物'}")
        print("\n".join(STEPS))
        return 1
    window.list.setCurrentRow(index)
    app.processEvents()
    log("选中人物", bool(window._current_id), window.list.item(index).text().split("·")[0].strip())
    snap("选中人物")

    # 4 建人物文件夹
    window.build_folder_btn.click()
    app.processEvents()
    log("建人物文件夹", "失败" not in window.status_label.text(), window.status_label.text()[:120])

    # 5 抠像
    if not args.skip_matte:
        window.matte_btn.click()
        app.processEvents()
        text = window.status_label.text()
        failed_n = re.search(r"失败 (\d+) 张", text)
        ok = "抠像完成" in text and (failed_n is None or int(failed_n.group(1)) == 0)
        log("抠像", ok, text[:150])
        snap("抠像完成")

    # 6 设定板
    window.sheet_btn.click()
    app.processEvents()
    log("生成设定板", "失败" not in window.status_label.text(), window.status_label.text()[:150])
    snap("生成设定板")

    # 7 换装预览
    window.garment_btn.click()
    app.processEvents()
    status = window.status_label.text()
    # The pair is overwritten each run, so "new file" is the wrong signal; the
    # status line carries the audit numbers and the path, which is what matters.
    made = re.findall(r"对比图：(.+)", status)
    pair = bool(made) and Path(made[0].strip()).is_file()
    log("换装预览", "失败" not in status and pair, status[:140])
    snap("换装预览")

    window.close()
    print("\n--- 汇总 ---")
    print(f"{sum(1 for s in STEPS if s.startswith('PASS'))}/{len(STEPS)} 通过")
    if dialogs:
        print("弹窗：")
        for d in dialogs:
            print(" -", d[:200])
    return 0 if all(s.startswith("PASS") for s in STEPS) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
