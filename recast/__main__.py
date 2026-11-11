"""CLI entry: python -m recast process | ui | ui-web | ui-simple | doctor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from recast import __version__
from recast.config import DEFAULT_CONFIG, RecastConfig, resolve_config_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m recast",
        description="本地再铸 V1.0：导入现成视频或按剧本生成，L1 人物替换、四级模板、成本表与 3D 头像资产（4060 Ti）。",
    )
    parser.add_argument("--version", action="version", version=f"recast {__version__}")
    sub = parser.add_subparsers(dest="command", required=False)

    p = sub.add_parser("process", help="处理一支视频：提取字幕/人物库，可选 L1 人物替换后导出")
    p.add_argument("input", help="输入视频路径，例如 INPUT.mp4")
    p.add_argument("--face", default=None, help="参考脸 / 角色图（可选；不提供则跳过替换）")
    p.add_argument("--out", default=None, help="成片路径，默认写到 outputs/<run>/output.mp4")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置文件，默认 configs/4060ti-8g.yaml")
    p.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="只替换这一条轨迹（覆盖该轨迹所属人物的整组聚类）。不填则替换每个已匹配人物聚类里的全部轨迹",
    )
    p.add_argument("--names", default=None, help="人名花名册 YAML，或 '1=张三,2=李四'")
    p.add_argument("--skip-replace", action="store_true", help="只提取字幕/轨迹/姿态，不换脸")
    p.add_argument("--lang", default="auto", help="字幕语言，默认 auto（对中文友好）；可设 zh / en")
    p.add_argument("--burn-subs", action="store_true", help="把字幕烧进成片（同时仍写出旁路 SRT）")
    p.add_argument("--only", default=None, help="只跑指定阶段，逗号分隔: scan,extract,library,replace,compose（也兼容 preprocess,subtitles 等旧名）")
    p.add_argument("--workdir", default=None, help="复用已有运行目录，便于续跑某一阶段")
    p.add_argument("--force-rerun", action="store_true", help="强制重跑所选阶段，即使产物已存在")

    u = sub.add_parser("ui", help="启动桌面工作台（PySide6 主界面）")
    uw = sub.add_parser("ui-web", help="启动网页版批量工作台（FastAPI 备用）")
    uw.add_argument("--host", default="127.0.0.1", help="监听地址，默认本机")
    uw.add_argument("--port", type=int, default=7860, help="端口")
    uw.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")

    us = sub.add_parser("ui-simple", help="启动单文件 Gradio 界面（备用）")
    us.add_argument("--config", default=str(DEFAULT_CONFIG), help="默认配置文件")
    us.add_argument("--host", default="127.0.0.1", help="监听地址")
    us.add_argument("--port", type=int, default=7860, help="端口")
    us.add_argument("--share", action="store_true", help="开启 Gradio share 隧道")

    d = sub.add_parser("doctor", help="自检 Python / CUDA / ffmpeg / 配置")
    d.add_argument("--config", default=str(DEFAULT_CONFIG), help="要检查的配置文件")
    return parser


def _cmd_process(args: argparse.Namespace) -> int:
    from recast.pipeline import process, setup_logging

    setup_logging()
    stages = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    summary = process(
        args.input,
        face=args.face,
        out=args.out,
        config=args.config,
        track_id=args.track_id,
        names=args.names,
        skip_replace=args.skip_replace,
        lang=args.lang,
        burn_subs=True if args.burn_subs else None,
        stages=stages,
        workdir=args.workdir,
        force_rerun=bool(args.force_rerun),
    )
    print(f"\n完成。成片: {summary['output']}")
    print(f"工作目录: {summary['workdir']}")
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    from recast.desktop.app import launch

    return int(launch() or 0)


def _cmd_ui_web(args: argparse.Namespace) -> int:
    from recast.workbench import launch

    launch(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def _cmd_ui_simple(args: argparse.Namespace) -> int:
    from recast.ui import launch

    launch(config=args.config, host=args.host, port=args.port, share=args.share)
    return 0


def _ok(ok: bool) -> str:
    return "OK" if ok else "缺"


def _cmd_doctor(args: argparse.Namespace) -> int:
    print(f"recast {__version__}  自检")
    print("-" * 48)
    issues: list[str] = []

    ver = sys.version.replace("\n", " ")
    py_ok = sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12
    print(f"[{_ok(py_ok)}] Python  {ver}")
    print(f"      可执行文件: {sys.executable}")
    if not py_ok:
        issues.append("建议使用 Python 3.10-3.12（推荐 3.11）。3.13 上 insightface / mediapipe 经常没有轮子。")

    try:
        cfg_path = resolve_config_path(args.config)
        cfg = RecastConfig.from_yaml(cfg_path)
        print(f"[OK] 配置    {cfg_path}")
        print(f"      device={cfg.device}  vram_gb={cfg.vram_gb}  whisper={cfg.whisper.model}  detect={cfg.detect.model}  imgsz={cfg.detect.imgsz}")
        if cfg.vram_gb >= 16:
            print("      提示: 这是 16GB 配置。多数 4060 Ti 是 8GB，请确认后再用。")
    except Exception as exc:
        print(f"[缺] 配置    {exc}")
        issues.append(f"无法加载配置: {exc}")
        cfg = None

    try:
        import torch

        print(f"[OK] PyTorch {torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"[{_ok(cuda)}] CUDA     torch.cuda.is_available()={cuda}")
        if cuda:
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"      设备: {name}  能力 {major}.{minor}  显存 {total:.1f} GB")
            if cfg is not None and total < 10 and cfg.vram_gb >= 16:
                issues.append("当前卡看起来不是 16GB，请改用 configs/4060ti-8g.yaml。")
            # allocate nothing, just report current
            free_b, total_b = torch.cuda.mem_get_info()
            print(f"      当前空闲 {free_b/(1024**3):.2f}/{total_b/(1024**3):.2f} GB")
        else:
            print("      未检测到 CUDA GPU，将按基础档运行：切镜/字幕/分镜预览成片可用，L2 驱动与 ComfyUI 重生成已禁用。")
            print("      若这是目标电脑，请用 scripts/install_windows.bat 安装官方 cu124/cu126 轮子，并确认 nvidia-smi 正常。")
    except Exception as exc:
        print(f"[缺] PyTorch 未安装或导入失败: {exc}")
        issues.append("未安装 PyTorch。请先运行 scripts/install_windows.bat。")

    import shutil

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    print(f"[{_ok(bool(ffmpeg))}] ffmpeg   {ffmpeg or '不在 PATH'}")
    print(f"[{_ok(bool(ffprobe))}] ffprobe  {ffprobe or '不在 PATH'}")
    if not ffmpeg or not ffprobe:
        issues.append("请安装 ffmpeg 并加入 PATH（Windows 推荐 https://www.gyan.dev/ffmpeg/builds/ ）。")

    modules = [
        ("faster_whisper", "faster-whisper"),
        ("ultralytics", "ultralytics"),
        ("insightface", "insightface"),
        ("onnxruntime", "onnxruntime-gpu"),
        ("cv2", "opencv-python"),
        ("sklearn", "scikit-learn"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("huggingface_hub", "huggingface_hub"),
        ("PySide6", "PySide6（桌面工作台）"),
    ]
    optional = [
        ("scenedetect", "scenedetect（切镜首选，缺了会回退到 FFmpeg）"),
        ("rapidocr", "rapidocr 或 rapidocr_onnxruntime（画面 OCR；缺了只用 ASR 字幕）"),
        ("fastapi", "fastapi（网页工作台备用）"),
        ("uvicorn", "uvicorn（网页工作台备用）"),
        ("gradio", "gradio（单文件备用界面）"),
        ("rtmlib", "rtmlib（可选姿态，缺了会回退）"),
        ("mediapipe", "mediapipe（姿态回退）"),
        ("openpyxl", "openpyxl（成本表 xlsx）"),
        ("cryptography", "cryptography（云密钥加密，缺了会用本地密钥文件）"),
        ("edge_tts", "edge-tts（微软在线语音，不是本地引擎；离线请用 piper 模型或导入 wav）"),
    ]
    print("-" * 48)
    print("依赖检查（不下载模型）:")
    for mod, label in modules:
        try:
            __import__(mod)
            print(f"  [OK] {label}")
        except Exception as exc:
            print(f"  [缺] {label}: {exc}")
            issues.append(f"缺少依赖 {label}")
    for mod, label in optional:
        try:
            __import__(mod)
            print(f"  [OK] {label}")
        except Exception as exc:
            print(f"  [..] {label}: 未安装（{exc}）。流水线仍可回退。")

    # OCR can live under either package name
    try:
        from recast.extract.ocr import ocr_available
        ok, reason = ocr_available()
        print(f"  [{'OK' if ok else '..'}] 画面 OCR: {'可用' if ok else '未安装，提取阶段将只用 ASR 字幕'}")
    except Exception as exc:
        print(f"  [..] 画面 OCR 检查失败: {exc}")

    try:
        from recast.core.engine import first_run_report
        from recast.core.settings import AppSettings
        from recast.store.db import Store, default_db_path
        from recast.store.migrate import SCHEMA_VERSION, current_version

        store = Store(default_db_path())
        ver = current_version(store.connect())
        print(f"  [OK] 数据库 schema v{ver}（当前 {SCHEMA_VERSION}，V0.1 库会自动迁移）")
        report = first_run_report(AppSettings())
        snap = report["snapshot"]
        print(f"  硬件档建议: {report.get('suggested_tier')}    磁盘剩余: {snap.get('disk_free_gb')}")
        comfy = snap.get("comfyui") or {}
        print(f"  [{'OK' if comfy.get('ok') else '..'}] ComfyUI: {'已连接 ' + (comfy.get('url') or '') if comfy.get('ok') else '未连接，按剧本生成将使用「分镜预览成片」'}")
        print(f"  [{'OK' if snap.get('liveportrait') else '..'}] LivePortrait: {'已安装' if snap.get('liveportrait') else '未安装，3D头像资产仍可保存，不能当场驱动'}")
        for item in report.get("fixes") or []:
            print(f"  提示: {item}")
    except Exception as exc:
        print(f"  [..] 扩展自检未完成: {exc}")

    print("-" * 48)
    if issues:
        print("需要处理:")
        for item in issues:
            print(f"  - {item}")
        print("修复后可运行:  python -m recast ui")
        return 1
    print("自检通过。可以运行:")
    print("  python -m recast ui          # 桌面工作台（V1.0 主界面）")
    print("  python -m recast ui-web      # 网页工作台备用")
    print("  python -m recast process INPUT.mp4 --face REF.jpg --out OUT.mp4 --config configs/4060ti-8g.yaml")
    print("说明: 默认 4060 Ti 8GB。L1 人物替换是本地换脸，不是提取了3D模型。L2 才叫 3D头像。自动上传默认关闭。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        from recast.desktop.app import launch
        return int(launch() or 0)
    try:
        if args.command == "process":
            return _cmd_process(args)
        if args.command == "ui":
            return _cmd_ui(args)
        if args.command == "ui-web":
            return _cmd_ui_web(args)
        if args.command == "ui-simple":
            return _cmd_ui_simple(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        parser.error("未知命令")
        return 2
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
