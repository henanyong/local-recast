# -*- coding: utf-8 -*-
"""Windows installer for 本地再铸. Called by install_windows.bat."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
REQ = ROOT / "requirements.txt"


def run(cmd: list[str], extra_env: dict[str, str] | None = None) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(">", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=ROOT, env=env)
    return int(completed.returncode)


def python_ok(version: tuple[int, int, int]) -> bool:
    return version[0] == 3 and 10 <= version[1] <= 12


def main() -> int:
    os.chdir(ROOT)
    print("=" * 44)
    print("  本地再铸 V1.0  Windows 安装")
    print("  目标硬件: i5-12600KF + RTX 4060 Ti（默认 8GB）")
    print("=" * 44)
    print()
    print("请先确认:")
    print("  1. 已安装 Python 3.11（推荐；3.12 一般可用。不要用 3.13）")
    print("  2. python 已加入 PATH")
    print("  3. 已安装 ffmpeg，并且 ffmpeg / ffprobe 在 PATH 中")
    print("     下载: https://www.gyan.dev/ffmpeg/builds/")
    print("     也可以把 ffmpeg 目录放到本项目的 ffmpeg\\ 下")
    print("  4. 已安装 NVIDIA 驱动，可运行 nvidia-smi")
    print()

    print(f"当前 Python: {sys.version.split()[0]}  ({sys.executable})")
    if not python_ok(sys.version_info[:3]):
        print("[警告] 当前 Python 版本不是 3.10-3.12。insightface / mediapipe 在 3.13 上经常没有可用轮子。")
        print("建议改用 Python 3.11 后重新运行本脚本。")
        print()

    if subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        print("[警告] 找不到 ffmpeg。请安装并把 ffmpeg.exe 所在目录加入 PATH 后再继续。")
        print()

    if not VENV_PY.exists():
        print("[1/5] 创建虚拟环境 .venv ...")
        code = run([sys.executable, "-m", "venv", ".venv"])
        if code != 0:
            print("[错误] 创建虚拟环境失败。")
            return code
    else:
        print("[1/5] 已存在虚拟环境 .venv，跳过创建。")

    if not VENV_PY.exists():
        print("[错误] 找不到 .venv\\Scripts\\python.exe。")
        return 1

    py = str(VENV_PY)

    print("[2/5] 升级 pip ...")
    code = run([py, "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"])
    if code != 0:
        print("[错误] 升级 pip 失败。")
        return code

    print("[3/5] 安装 PyTorch CUDA 12.4 官方轮子 ...")
    print("    如需 CUDA 12.6，请改为:")
    print("    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
    code = run(
        [py, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"]
    )
    if code != 0:
        print("[错误] 安装 PyTorch 失败，请检查网络或改用 cu126 索引。")
        return code

    print("[4/5] 安装项目依赖并注册 recast 包 ...")
    if not REQ.exists():
        print("[错误] 找不到 requirements.txt")
        return 1
    code = run([py, "-m", "pip", "install", "-r", str(REQ)])
    if code != 0:
        print("[错误] 安装 requirements.txt 失败。")
        return code
    code = run([py, "-m", "pip", "install", "-e", "."])
    if code != 0:
        print("[错误] pip install -e . 失败。")
        return code

    print("[5/5] 尝试安装可选的 RapidOCR（画面字幕）。失败不影响主流程。")
    code = run([py, "-m", "pip", "install", "rapidocr>=2.0"])
    if code != 0:
        print("[警告] RapidOCR 安装失败，解析提取将只用语音转写 ASR。可稍后手动安装 rapidocr。")

    print()
    print("安装完成。接下来请运行自检:")
    print(r"  .venv\Scripts\python.exe -m recast doctor")
    print()
    print("启动桌面工作台（V1.0 主界面）:")
    print(r"  .venv\Scripts\python.exe -m recast ui")
    print()
    print("网页工作台备用:")
    print(r"  .venv\Scripts\python.exe -m recast ui-web")
    print()
    print("注意:")
    print("  - 请使用 Python 3.11，不要用 3.13。")
    print("  - 多数 4060 Ti 是 8GB，请默认使用 configs\\4060ti-8g.yaml。")
    print("  - 第一次跑某个阶段时会自动下载模型（合计约 1-1.5 GB），请保持网络畅通。")
    print("  - 本版「人物替换」是本地换脸，不是提取了3D模型。")
    print("  - InsightFace / inswapper 请自行阅读其许可证后再用于你的素材。")
    print("  - 离线运行不需要任何云密钥。云通道与自动上传默认关闭。")
    print("  - 未连接 ComfyUI 时，按剧本生成使用「分镜预览成片」，不是完整图生视频。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
