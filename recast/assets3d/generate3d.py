"""Automatic 3D generation: image -> GLB model, registered in the character folder.

Backends are pluggable (``GenerationBackend`` protocol); the default is
TripoSG (VAST-AI-Research/TripoSG, MIT, 1.5B rectified flow transformer).
What is verified, and what is not:

- Verified from the TripoSG project pages: it needs a CUDA GPU with at
  least 8GB VRAM, Python 3.10, conda + pip install, and its weights
  download from Hugging Face on first run. It generates *shape only* —
  the exported GLB has no texture. (TripoSR, the lighter sibling, needs
  ~6GB VRAM and has an optional --bake-texture flag.)
- Verified on this machine (2026-11-12): no GPU (nvidia-smi absent), no
  torch installed, 8GB RAM total. TripoSG cannot run here, so it was NOT
  installed — installing it would only burn disk and time.
- Verified 2026-11-12 against the real TripoSG README: the entry point is
  ``python -m scripts.inference_triposg --image-input <img> --output-path
  <out.glb>`` (run from the repo root; there is no run.py). Weights
  auto-download to ``pretrained_weights/`` on first run. The backend calls
  exactly that command, and the produced GLB is validated with
  ``validate_glb`` afterwards, so a wrong invocation can never silently
  produce a fake model.

On the target machine (Windows + RTX 4060 Ti 8GB): clone the repo, install
its requirements, let the weights download on first run, then point
``model3d.triposg_repo`` at it (and ``model3d.triposg_python`` at the
TripoSG conda env's python when it is not the same interpreter running
local-recast). One character at a time — never batch in parallel on 8GB.

Registration: a generated model is written to ``model/model.glb`` and
registered in meta.json ``model`` as {source: "auto", format: "glb",
path, backend, generated_at, source_image, note} — and ``assembly.body``
is pointed at it, so the assembly view can use it right away. A failed
generation registers nothing. Regenerating keeps any existing costume
variants and costume bindings (see recast/assets3d/variants.py).

Honest limit: TripoSG generates shape only — no texture, no skeleton.
A generated GLB works for assembly display and costume variants, but
NOT for body re-rendering (that needs a rigged model or auto-rigging).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger("recast")

TRIPOSG_REPO_URL = "https://github.com/VAST-AI-Research/TripoSG"
TRIPOSG_MIN_VRAM_GB = 8
# Real entry point, per the TripoSG README (verified 2026-11-12):
#   python -m scripts.inference_triposg --image-input <img> --output-path <out.glb>
TRIPOSG_ENTRY = Path("scripts") / "inference_triposg.py"

# Windows 准备步骤（界面失败提示和按钮文案直接复用这份，别让用户猜）。
# 顺序不能乱：先跑通 example 图，再在 local-recast 里配路径。
TRIPOSG_SETUP_STEPS: tuple[str, ...] = (
    "1. 克隆仓库：git clone https://github.com/VAST-AI-Research/TripoSG",
    "2. 建 conda 环境：conda create -n tripoSG python=3.10（TripoSG 要求 Python 3.10）",
    "3. 在 tripoSG 环境里按 TripoSG README 装 CUDA 版 torch（cu121），再 pip install -r requirements.txt（在仓库目录里跑）",
    "4. 先用仓库自带的 example 图跑一遍：python -m scripts.inference_triposg --image-input <example 图> --output-path <out.glb>，确认能出 GLB（首次运行会自动从 Hugging Face 下载权重到 pretrained_weights/，需要联网）",
    "5. 在 local-recast 配置里填 model3d.triposg_repo（克隆出来的目录）和 model3d.triposg_python（tripoSG 环境的 python 路径）",
    "6. 再点「生成 3D 模型」。一次只生成一个人，不要并行（8GB 显存）。",
)
TRIPOSG_SETUP_TEXT = "TripoSG 准备步骤（Windows + RTX 4060 Ti）：\n" + "\n".join(TRIPOSG_SETUP_STEPS)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class GenerationError(RuntimeError):
    """Automatic 3D generation cannot run: backend, model or GPU is missing."""


class GenerationBackend(Protocol):
    """Anything that can turn one image into a GLB file."""

    name: str

    def check(self) -> tuple[bool, str]:
        """(available, reason). ``reason`` is empty when available."""
        ...

    def generate(self, image: str | Path, dest: str | Path) -> Path:
        """Generate a GLB from ``image`` and write it to ``dest``."""
        ...


class TripoSGBackend:
    """TripoSG image-to-3D, invoked as ``python -m scripts.inference_triposg``."""

    name = "triposg"

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        *,
        python: str | None = None,
    ) -> None:
        self.repo_dir = Path(repo_dir).expanduser() if repo_dir else None
        self.python = python or sys.executable

    def check(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, (
                "自动生成 3D 需要 torch，未安装。请在 Windows 上按 TripoSG 的要求装好 "
                "torch（CUDA 版），或先用手动导入 GLB。"
                "下一步：准备步骤 2–3（建 tripoSG 环境、装 CUDA 版 torch）。"
            )
        try:
            import torch

            if not torch.cuda.is_available():
                return False, (
                    "自动生成 3D 需要 CUDA 显卡，未检测到可用 GPU。"
                    f"TripoSG 至少需要 {TRIPOSG_MIN_VRAM_GB}GB 显存。"
                    "这台机器没有 GPU，请在 Windows + RTX 4060 Ti 上跑，或先用手动导入 GLB。"
                )
        except Exception:
            return False, "无法检测 GPU 状态，请检查 torch 安装。"
        if self.repo_dir is None or not self.repo_dir.is_dir():
            return False, (
                "找不到 TripoSG 代码目录。请先克隆仓库："
                f"git clone {TRIPOSG_REPO_URL}，"
                "按仓库 README 装好依赖和模型权重，"
                "再把 model3d.triposg_repo 指到克隆出来的目录。"
                "下一步：准备步骤 1（克隆仓库）和步骤 5（填 triposg_repo）。"
            )
        if not (self.repo_dir / TRIPOSG_ENTRY).is_file():
            return False, (
                f"TripoSG 目录 {self.repo_dir} 里没有 {TRIPOSG_ENTRY}，"
                "可能仓库版本变了或目录指错了。请检查目录，或按当前仓库 README 的入口重新接线。"
                "下一步：确认 triposg_repo 指的是克隆出来的 TripoSG 仓库根目录（里面应有 scripts/inference_triposg.py）。"
            )
        return True, ""

    def generate(self, image: str | Path, dest: str | Path) -> Path:
        from recast.assets3d.glb import validate_glb

        ok, reason = self.check()
        if not ok:
            raise GenerationError(reason)
        image = Path(image)
        dest = Path(dest)
        if not image.is_file():
            raise GenerationError(f"找不到要生成 3D 的图片：{image}")
        assert self.repo_dir is not None
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Real CLI, per the TripoSG README (verified 2026-11-12):
        #   python -m scripts.inference_triposg --image-input <img> --output-path <out.glb>
        cmd = [
            self.python,
            "-m",
            "scripts.inference_triposg",
            "--image-input",
            str(image),
            "--output-path",
            str(dest),
        ]
        log.info("TripoSG 生成中: %s", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, cwd=str(self.repo_dir), capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired as exc:
            raise GenerationError(f"TripoSG 生成超时（1 小时）：{exc}") from exc
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-500:]
            raise GenerationError(f"TripoSG 生成失败（退出码 {proc.returncode}）：{tail}")
        if not dest.is_file():
            raise GenerationError(
                f"TripoSG 跑完了，但在 {dest} 没找到生成的 GLB。"
                "可能是仓库版本变了，输出位置和预期的不一样。"
            )
        validate_glb(dest)
        log.info("TripoSG 生成完成: %s", dest)
        return dest


def get_backend(
    kind: str,
    *,
    repo_dir: str | Path | None = None,
) -> GenerationBackend:
    """Build a generation backend by name. Only "triposg" exists for now."""
    kind = (kind or "triposg").strip().lower()
    if kind == "triposg":
        return TripoSGBackend(repo_dir)
    raise GenerationError(f"未知的 3D 自动生成方式：{kind}（目前只有 triposg）。")


def generate_model_glb(
    image: str | Path,
    char_folder: str | Path,
    *,
    backend: GenerationBackend | None = None,
    repo_dir: str | Path | None = None,
    python: str | None = None,
) -> dict[str, Any]:
    """Generate a GLB from one image into the character folder's model/.

    Registers meta.json ``model`` as {source: "auto", format: "glb",
    path, backend, generated_at, source_image, note} and points
    ``assembly.body`` at the new GLB, so the assembly view can use it
    right away. Existing costume variants and bindings are kept.
    The source image is usually the character's cutout PNG (transparent
    background helps the generator). The backend is injectable for tests.
    ``python`` is the interpreter TripoSG runs under (its own conda env).
    """
    from recast.assets3d.glb import validate_glb
    from recast.core.characters import load_meta, save_meta, set_assembly

    folder = Path(char_folder)
    if not folder.is_dir():
        raise GenerationError(f"找不到人物文件夹：{folder}")
    image = Path(image)
    if not image.is_file():
        raise GenerationError(f"找不到要生成 3D 的图片：{image}")
    if backend is None:
        backend = TripoSGBackend(repo_dir, python=python)
    model_dir = folder / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / "model.glb"
    backend.generate(image, dest)
    info = validate_glb(dest)
    meta = load_meta(folder) or {"name": folder.name}
    name = meta.get("name") or folder.name
    old_model = meta.get("model") if isinstance(meta.get("model"), dict) else {}
    try:
        source_image = str(image.resolve().relative_to(folder.resolve())).replace("\\", "/")
    except (OSError, ValueError):
        source_image = image.name
    model_entry: dict[str, Any] = {
        "source": "auto",
        "format": "glb",
        "path": "model/model.glb",
        "backend": backend.name,
        "generated_at": _now(),
        "source_image": source_image,
        "note": (
            f"{backend.name} 自动生成（{info['size']} 字节，glTF {info['version']}.0）。"
            "注意：TripoSG 只生成形状，不带贴图、不带骨骼，不能直接用于身体重渲染。"
        ),
    }
    # Regenerating must not wipe what the user already registered.
    for key in ("bindings", "variants", "active_variant"):
        if old_model.get(key):
            model_entry[key] = old_model[key]
    meta["model"] = model_entry
    save_meta(folder, meta)
    # Point the assembly at the new body so it is selectable/used at once —
    # unless the character is currently wearing a costume variant, in which
    # case regenerating the main model must not change what they wear.
    # set_assembly resolves relative paths against the working directory,
    # so pass the absolute path; it stores the folder-relative form.
    active_variant = str(old_model.get("active_variant") or "").strip()
    wearing_variant = bool(
        active_variant
        and any(
            isinstance(v, dict)
            and v.get("name") == active_variant
            and (folder / str(v.get("path") or "")).is_file()
            for v in (old_model.get("variants") or [])
        )
    )
    if not wearing_variant:
        set_assembly(folder, body=str(dest))
    return {
        "character": name,
        "path": str(dest),
        "size": info["size"],
        "format": "glb",
        "source": "auto",
        "backend": backend.name,
        "generated_at": model_entry["generated_at"],
        "source_image": source_image,
        "assembly_body": "model/model.glb",
    }
