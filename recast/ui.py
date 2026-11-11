"""Gradio UI with Chinese labels: upload video, optional face, progress, preview, download."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

from recast.config import DEFAULT_CONFIG, PROJECT_ROOT, resolve_config_path
from recast.pipeline import process, setup_logging

log = logging.getLogger("recast")


def _config_choices() -> list[str]:
    configs = sorted((PROJECT_ROOT / "configs").glob("*.yaml"))
    items = [str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p) for p in configs]
    default = str(DEFAULT_CONFIG.relative_to(PROJECT_ROOT)) if DEFAULT_CONFIG.exists() else "configs/4060ti-8g.yaml"
    if default not in items:
        items.insert(0, default)
    else:
        items.remove(default)
        items.insert(0, default)
    return items


def _as_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("path") or value.get("name") or value.get("video")
    return str(value)


def _run_job(
    video: str | None,
    face: str | None,
    config_name: str,
    track_id: float | int | None,
    names: str,
    lang: str,
    skip_replace: bool,
    burn_subs: bool,
    progress: Any = None,
):
    video = _as_path(video)
    face = _as_path(face)
    if not video:
        raise ValueError("请先上传或选择一支输入视频。")

    logs: list[str] = []

    def on_progress(stage: str, message: str, fraction: float) -> None:
        line = f"{int(fraction * 100):3d}%  [{stage}] {message}"
        logs.append(line)
        if progress is not None:
            try:
                progress(fraction, desc=f"{stage}: {message}")
            except Exception:
                pass

    tid = None
    if track_id is not None and str(track_id).strip() != "":
        try:
            tid = int(track_id)
        except (TypeError, ValueError):
            tid = None
        if tid is not None and tid <= 0:
            tid = None

    setup_logging()
    try:
        summary = process(
            video,
            face=face or None,
            out=None,
            config=config_name or "configs/4060ti-8g.yaml",
            track_id=tid,
            names=names.strip() or None,
            skip_replace=bool(skip_replace),
            lang=lang or "auto",
            burn_subs=bool(burn_subs),
            progress=on_progress,
        )
    except Exception as exc:
        logs.append(f"失败: {exc}")
        logs.append(traceback.format_exc())
        return "\n".join(logs), None, None, f"处理失败: {exc}"

    persons = summary.get("persons") or []
    person_lines = []
    for p in persons:
        person_lines.append(
            f"- {p.get('name')}  (自动名 {p.get('auto_name')}, 轨迹 {p.get('track_ids')}, 人脸样本 {p.get('face_samples')})"
        )
    status = (
        f"完成。\n成片: {summary.get('output')}\n工作目录: {summary.get('workdir')}\n"
        f"人物:\n" + ("\n".join(person_lines) if person_lines else "- （未检测到人）")
    )
    output = summary.get("output")
    preview = summary.get("skeleton") or output
    return "\n".join(logs), preview, output, status


def build_app(default_config: str | None = None):
    import gradio as gr

    choices = _config_choices()
    initial = default_config or choices[0]
    try:
        initial = str(resolve_config_path(initial).relative_to(PROJECT_ROOT))
    except Exception:
        initial = choices[0] if choices else "configs/4060ti-8g.yaml"

    with gr.Blocks(title="local-recast 本地视频再铸") as demo:
        gr.Markdown(
            """
# local-recast  本地视频再铸
面向 **i5-12600KF + RTX 4060 Ti** 的本地流水线：字幕、人物跟踪、命名、姿态关键点，以及默认的 InsightFace 换脸。

v1 **不会**运行 Wan2.2 / AniCrafter（8GB 显存放不下完整视频扩散替换）。完整 SMPL-X 网格导出也需要额外许可文件，默认只写出关键点钩子。
            """.strip()
        )
        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="输入视频", sources=["upload"])
                face_in = gr.Image(label="参考脸 / 角色图（可选，不提供则只提取不替换）", type="filepath")
                config = gr.Dropdown(choices=choices, value=initial, label="配置", allow_custom_value=True)
                track_id = gr.Number(label="目标轨迹 ID（留空则自动选出现最多的人）", precision=0, value=None)
                names = gr.Textbox(
                    label="人名映射",
                    placeholder="1=张三,2=李四  或填写 examples/names.yaml 路径",
                    lines=2,
                )
                lang = gr.Dropdown(
                    label="字幕语言",
                    choices=["auto", "zh", "en", "ja", "ko"],
                    value="auto",
                )
                skip_replace = gr.Checkbox(label="跳过替换（只提取字幕 / 轨迹 / 姿态）", value=False)
                burn_subs = gr.Checkbox(label="把字幕烧进成片（旁路 SRT 始终会写）", value=False)
                run_btn = gr.Button("开始处理", variant="primary")
            with gr.Column(scale=1):
                status = gr.Textbox(label="状态", lines=8)
                log_box = gr.Textbox(label="日志与进度", lines=16)
                preview = gr.Video(label="预览（骨架叠加，没有则显示成片）")
                download = gr.File(label="下载成片")

        run_btn.click(
            _run_job,
            inputs=[video_in, face_in, config, track_id, names, lang, skip_replace, burn_subs],
            outputs=[log_box, preview, download, status],
        )
    return demo


def launch(
    config: str | None = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        warning = (
            f"界面正监听 {host}:{port}。该服务没有鉴权，请只在受信任的网络上使用，或改为监听 127.0.0.1。"
        )
        log.warning(warning)
        print(warning, flush=True)
    demo = build_app(default_config=config)
    demo.queue()
    demo.launch(server_name=host, server_port=port, share=share, inbrowser=False)
