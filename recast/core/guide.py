"""Five-step guide: the workflow the buttons point at.

The main window used to show nine stage checkboxes and twelve action
buttons with no order and no pointer: nothing said "do this, then that".
The guide is the order:

    ① 导入视频/照片  →  ② 人物提取与聚类  →  ③ 人物审核确认  →  ④ 替换/生成/导出  →  ⑤ 声音与台词

Step 3 is the gate: step 4 (replace / generate) stays locked until the
review queue is empty. The lock is enforced in the UI (buttons greyed,
``_start`` refuses) and reported by :func:`replace_locked` so every entry
point uses the same rule. Step 5 is never locked: it follows step 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recast.core.constants import ST_DONE

GUIDE_STEPS: list[tuple[str, str]] = [
    ("import", "① 导入视频/照片"),
    ("extract", "② 人物提取与聚类"),
    ("review", "③ 人物审核确认"),
    ("replace", "④ 替换 / 生成 / 导出"),
    ("audio", "⑤ 声音与台词"),
]


@dataclass
class GuideStep:
    id: str
    number: int
    label: str
    state: str  # done | active | todo | locked
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "number": self.number,
            "label": self.label,
            "state": self.state,
            "detail": self.detail,
        }


def _audio_done(service: Any) -> bool:
    """Step 5 done? Only checks files exist — never creates anything.

    Done when a finished task has a dialogue mix, or a dialogue table
    exists with no dirty lines waiting for re-dub; on the import path
    (no dialogue table) every confirmed character having voice samples
    also counts. ``load_dialogue`` is only called when dialogue.json
    already exists.
    """
    from pathlib import Path

    try:
        done_tasks = service.store.list_rows("tasks", "status = ?", (ST_DONE,))
    except Exception:
        return False
    if not done_tasks:
        return False
    for task in done_tasks:
        workdir = task.get("workdir")
        if not workdir:
            continue
        wd = Path(workdir)
        if (wd / "dialogue_mix.wav").is_file():
            return True
        dlg = wd / "dialogue.json"
        if dlg.is_file():
            try:
                from recast.generate.dialogue import load_dialogue

                lines = load_dialogue(wd)
            except Exception:
                continue
            if lines and not any(line.get("dirty") for line in lines):
                return True
    try:
        characters = service.characters()
    except Exception:
        return False
    confirmed = [c for c in characters if c.get("confirmed")]
    if not confirmed:
        return False
    from recast.core.characters import character_folder, load_meta

    for ch in confirmed:
        name = ch.get("display_name") or ch.get("name") or ""
        folder = character_folder(service.settings.output_root, str(name))
        meta = load_meta(folder) or {}
        voice = meta.get("voice") if isinstance(meta.get("voice"), dict) else {}
        if not (voice.get("samples") or []):
            return False
    return True


def get_guide_state(service: Any) -> list[GuideStep]:
    """Current state of the five guide steps for this service."""
    dramas = service.store.list_dramas()
    characters = service.store.list_characters()
    queue = service.list_review_queue()
    done_tasks = service.store.list_rows("tasks", "status = ?", (ST_DONE,))

    step1_done = bool(dramas)
    step2_done = bool(characters)
    step3_done = step2_done and not queue
    step4_done = bool(done_tasks)
    step5_done = _audio_done(service)

    done = {
        "import": step1_done,
        "extract": step2_done,
        "review": step3_done,
        "replace": step4_done,
        "audio": step5_done,
    }
    details = {
        "import": f"已导入 {len(dramas)} 部剧目" if dramas else "还没导入视频：先选一个作品目录扫描。",
        "extract": (
            f"已提取 {len(characters)} 个人物" if characters else "还没有人物：跑「02 解析提取」或在人物库里手动添加。"
        ),
        "review": (
            "全部人物已确认，样本没有低分项。"
            if step3_done
            else f"待审核 {len(queue)} 项：先在人物库里处理完。"
            if step2_done
            else "等第 ② 步有了人物再审核。"
        ),
        "replace": (
            "已有完成的替换/生成任务。"
            if step4_done
            else "先完成第 ③ 步人物审核，再替换。"
            if not step3_done
            else "可以替换了：点「一键成片」或「按勾选阶段执行」。"
        ),
        "audio": (
            "声音与台词已完成：台词无待重配的改动，人物都有声音样本。"
            if step5_done
            else "可以做声音与台词了：先看分轨结果，再试听 / 调音色 / 改台词 / 重配音。"
            if step4_done
            else "等第 ④ 步完成后再做声音与台词。"
        ),
    }

    steps: list[GuideStep] = []
    active_set = False
    for i, (sid, label) in enumerate(GUIDE_STEPS, start=1):
        if done[sid]:
            state = "done"
        elif sid == "audio":
            # Step 5 is never locked: it follows step 4, and only becomes
            # the active step once step 4 is done.
            state = "active" if done["replace"] else "todo"
            if state == "active":
                active_set = True
        elif sid == "replace" and queue:
            # The only hard lock: review items outstanding. With no
            # characters at all there is nothing to review, so step 4 is
            # merely "todo" — replace itself will refuse with its own
            # "missing faces" message (check_replace_ready).
            state = "locked"
        elif not active_set:
            state = "active"
            active_set = True
        else:
            state = "todo"
        steps.append(GuideStep(id=sid, number=i, label=label, state=state, detail=details[sid]))
    return steps


def replace_locked(service: Any) -> tuple[bool, str]:
    """Whether replace/generate must stay locked, and why.

    Locked only when the review queue is not empty (unconfirmed clusters
    or low-score samples): replacing with unreviewed characters is how a
    wrong face gets baked into the video. With no characters at all the
    lock does not apply — replace refuses on its own via
    ``check_replace_ready`` ("missing faces"). The UI greys the buttons
    and ``_start`` skips the locked stages with this message; the rule
    lives here so every entry point agrees.
    """
    queue = service.list_review_queue()
    if queue:
        return True, f"人物审核还没完成：待审核 {len(queue)} 项，先在人物库里处理完再替换。"
    return False, ""
