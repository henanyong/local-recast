"""入口–阶段映射与替换授权关口的共用逻辑（R6 拆分第一步，P3）。

这里是两样东西的唯一来源，UI 层只 import、不再各抄一份：

* ``ENTRY_STAGES``——入口类型 → 适用的管线阶段；
* ``check_replace_auth``——替换授权关口的纯逻辑部分（N13/N18）。

已统一（N35，P3 第 3 波）：``main_window.py`` 与 ``queue_panel.py`` 都
改成 import 这里；任务历史重试逻辑里本来就没有这份拷贝。不要在这里
再抄一份，也不要在别处再抄一份。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# 入口类型 → 适用的管线阶段（PM 第 1 轮 U2 / 第 2 轮流程改动）：
# 导入视频只跑前 6 个，剧本生成只跑后 5 个；不相关的阶段不显示为待办或失败。
ENTRY_STAGES: dict[str, frozenset[str]] = {
    "import": frozenset({"scan", "extract", "library", "replace", "compose", "export"}),
    "script": frozenset({"template", "generate", "dub", "compose", "export"}),
}


def check_replace_auth(
    service: Any, drama_ids: list[str] | None = None
) -> dict[str, Any]:
    """替换授权关口的纯逻辑部分（N13/N18），不弹框、不写库。

    返回 ``{"authorized_count": int, "unauthorized": list[dict]}``：

    * ``authorized_count``——有参考脸且已授权的人物数（和
      ``unauthorized_replace_characters`` 同一来源：参考脸文件必须
      真实存在于磁盘上，只登记了路径但文件没了的不算）；
    * ``unauthorized``——有参考脸但未授权的人物（每项至少含 ``name``，
      其余字段 ``id`` / ``face_paths`` 原样透传 service 的结果）。

    N40：``drama_ids`` 给出时只算属于所选剧目的人物。映射见
    ``Store.character_in_scope``：人物行的 ``source_task_id`` 或
    各任务 ``selected_characters`` 快照链到的剧目；两处都链不到
    的人物算归属未知，照样算进来（fail-open：手动添加/旧库人物
    没有链路，漏算会让关口静默放行一次本该提示的替换）。只链到
    别的剧目的人物不再计入已授权数、也不再出现在未授权名单里。
    ``drama_ids`` 为空（None/[]）时保持老行为：按人物库全局算。
    service 读不到时抛异常，由调用方按“不挡路”处理（关口返回
    True / 计数返回 0），本函数不吞异常。
    """
    authorized = 0
    characters = service.store.list_characters()
    store = getattr(service, "store", None)
    in_scope = getattr(store, "character_in_scope", None)

    def _belongs(ch: dict[str, Any]) -> bool:
        if not drama_ids or not callable(in_scope):
            return True
        try:
            return bool(in_scope(ch, drama_ids))
        except Exception:
            return True

    for ch in characters:
        if not _belongs(ch):
            continue
        if not (ch.get("authorized") and ch.get("auth_type")):
            continue
        faces = [
            a
            for a in (ch.get("assets") or [])
            if a.get("kind") == "face"
            and a.get("path")
            and Path(str(a["path"])).is_file()
        ]
        if faces:
            authorized += 1
    try:
        unauthorized = service.unauthorized_replace_characters(drama_ids)
    except TypeError:
        # 旧签名 / 替身 service 不接受 drama_ids：拿全局名单再按
        # 同一规则本地过滤，行为与 service 侧收窄一致。
        unauthorized = [
            m
            for m in service.unauthorized_replace_characters()
            if _belongs(m)
        ]
    return {"authorized_count": authorized, "unauthorized": unauthorized}
