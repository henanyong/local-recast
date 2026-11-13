"""P4 batch, N40: auth gate scoped to the selected drama(s), not the
whole character library — and no more fake skip records.

Scenario (PM round 6, §四 N40): run drama A (everyone authorized),
while the library happens to hold an unauthorized character from
drama B. Before this fix the gate popped B's name for A, and clicking
[继续] permanently recorded B's name in A's ``auth_skips``.

Mapping used (``Store.character_in_scope``, no new tables):
``characters.source_task_id`` → task → drama (extraction origin), plus
``tasks.selected_characters`` snapshots. A character linked to no
drama at all (manual add / legacy row) is "unknown" and fails OPEN:
it still counts for the gate, so an unlinked person is never silently
skipped. Only a character explicitly linked to another drama is
scoped out.

Only synthetic data + stub dialogs: no real video, no TTS, no network.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def svc(tmp_path: Path):
    from recast.core.service import RecastService

    out = tmp_path / "out"
    works = tmp_path / "works"
    out.mkdir()
    works.mkdir()
    service = RecastService(tmp_path / "test.db")
    service.update_settings(output_root=str(out), works_root=str(works), first_run_done=True)
    return service


def _add_drama(svc, name: str, *, root_path: str | None = None) -> dict:
    return svc.store.insert(
        "dramas", origin_name=name, root_path=root_path or f"/tmp/{name}",
        episode_count=1, entry_type="import",
    )


def _add_task(svc, drama_id, *, video_path: str = "/tmp/ep1.mp4", **extra):
    fields: dict = {
        "drama_id": drama_id, "episode_name": "第1集",
        "video_path": video_path, "status": "待执行",
    }
    fields.update(extra)
    return svc.store.insert("tasks", **fields)


def _add_character(svc, name: str, *, authorized: int = 0,
                   auth_type: str | None = None, source_task_id: str | None = None):
    fields: dict = {
        "name": name, "display_name": name, "confirmed": 1,
        "authorized": authorized, "cluster_key": f"test:{name}",
    }
    if auth_type:
        fields["auth_type"] = auth_type
    if source_task_id is not None:
        fields["source_task_id"] = source_task_id
    return svc.store.insert("characters", **fields)


def _real_face(tmp_path: Path, name: str = "face.jpg") -> Path:
    from PIL import Image

    img = Image.new("L", (200, 200))
    px = img.load()
    for y in range(200):
        for x in range(200):
            px[x, y] = 255 if (x // 10 + y // 10) % 2 == 0 else 0
    path = tmp_path / name
    img.convert("RGB").save(path, "JPEG")
    return path


def _add_face_asset(svc, character_id: str, path: str):
    return svc.store.insert(
        "character_assets", character_id=character_id, kind="face", path=path,
    )


def _two_dramas(svc, tmp_path: Path):
    """A drama (all authorized) + B drama (one unauthorized person).

    Returns (drama_a, task_a, drama_b, task_b, face_path).
    """
    face_ok = _real_face(tmp_path)
    drama_a = _add_drama(svc, "A剧", root_path="/tmp/A剧")
    task_a = _add_task(svc, drama_a["id"], video_path="/tmp/A/ep1.mp4")
    drama_b = _add_drama(svc, "B剧", root_path="/tmp/B剧")
    task_b = _add_task(svc, drama_b["id"], video_path="/tmp/B/ep1.mp4")
    ok_a = _add_character(svc, "温念", authorized=1, auth_type="自有拍摄",
                          source_task_id=task_a["id"])
    _add_face_asset(svc, ok_a["id"], str(face_ok))
    bad_b = _add_character(svc, "温雪", authorized=0,
                           source_task_id=task_b["id"])
    _add_face_asset(svc, bad_b["id"], str(face_ok))
    return drama_a, task_a, drama_b, task_b, face_ok


class _FakeButton:
    def __init__(self, text):
        self.text = text
        self._enabled = True

    def setEnabled(self, enabled):
        self._enabled = bool(enabled)

    def isEnabled(self):
        return self._enabled


class _FakeBox:
    """Minimal QMessageBox stand-in for the shared auth gate."""

    from PySide6.QtWidgets import QMessageBox as _RealBox

    ButtonRole = _RealBox.ButtonRole
    shown: list = []
    click: str | None = None

    def __init__(self, parent=None):
        self.parent = parent
        self._buttons: dict = {}
        _FakeBox.shown.append(self)

    def setWindowTitle(self, title):
        self.title = title

    def setText(self, text):
        self.text = text

    def addButton(self, text, role):
        btn = _FakeButton(text)
        self._buttons[text] = btn
        return btn

    def exec(self):
        return 1

    def clickedButton(self):
        if type(self).click is None:
            return None
        return self._buttons.get(type(self).click)


@pytest.fixture(autouse=True)
def _reset_fakebox():
    _FakeBox.shown.clear()
    _FakeBox.click = None
    yield
    _FakeBox.shown.clear()
    _FakeBox.click = None


# ----- 关口：跑 A 剧时 B 剧未授权人物不再弹 -----


def test_gate_for_drama_a_ignores_drama_b_unauthorized(svc, tmp_path: Path):
    """正向：A 全已授权、B 有未授权人物——跑 A 时关口名单为空。"""
    from recast.core.entry_shared import check_replace_auth

    drama_a, _ta, drama_b, _tb, _face = _two_dramas(svc, tmp_path)
    result_a = check_replace_auth(svc, [drama_a["id"]])
    assert result_a["unauthorized"] == []
    assert result_a["authorized_count"] == 1
    # 对照：跑 B 时仍弹 B 的人；全局仍弹 B 的人。
    result_b = check_replace_auth(svc, [drama_b["id"]])
    assert [m["name"] for m in result_b["unauthorized"]] == ["温雪"]
    assert result_b["authorized_count"] == 0
    result_all = check_replace_auth(svc, None)
    assert [m["name"] for m in result_all["unauthorized"]] == ["温雪"]


def test_gate_for_drama_a_own_unauthorized_still_blocks(svc, tmp_path: Path):
    """正向：A 自己的未授权人物仍弹关口（收窄不是放行）。"""
    from recast.core.entry_shared import check_replace_auth

    face_ok = _real_face(tmp_path)
    drama_a = _add_drama(svc, "A剧", root_path="/tmp/A剧")
    task_a = _add_task(svc, drama_a["id"], video_path="/tmp/A/ep1.mp4")
    ok = _add_character(svc, "温念", authorized=1, auth_type="自有拍摄",
                        source_task_id=task_a["id"])
    _add_face_asset(svc, ok["id"], str(face_ok))
    bad = _add_character(svc, "陆寒峥", authorized=0,
                         source_task_id=task_a["id"])
    _add_face_asset(svc, bad["id"], str(face_ok))
    result = check_replace_auth(svc, [drama_a["id"]])
    assert [m["name"] for m in result["unauthorized"]] == ["陆寒峥"]
    assert result["authorized_count"] == 1


def test_gate_unlinked_character_fails_open(svc, tmp_path: Path):
    """边界：没链路的人物（手动添加、无 source_task_id）照样弹关口，
    不因“归属未知”被静默漏掉；全局与按剧查的结果一致。"""
    from recast.core.entry_shared import check_replace_auth

    face_ok = _real_face(tmp_path)
    drama_a = _add_drama(svc, "A剧", root_path="/tmp/A剧")
    _add_task(svc, drama_a["id"], video_path="/tmp/A/ep1.mp4")
    ok = _add_character(svc, "温念", authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ok["id"], str(face_ok))
    bad = _add_character(svc, "路人甲", authorized=0)  # 无 source_task_id
    _add_face_asset(svc, bad["id"], str(face_ok))
    result = check_replace_auth(svc, [drama_a["id"]])
    assert [m["name"] for m in result["unauthorized"]] == ["路人甲"]
    assert result == check_replace_auth(svc, None)


def test_gate_authorized_count_scoped_to_selected_drama(svc, tmp_path: Path):
    """边界：已授权数也按剧收窄——只有 B 有已授权人物时，跑 A 的
    authorized_count 是 0（[继续] 应灰掉），跑 B 时是 1。"""
    from recast.core.entry_shared import check_replace_auth

    face_ok = _real_face(tmp_path)
    drama_a = _add_drama(svc, "A剧", root_path="/tmp/A剧")
    task_a = _add_task(svc, drama_a["id"], video_path="/tmp/A/ep1.mp4")
    drama_b = _add_drama(svc, "B剧", root_path="/tmp/B剧")
    task_b = _add_task(svc, drama_b["id"], video_path="/tmp/B/ep1.mp4")
    ok_b = _add_character(svc, "温雪", authorized=1, auth_type="自有拍摄",
                          source_task_id=task_b["id"])
    _add_face_asset(svc, ok_b["id"], str(face_ok))
    assert check_replace_auth(svc, [drama_a["id"]])["authorized_count"] == 0
    assert check_replace_auth(svc, [drama_b["id"]])["authorized_count"] == 1
    assert check_replace_auth(svc, [drama_a["id"], drama_b["id"]])["authorized_count"] == 1
    # 空 drama_ids 保持老行为：全局计数。
    assert check_replace_auth(svc, [])["authorized_count"] == 1


def test_gate_character_linked_via_selected_characters_snapshot(svc, tmp_path: Path):
    """边界：第二映射来源——人物行没 source_task_id，但名字落在 B 剧
    分集任务的 selected_characters 快照里，跑 A 时同样不弹。"""
    from recast.core.entry_shared import check_replace_auth

    face_ok = _real_face(tmp_path)
    drama_a = _add_drama(svc, "A剧", root_path="/tmp/A剧")
    _add_task(svc, drama_a["id"], video_path="/tmp/A/ep1.mp4")
    drama_b = _add_drama(svc, "B剧", root_path="/tmp/B剧")
    _add_task(svc, drama_b["id"], video_path="/tmp/B/ep1.mp4",
              selected_characters=["温雪"])
    ok = _add_character(svc, "温念", authorized=1, auth_type="自有拍摄")
    _add_face_asset(svc, ok["id"], str(face_ok))
    bad = _add_character(svc, "温雪", authorized=0)  # 无 source_task_id
    _add_face_asset(svc, bad["id"], str(face_ok))
    assert check_replace_auth(svc, [drama_a["id"]])["unauthorized"] == []
    assert [m["name"] for m in check_replace_auth(svc, [drama_b["id"]])["unauthorized"]] == ["温雪"]


def test_gate_character_in_both_dramas_counts_for_both(svc, tmp_path: Path):
    """边界：同一人名在两部剧都有链路（A 是提取来源、B 的快照里有
    他）——跑哪部都算在内，不会因“也属于别剧”被排除。"""
    from recast.core.entry_shared import check_replace_auth

    face_ok = _real_face(tmp_path)
    drama_a = _add_drama(svc, "A剧", root_path="/tmp/A剧")
    task_a = _add_task(svc, drama_a["id"], video_path="/tmp/A/ep1.mp4")
    drama_b = _add_drama(svc, "B剧", root_path="/tmp/B剧")
    _add_task(svc, drama_b["id"], video_path="/tmp/B/ep1.mp4",
              selected_characters=["温雪"])
    ok = _add_character(svc, "温念", authorized=1, auth_type="自有拍摄",
                        source_task_id=task_a["id"])
    _add_face_asset(svc, ok["id"], str(face_ok))
    bad = _add_character(svc, "温雪", authorized=0,
                         source_task_id=task_a["id"])
    _add_face_asset(svc, bad["id"], str(face_ok))
    assert [m["name"] for m in check_replace_auth(svc, [drama_a["id"]])["unauthorized"]] == ["温雪"]
    assert [m["name"] for m in check_replace_auth(svc, [drama_b["id"]])["unauthorized"]] == ["温雪"]


# ----- service.unauthorized_replace_characters 本身也收窄 -----


def test_service_unauthorized_scoped_and_global(svc, tmp_path: Path):
    """正向 + 边界：带 drama_ids 只返回本剧未授权；不带参数保持
    全局老行为（旧调用方不受影响）。"""
    drama_a, _ta, drama_b, _tb, _face = _two_dramas(svc, tmp_path)
    assert svc.unauthorized_replace_characters([drama_a["id"]]) == []
    assert [m["name"] for m in svc.unauthorized_replace_characters([drama_b["id"]])] == ["温雪"]
    assert [m["name"] for m in svc.unauthorized_replace_characters()] == ["温雪"]
    assert [m["name"] for m in svc.unauthorized_replace_characters([])] == ["温雪"]


def test_service_unauthorized_unlinked_still_listed(svc, tmp_path: Path):
    """边界：没链路的未授权人物在按剧查时仍列出（fail-open）。"""
    face_ok = _real_face(tmp_path)
    drama_a = _add_drama(svc, "A剧", root_path="/tmp/A剧")
    _add_task(svc, drama_a["id"], video_path="/tmp/A/ep1.mp4")
    bad = _add_character(svc, "路人甲", authorized=0)
    _add_face_asset(svc, bad["id"], str(face_ok))
    assert [m["name"] for m in svc.unauthorized_replace_characters([drama_a["id"]])] == ["路人甲"]


# ----- 记账口：只记本剧人物 -----


def test_record_auth_skips_writes_only_in_drama_names(svc, tmp_path: Path):
    """正向：missing 里混着 A、B 两剧人名，对 A 记账时只写 A 的人，
    B 的人名不进 A 的 auth_skips。"""
    from recast.desktop.auth_gate_dialog import record_auth_skips

    drama_a, task_a, drama_b, task_b, _face = _two_dramas(svc, tmp_path)
    # A 自己的未授权人物（链 A），与 B 的人一起传进来。
    face_ok = _real_face(tmp_path, "face2.jpg")
    bad_a = _add_character(svc, "陆寒峥", authorized=0,
                           source_task_id=task_a["id"])
    _add_face_asset(svc, bad_a["id"], str(face_ok))
    record_auth_skips(
        svc,
        [{"name": "陆寒峥"}, {"name": "温雪"}],
        [drama_a["id"]],
    )
    assert svc.store.skipped_character_names(drama_id=drama_a["id"]) == ["陆寒峥"]
    assert svc.store.list_auth_skips(drama_id=drama_a["id"])[0]["character_name"] == "陆寒峥"
    # B 那边同理：只写 B 的人。
    record_auth_skips(
        svc,
        [{"name": "陆寒峥"}, {"name": "温雪"}],
        [drama_b["id"]],
    )
    assert svc.store.skipped_character_names(drama_id=drama_b["id"]) == ["温雪"]


def test_record_auth_skips_multi_drama_no_cross_write(svc, tmp_path: Path):
    """边界：一次跑 A、B 两部剧，missing 全量传入——每个人名只写进
    自己所属的剧目，不跨剧串记；无链路人名两边都记（fail-open）。"""
    from recast.desktop.auth_gate_dialog import record_auth_skips

    drama_a, _ta, drama_b, _tb, _face = _two_dramas(svc, tmp_path)
    record_auth_skips(
        svc,
        [{"name": "温雪"}, {"name": "路人甲"}],
        [drama_a["id"], drama_b["id"]],
    )
    assert svc.store.skipped_character_names(drama_id=drama_a["id"]) == ["路人甲"]
    assert svc.store.skipped_character_names(drama_id=drama_b["id"]) == ["温雪", "路人甲"]


def test_record_auth_skips_no_drama_writes_nothing(svc, tmp_path: Path):
    """边界：drama_ids 为空时不写（老行为：无归属剧目可记）。"""
    from recast.desktop.auth_gate_dialog import record_auth_skips

    drama_a, _ta, _db, _tb, _face = _two_dramas(svc, tmp_path)
    record_auth_skips(svc, [{"name": "温雪"}], None)
    record_auth_skips(svc, [{"name": "温雪"}], [])
    assert svc.store.list_auth_skips(drama_id=drama_a["id"]) == []


# ----- 共用关口对话框：端到端 -----


def test_auth_gate_ok_for_drama_a_no_dialog(svc, tmp_path: Path, monkeypatch):
    """正向：跑 A（全已授权）时 B 的未授权人物不再弹框，直接放行，
    且不产生任何跳过记录。"""
    from recast.desktop.auth_gate_dialog import auth_gate_ok

    drama_a, _ta, _db, _tb, _face = _two_dramas(svc, tmp_path)
    monkeypatch.setattr("recast.desktop.auth_gate_dialog.QMessageBox", _FakeBox)
    assert auth_gate_ok(None, svc, [drama_a["id"]]) is True
    assert _FakeBox.shown == []
    assert svc.store.list_auth_skips(drama_id=drama_a["id"]) == []


def test_auth_gate_ok_for_drama_a_continue_records_only_own(svc, tmp_path: Path, monkeypatch):
    """正向：A 有自己的未授权人物时弹框只列 A 的人（不含 B 的名），
    点 [继续] 后 A 的记录里只有 A 的人。"""
    from recast.desktop.auth_gate_dialog import auth_gate_ok

    drama_a, task_a, _db, _tb, _face = _two_dramas(svc, tmp_path)
    face_ok = _real_face(tmp_path, "face2.jpg")
    bad_a = _add_character(svc, "陆寒峥", authorized=0,
                           source_task_id=task_a["id"])
    _add_face_asset(svc, bad_a["id"], str(face_ok))
    _FakeBox.click = "继续，只替换已授权人物"
    monkeypatch.setattr("recast.desktop.auth_gate_dialog.QMessageBox", _FakeBox)
    assert auth_gate_ok(None, svc, [drama_a["id"]]) is True
    assert len(_FakeBox.shown) == 1
    assert "陆寒峥" in _FakeBox.shown[0].text
    assert "温雪" not in _FakeBox.shown[0].text
    assert svc.store.skipped_character_names(drama_id=drama_a["id"]) == ["陆寒峥"]


# ----- 读口：已产生的假记录被标注/过滤，库里原行还在 -----


def test_read_ports_annotate_and_filter_fake_record(svc, tmp_path: Path):
    """正向：P3 时代已把 B 的人名写进 A 的 auth_skips（直接写库模拟
    假记录）——查 A 时该行还在（不删数据），但带 scope_note 标注，
    且 skipped_character_names 里看不到它。"""
    drama_a, _ta, drama_b, _tb, _face = _two_dramas(svc, tmp_path)
    svc.store.record_auth_skip(drama_a["id"], "温雪", reason="未授权")
    svc.store.record_auth_skip(drama_a["id"], "陆寒峥", reason="未授权")  # 无链路名
    rows = svc.store.list_auth_skips(drama_id=drama_a["id"])
    assert len(rows) == 2, "假记录原行必须还在库里，不静默删除"
    by_name = {r["character_name"]: r for r in rows}
    assert by_name["温雪"].get("scope_note") == "全局未授权，非本剧人物"
    assert "scope_note" not in by_name["陆寒峥"]
    assert svc.store.skipped_character_names(drama_id=drama_a["id"]) == ["陆寒峥"]
    # 对照：查 B 时同一人名是本剧的，不标注、不过滤。
    rows_b = svc.store.list_auth_skips(drama_id=drama_b["id"])
    assert rows_b == []
    svc.store.record_auth_skip(drama_b["id"], "温雪", reason="未授权")
    assert "scope_note" not in svc.store.list_auth_skips(drama_id=drama_b["id"])[0]
    assert svc.store.skipped_character_names(drama_id=drama_b["id"]) == ["温雪"]


def test_read_ports_task_scoped_filter(svc, tmp_path: Path):
    """边界：只按 task 查时也能由任务反查剧目、同样标注/过滤。"""
    drama_a, task_a, _db, _tb, _face = _two_dramas(svc, tmp_path)
    svc.store.record_auth_skip(drama_a["id"], "温雪", task_id=task_a["id"], reason="未授权")
    rows = svc.store.list_auth_skips(task_id=task_a["id"])
    assert len(rows) == 1
    assert rows[0].get("scope_note") == "全局未授权，非本剧人物"
    assert svc.store.skipped_character_names(task_id=task_a["id"]) == []


def test_read_ports_unscoped_returns_everything(svc, tmp_path: Path):
    """边界：不给剧目/task 时读口不过滤、不标注（无法判定归属）。"""
    drama_a, _ta, _db, _tb, _face = _two_dramas(svc, tmp_path)
    svc.store.record_auth_skip(drama_a["id"], "温雪", reason="未授权")
    rows = svc.store.list_auth_skips()
    assert len(rows) == 1
    assert "scope_note" not in rows[0]
    assert svc.store.skipped_character_names() == ["温雪"]


def test_read_ports_unknown_name_kept(svc, tmp_path: Path):
    """边界：库里没链路的人名记录，查本剧时保留、不标注（fail-open，
    与 P3 旧测试的直接记账行为一致）。"""
    drama_a, _ta, _db, _tb, _face = _two_dramas(svc, tmp_path)
    svc.store.record_auth_skip(drama_a["id"], "温雪", reason="未授权")
    svc.store.record_auth_skip(drama_a["id"], "陈秋薇", reason="未授权")
    assert svc.store.skipped_character_names(drama_id=drama_a["id"]) == ["陈秋薇"]
    rows = svc.store.list_auth_skips(drama_id=drama_a["id"])
    by_name = {r["character_name"]: r for r in rows}
    assert "scope_note" not in by_name["陈秋薇"]
    assert by_name["温雪"].get("scope_note") == "全局未授权，非本剧人物"


# ----- 映射 helper 本身 -----


def test_character_in_scope_mapping(svc, tmp_path: Path):
    """映射 helper：source_task_id 链剧、快照链剧、无链路放行、
    空 drama_ids 不收窄。"""
    drama_a, task_a, drama_b, task_b, _face = _two_dramas(svc, tmp_path)
    store = svc.store
    assert store.linked_drama_ids_for_character_name("温雪") == {drama_b["id"]}
    assert store.linked_drama_ids_for_character_name("温念") == {drama_a["id"]}
    assert store.linked_drama_ids_for_character_name("不存在的人") == set()
    assert store.character_in_scope("温雪", [drama_a["id"]]) is False
    assert store.character_in_scope("温雪", [drama_b["id"]]) is True
    assert store.character_in_scope("温雪", [drama_a["id"], drama_b["id"]]) is True
    assert store.character_in_scope("不存在的人", [drama_a["id"]]) is True
    assert store.character_in_scope("温雪", None) is True
    assert store.character_in_scope("温雪", []) is True
    # 人物行：改名后仍靠 source_task_id 链到原剧。
    ch_b = store.list_rows("characters", "name = ?", ("温雪",))[0]
    assert store.linked_drama_ids_for_character(ch_b) == {drama_b["id"]}
    renamed = dict(ch_b, display_name="小雪", name="小雪")
    assert store.linked_drama_ids_for_character(renamed) == {drama_b["id"]}
