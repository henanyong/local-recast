from recast.templates.validate import validate_shot, validate_pack, primary_actions_of


def test_one_action_rejected():
    shot = {
        "shot_code": "F2-L1",
        "primary_actions": ["踹门", "再扇一耳光"],
        "people": ["陈秋微"],
        "ref_assets": [{"file": "02-陈秋微.jpg", "character": "陈秋微"}],
    }
    issues = validate_shot(shot, pack_root=None)
    assert any(i.code == "one_action" and i.is_error for i in issues)


def test_one_action_list_field():
    shot = {"shot_code": "F1-L1", "primary_action": ["推", "打"], "people": []}
    assert len(primary_actions_of(shot)) == 2
    assert any(i.code == "one_action" for i in validate_shot(shot))


def test_single_action_ok():
    shot = {
        "shot_code": "F2-L2",
        "primary_action": "陈秋微双手掐住温念胳膊",
        "people": ["陈秋微", "温念"],
        "ref_assets": [
            {"file": "02-陈秋微.jpg", "character": "陈秋微"},
            {"file": "01-温念.jpg", "character": "温念"},
        ],
        "is_fight": True,
    }
    assert not [i for i in validate_shot(shot) if i.is_error]


def test_missing_refs_are_named():
    shot = {
        "shot_code": "F3-L2",
        "primary_action": "温雪猛推温念",
        "people": ["温雪", "温念"],
        "ref_assets": [{"file": "03-温雪.jpg", "character": "温雪"}],
    }
    issues = [i for i in validate_shot(shot) if i.code == "missing_refs"]
    assert issues
    assert "温念.png" in issues[0].missing_files
    assert "缺少 温念.png" in issues[0].message
    assert "资料不全" not in issues[0].message


def test_fight_two_person_cap():
    shot = {
        "shot_code": "F9-L1",
        "primary_action": "三人互殴",
        "people": ["温念", "温雪", "陈秋微"],
        "ref_assets": [
            {"file": "01-温念.jpg", "character": "温念"},
            {"file": "03-温雪.jpg", "character": "温雪"},
            {"file": "02-陈秋微.jpg", "character": "陈秋微"},
        ],
        "is_fight": True,
    }
    issues = [i for i in validate_shot(shot, fight_max_two=True) if i.code == "fight_two_person"]
    assert issues and issues[0].is_error
    # optional: can be turned off
    assert not [i for i in validate_shot(shot, fight_max_two=False) if i.code == "fight_two_person"]


def test_pack_style_lock_lives_on_script():
    pack = {
        "script": {"style_lock": "土黄灰调", "fight_max_two": True},
        "shots": [
            {
                "shot_code": "F1-L1",
                "primary_action": "远景推进",
                "people": [],
                "ref_assets": [],
            }
        ],
    }
    assert not [i for i in validate_pack(pack) if i.is_error]
