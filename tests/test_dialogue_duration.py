from recast.templates.duration import char_count, check_dialogue_duration, check_shots
from recast.generate.stitch import stitch_and_dub
from pathlib import Path
import pytest


def test_char_count_ignores_punct():
    assert char_count("快起来！别装死！") == 6
    assert char_count("Hello, 世界") == 7


def test_overflow_does_not_recommend_speed():
    # 22 chars into a 3-second slot at 4 cps → overflow
    text = "你心肠歹毒妄想陆寒铮陆家看不上你这种村姑"
    issue = check_dialogue_duration(text, 3.0, shot_code="F4-L3")
    assert issue is not None
    assert "不要加速" in issue.message or "不" in issue.message and "加速" in issue.message
    assert "加速音频" in issue.message


def test_fits_when_short_enough():
    assert check_dialogue_duration("快起来！别装死！", 5.0, shot_code="F2-L1") is None


def test_stitch_raises_and_never_sets_atempo(tmp_path: Path):
    shots = [
        {
            "shot_code": "F4-L3",
            "dialogue": "你心肠歹毒妄想陆寒铮陆家看不上你这种村姑再加几个字",
            "start_time": 0,
            "end_time": 3,
            "duration_sec": 3,
            "output_path": str(tmp_path / "missing.mp4"),
        }
    ]
    with pytest.raises(ValueError) as exc:
        stitch_and_dub(shots, tmp_path, precheck_dialogue=True)
    assert "加速" in str(exc.value)
    assert "atempo" not in str(exc.value).lower()
