"""B0-1: identity model buffalo_s -> buffalo_l (SCRFD-10GF + ArcFace R50).

Only the config wiring is tested here: this VM has no GPU and no real
InsightFace weights, so FaceAnalysis is always faked. Real recognition
behaviour (first-download ~326MB, detection/recall on side/blurry/small
faces) must be verified on Daniel's Windows machine.
"""

import json
import sys
import types
from pathlib import Path

import pytest

from recast.config import PROJECT_ROOT, IdentityConfig, RecastConfig
from recast.ffmpeg import VideoInfo
from recast.quality import face_review
from recast.stages import identity as identity_stage


class _FakeFaceAnalysis:
    """Records how FaceAnalysis was constructed; never loads weights."""

    instances: list["_FakeFaceAnalysis"] = []

    def __init__(self, name, providers=None):
        self.name = name
        self.providers = providers
        self.prepared = None
        type(self).instances.append(self)

    def prepare(self, ctx_id=-1, det_size=(640, 640)):
        self.prepared = {"ctx_id": ctx_id, "det_size": det_size}

    def get(self, image):
        return []


@pytest.fixture(autouse=True)
def _reset_face_app_cache():
    face_review.try_embed_sample.__dict__.pop("_app", None)
    face_review.try_embed_sample.__dict__.pop("_app_model", None)
    _FakeFaceAnalysis.instances.clear()
    yield
    face_review.try_embed_sample.__dict__.pop("_app", None)
    face_review.try_embed_sample.__dict__.pop("_app_model", None)
    _FakeFaceAnalysis.instances.clear()


def _install_fake_insightface(monkeypatch):
    parent = types.ModuleType("insightface")
    parent.__path__ = []
    child = types.ModuleType("insightface.app")
    child.FaceAnalysis = _FakeFaceAnalysis
    monkeypatch.setitem(sys.modules, "insightface", parent)
    monkeypatch.setitem(sys.modules, "insightface.app", child)


def _install_fake_cv2(monkeypatch):
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.imread = lambda path: None
    fake_cv2.CAP_PROP_POS_FRAMES = 1

    class _Cap:
        def set(self, *args):
            return True

        def read(self):
            return False, None

        def release(self):
            pass

    fake_cv2.VideoCapture = lambda path: _Cap()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)


def test_identity_config_default_is_buffalo_l():
    assert IdentityConfig().model == "buffalo_l"
    assert RecastConfig().identity.model == "buffalo_l"


def test_yaml_presets_use_buffalo_l():
    for name in ("4060ti-8g.yaml", "4060ti-16g.yaml"):
        cfg = RecastConfig.from_yaml(PROJECT_ROOT / "configs" / name)
        assert cfg.identity.model == "buffalo_l", name


def test_build_face_app_receives_buffalo_l(monkeypatch):
    _install_fake_insightface(monkeypatch)
    cfg = RecastConfig.from_yaml(PROJECT_ROOT / "configs" / "4060ti-8g.yaml")
    app = identity_stage.build_face_app(cfg.identity.model, cfg.identity.det_size, "cpu")
    assert isinstance(app, _FakeFaceAnalysis)
    assert app.name == "buffalo_l"
    assert app.prepared == {"ctx_id": -1, "det_size": (512, 512)}


def test_explicit_buffalo_s_config_still_loads(monkeypatch):
    """Old value is not rejected; only the default changed."""
    _install_fake_insightface(monkeypatch)
    cfg = RecastConfig(identity=IdentityConfig(model="buffalo_s"))
    app = identity_stage.build_face_app(cfg.identity.model, cfg.identity.det_size, "cpu")
    assert app.name == "buffalo_s"


@pytest.mark.parametrize("model", ["buffalo_l", "buffalo_s"])
def test_identity_run_payload_carries_configured_model(monkeypatch, tmp_path, model):
    """run() passes cfg.identity.model to build_face_app and into identity.json."""
    seen: dict[str, str] = {}

    def _fake_build(model_name, det_size, device):
        seen["model"] = model_name
        return _FakeFaceAnalysis(model_name)

    class _Cap:
        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(identity_stage, "build_face_app", _fake_build)
    monkeypatch.setattr(identity_stage, "open_capture", lambda video: _Cap())
    cfg = RecastConfig(identity=IdentityConfig(model=model))
    info = VideoInfo(
        path="x.mp4", width=1280, height=720, fps=25.0,
        duration=1.0, frame_count=25, has_audio=False,
    )
    payload = identity_stage.run("x.mp4", info, {"tracks": []}, tmp_path, cfg)
    assert seen["model"] == model
    assert payload["model"] == model
    on_disk = json.loads((tmp_path / "identity.json").read_text(encoding="utf-8"))
    assert on_disk["model"] == model


def test_face_review_embedder_defaults_to_buffalo_l(monkeypatch, tmp_path):
    _install_fake_insightface(monkeypatch)
    _install_fake_cv2(monkeypatch)
    ref = tmp_path / "ref.jpg"
    ref.write_bytes(b"fake")
    video = tmp_path / "replaced.mp4"
    video.write_bytes(b"fake")
    result = face_review.try_embed_sample(
        {"reference": str(ref), "frame_idx": 0}, video
    )
    assert result is None  # fake cv2 returns no image; only the wiring is checked
    assert len(_FakeFaceAnalysis.instances) == 1
    assert _FakeFaceAnalysis.instances[0].name == "buffalo_l"


def test_face_review_embedder_follows_explicit_model(monkeypatch, tmp_path):
    _install_fake_insightface(monkeypatch)
    _install_fake_cv2(monkeypatch)
    ref = tmp_path / "ref.jpg"
    ref.write_bytes(b"fake")
    video = tmp_path / "replaced.mp4"
    video.write_bytes(b"fake")
    face_review.try_embed_sample({"reference": str(ref), "frame_idx": 0}, video)
    assert _FakeFaceAnalysis.instances[-1].name == "buffalo_l"
    # Same process, explicit old model: the cached app must be rebuilt, not reused.
    face_review.try_embed_sample(
        {"reference": str(ref), "frame_idx": 0}, video, model="buffalo_s"
    )
    assert _FakeFaceAnalysis.instances[-1].name == "buffalo_s"
    assert len(_FakeFaceAnalysis.instances) == 2


def test_registry_lists_buffalo_l():
    from recast.models.registry import REGISTRY

    keys = [spec.key for spec in REGISTRY]
    assert "buffalo_l" in keys
    assert "buffalo_s" not in keys
    spec = next(s for s in REGISTRY if s.key == "buffalo_l")
    assert spec.kind == "face"
    assert "library" in spec.required_for and "replace" in spec.required_for
