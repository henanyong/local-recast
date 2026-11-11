"""Shared path names, extensions, and V0.1 vocabulary."""

from __future__ import annotations

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".flv", ".mpg", ".mpeg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}
ROSTER_EXTS = {".yaml", ".yml", ".json", ".txt"}

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "recast-runs",
    ".insightface",
    "workbench-assets",
    "models",
    "outputs",
}

# Possible subfolder names for reference stills, videos, and roster files.
REF_DIR_NAMES = ("参考图", "参考脸", "人物参考", "faces", "refs", "cast", "images")
VIDEO_DIR_NAMES = ("视频", "视频文件", "episodes", "videos", "clips", "成片源")
ROSTER_FILENAMES = (
    "名单.yaml",
    "名单.yml",
    "names.yaml",
    "names.yml",
    "人物.yaml",
    "人物.yml",
    "cast.yaml",
    "cast.yml",
)

# State machine (architecture ch.9)
ST_NEED_ASSETS = "待补资料"
ST_PENDING = "待执行"
ST_RUNNING = "处理中"
ST_REVIEW = "待人工检查"
ST_DONE = "已完成"
ST_FAILED = "失败"
ALL_STATES = (ST_NEED_ASSETS, ST_PENDING, ST_RUNNING, ST_REVIEW, ST_DONE, ST_FAILED)

# Character authorization (required before L1 人物替换)
AUTH_OWN_SHOOT = "自有拍摄"
AUTH_LICENSED = "已授权"
AUTH_OWN_GEN = "自有生成角色"
AUTH_TYPES = (AUTH_OWN_SHOOT, AUTH_LICENSED, AUTH_OWN_GEN)

ASSET_FACE = "face"
ASSET_COSTUME = "costume"
ASSET_VOICE = "voice"
ASSET_3D = "3d"
ASSET_KINDS = (ASSET_FACE, ASSET_COSTUME, ASSET_VOICE, ASSET_3D)
ASSET_LABELS = {
    ASSET_FACE: "脸",
    ASSET_COSTUME: "服装",
    ASSET_VOICE: "声音",
    ASSET_3D: "3D",
}

# Engine status shown in the desktop title bar
ENG_IDLE = "待命"
ENG_RUNNING = "运行中"
ENG_MISSING = "模型缺失"

PRODUCT_NAME = "本地再铸"
PRODUCT_TAGLINE = "本地短剧再铸工作台"

# Workbench stages (original names; do not clone third-party product step labels)
STAGE_GROUPS: list[tuple[str, str, tuple[str, ...]]] = [
    ("scan", "01 扫描与补资料检查", ("scan", "check")),
    ("extract", "02 解析提取（切镜/字幕）", ("ingest", "scenes", "ocr", "asr", "merge")),
    ("library", "03 人物提取与建库", ("track", "identity", "library")),
    ("replace", "04 人物替换", ("replace",)),
    ("compose", "05 合成导出", ("compose",)),
    ("template", "06 模板套用", ("template",)),
    ("generate", "07 按剧本生成（先静音）", ("generate",)),
    ("dub", "08 贴轨与字幕", ("dub",)),
    ("export", "09 成本表 / 剪映工程导出", ("export", "cost", "jianying")),
]
STAGE_GROUP_IDS = [g[0] for g in STAGE_GROUPS]
STAGE_LABELS = {g[0]: g[1] for g in STAGE_GROUPS}

# Optional stages: not included when「全部阶段按序执行」is on. Upload default OFF.
OPTIONAL_STAGE_GROUPS: list[tuple[str, str, tuple[str, ...]]] = [
    ("pose", "动作关键点（可选）", ("pose",)),
    ("upload", "上传（可选，默认关闭）", ("upload",)),
]

# Resource tags for the CPU/GPU overlap scheduler.
CPU_PRE_STAGES = {"scan", "extract", "template"}
GPU_STAGES = {"library", "replace", "generate", "pose", "l2"}
CPU_POST_STAGES = {"compose", "dub", "export", "cost", "jianying", "upload", "quality"}

# Hardware capability tiers (architecture ch.12)
TIER_BASIC = "basic"
TIER_MAIN = "4060ti-8g"
TIER_RECO = "4060ti-16g"

MAX_STEP_RETRIES = 2
DEFAULT_CONCURRENCY = 1
MAX_CONCURRENCY = 2
