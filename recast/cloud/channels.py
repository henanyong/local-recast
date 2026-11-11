"""Volcengine / DeepSeek / OpenAI-compatible channels. Default OFF.

Any real call must be confirmed first with an estimate. Every call writes
cost into the runs table via the caller. Core pipeline works with no keys.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from recast.cloud.secrets import redact


class CloudDisabled(RuntimeError):
    pass


class CloudCancelled(RuntimeError):
    pass


ConfirmFn = Callable[[dict[str, Any]], bool]


# Rough published list prices, used only for the confirmation estimate.
# USD per 1K tokens (input, output). Adjust in settings if needed.
DEFAULT_RATES = {
    "deepseek": (0.00014, 0.00028),
    "openai": (0.0005, 0.0015),
    "volcengine": (0.0003, 0.0006),
    "generic": (0.0005, 0.0015),
}


@dataclass
class CloudCall:
    channel: str
    ok: bool
    text: str = ""
    cost: float = 0.0
    model: str = ""
    usage: dict[str, Any] | None = None
    error: str | None = None
    skipped: bool = False


def estimate_cost(
    channel: str,
    *,
    prompt_tokens: int = 800,
    completion_tokens: int = 400,
    rates: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    rates = rates or DEFAULT_RATES
    inn, out = rates.get(channel) or rates["generic"]
    cost = (prompt_tokens / 1000.0) * inn + (completion_tokens / 1000.0) * out
    return {
        "channel": channel,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimate": round(cost, 6),
        "currency": "USD",
        "message": f"预估调用「{channel}」约 {cost:.4f} USD（按输入 {prompt_tokens} / 输出 {completion_tokens} token 估算）。确认后才会真正发请求。",
    }


def invoke_chat(
    channel: str,
    messages: list[dict[str, str]],
    *,
    enabled: bool,
    credentials: dict[str, Any] | None,
    model: str | None = None,
    confirm: ConfirmFn | None = None,
    auto_confirm: bool = False,
    timeout: float = 60.0,
) -> CloudCall:
    """Call an OpenAI-compatible chat endpoint after confirmation.

    If `enabled` is False or there are no credentials, no HTTP request is made.
    """
    if not enabled:
        return CloudCall(channel=channel, ok=False, skipped=True, error="云通道已关闭（默认关闭，核心流程可离线跑完）")
    creds = credentials or {}
    api_key = creds.get("api_key") or creds.get("sk") or ""
    ak = creds.get("ak") or creds.get("access_key") or ""
    if not api_key and not ak:
        return CloudCall(channel=channel, ok=False, skipped=True, error="未配置密钥，已跳过云调用")

    est = estimate_cost(channel)
    if not auto_confirm:
        if confirm is None:
            # refuse rather than silently spend money
            raise CloudCancelled("云调用需要先确认预估费用。")
        if not confirm(est):
            raise CloudCancelled("用户取消了云调用。")

    base = creds.get("base_url") or _default_base(channel)
    model = model or creds.get("model") or _default_model(channel)
    url = base.rstrip("/") + "/chat/completions"
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.4}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    token = api_key or ak
    req.add_header("Authorization", f"Bearer {token}")
    # Volcengine also accepts AK/SK via custom headers if provided
    if creds.get("ak") and creds.get("sk") and channel == "volcengine":
        req.add_header("X-Access-Key", str(creds["ak"]))
        # do not log the secret
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        return CloudCall(channel=channel, ok=False, model=model, error=redact(str(exc)))

    text = ""
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        text = json.dumps(data, ensure_ascii=False)[:2000]
    usage = data.get("usage") or {}
    pt = int(usage.get("prompt_tokens") or est["prompt_tokens"])
    ct = int(usage.get("completion_tokens") or est["completion_tokens"])
    cost = estimate_cost(channel, prompt_tokens=pt, completion_tokens=ct)["estimate"]
    return CloudCall(channel=channel, ok=True, text=text, cost=cost, model=model, usage=usage)


def _default_base(channel: str) -> str:
    return {
        "deepseek": "https://api.deepseek.com/v1",
        "volcengine": "https://ark.cn-beijing.volces.com/api/v3",
        "openai": "https://api.openai.com/v1",
        "generic": "http://127.0.0.1:8000/v1",
    }.get(channel, "https://api.openai.com/v1")


def _default_model(channel: str) -> str:
    return {
        "deepseek": "deepseek-chat",
        "volcengine": "doubao-pro-32k",
        "openai": "gpt-4o-mini",
        "generic": "gpt-4o-mini",
    }.get(channel, "gpt-4o-mini")
