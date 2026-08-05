#!/usr/bin/env python3
import os
import json
from typing import Any, Dict, Optional

import requests
from mcp.server.fastmcp import FastMCP

try:
    from toon_basic import encode as toon_encode, decode as toon_decode
except Exception:
    toon_encode = None
    toon_decode = None

mcp = FastMCP("antigravity-mcp")

BASE_URL = os.getenv("ANTIGRAVITY_API_URL", "https://api.fireworks.ai/inference/v1")
MODEL = os.getenv("ANTIGRAVITY_MODEL", "accounts/fireworks/routers/kimi-k3-fast")
API_KEY = os.getenv("FIREWORKS_API_KEY") or os.getenv("ANTIGRAVITY_API_KEY")
TIMEOUT = float(os.getenv("ANTIGRAVITY_TIMEOUT", "25.0"))

GEMINI_MODEL = os.getenv("ANTIGRAVITY_GEMINI_MODEL", "gemini-3.1-pro")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models")

REASON_CODES = {
    "MISSING_API_KEY": {
        "explanation": "API key is not configured.",
        "action": "Configure FIREWORKS_API_KEY in mcp_config.json.",
    },
    "TIMEOUT_EXCEEDED": {
        "explanation": "Network read timed out on long stream.",
        "action": 'Use chunked generation or prompt "Gemini step in".',
    },
    "NETWORK_UNREACHABLE": {
        "explanation": "DNS or socket error.",
        "action": "Check local network connection.",
    },
    "HTTP_401": {
        "explanation": "Unauthorized request.",
        "action": "Verify your Fireworks AI API key.",
    },
    "HTTP_429": {
        "explanation": "Too many concurrent requests.",
        "action": "Retry query with backoff or use TOON format.",
    },
    "HTTP_500": {
        "explanation": "Provider service disruption.",
        "action": "Check Fireworks AI status or use Gemini fallback.",
    },
    "HTTP_503": {
        "explanation": "Provider service disruption.",
        "action": "Check Fireworks AI status or use Gemini fallback.",
    },
    "KEY_COLLISION": {
        "explanation": "Non-bijective key_map alias collision.",
        "action": "Provide unique alias mappings in key_map.",
    },
}


def soft_error(reason_code: str, detail: str = "") -> str:
    info = REASON_CODES.get(reason_code, {"explanation": "Unknown error.", "action": "Review configuration."})
    payload = {
        "ok": False,
        "reason_code": reason_code,
        "explanation": info["explanation"],
        "action": info["action"],
        "detail": detail,
    }
    return json.dumps(payload, ensure_ascii=False)


def _headers() -> Dict[str, str]:
    if not API_KEY or API_KEY in ("YOUR_FIREWORKS_KEY", "PLACEHOLDER", "changeme"):
        raise ValueError("MISSING_API_KEY")
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def _fireworks_chat(messages: list[dict[str, str]], use_toon: bool = False) -> str:
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    payload: Dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    if use_toon:
        payload["response_format"] = {"type": "text"}

    try:
        r = requests.post(url, headers=_headers(), json=payload, timeout=TIMEOUT)
        if r.status_code == 401:
            return soft_error("HTTP_401", r.text)
        if r.status_code == 429:
            return soft_error("HTTP_429", r.text)
        if r.status_code == 500:
            return soft_error("HTTP_500", r.text)
        if r.status_code == 503:
            return soft_error("HTTP_503", r.text)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except ValueError as e:
        return soft_error("MISSING_API_KEY", str(e))
    except requests.exceptions.Timeout as e:
        return soft_error("TIMEOUT_EXCEEDED", str(e))
    except requests.exceptions.ConnectionError as e:
        return soft_error("NETWORK_UNREACHABLE", str(e))
    except requests.exceptions.RequestException as e:
        return soft_error("NETWORK_UNREACHABLE", str(e))
    except Exception as e:
        return soft_error("NETWORK_UNREACHABLE", str(e))


def _gemini_generate(prompt: str) -> str:
    if "Gemini step in" not in prompt:
        return soft_error(
            "HTTP_503",
            "Gemini fallback blocked. Explicit approval required: include exact phrase 'Gemini step in'.",
        )
    if not GEMINI_API_KEY:
        return soft_error("MISSING_API_KEY", "GEMINI_API_KEY is not configured.")
    url = f"{GEMINI_API_URL.rstrip('/')}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    try:
        r = requests.post(url, json=body, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.Timeout as e:
        return soft_error("TIMEOUT_EXCEEDED", str(e))
    except requests.exceptions.ConnectionError as e:
        return soft_error("NETWORK_UNREACHABLE", str(e))
    except requests.exceptions.HTTPError as e:
        return soft_error("HTTP_503", str(e))
    except Exception as e:
        return soft_error("NETWORK_UNREACHABLE", str(e))


@mcp.tool(name="antigravity.query")
def query(prompt: str, use_toon: bool = False, allow_gemini_step_in: bool = False) -> str:
    if allow_gemini_step_in or "Gemini step in" in prompt:
        return _gemini_generate(prompt)
    result = _fireworks_chat([{"role": "user", "content": prompt}], use_toon=use_toon)
    if isinstance(result, str) and result.startswith('{"ok": false'):
        return result
    return result


@mcp.tool(name="antigravity.encode_toon")
def encode_toon(data: Any, key_map: Optional[Dict[str, str]] = None) -> str:
    if toon_encode is None:
        return soft_error("NETWORK_UNREACHABLE", "toon_basic is not installed")
    if key_map:
        rev = {}
        for k, v in key_map.items():
            if v in rev:
                return soft_error("KEY_COLLISION", f"Alias collision for key: {v}")
            rev[v] = k
    try:
        return toon_encode(data, key_map=key_map) if key_map else toon_encode(data)
    except Exception as e:
        return soft_error("KEY_COLLISION", str(e))


@mcp.tool(name="antigravity.decode_toon")
def decode_toon(text: str) -> Any:
    if toon_decode is None:
        return soft_error("NETWORK_UNREACHABLE", "toon_basic is not installed")
    try:
        return toon_decode(text)
    except Exception as e:
        return soft_error("KEY_COLLISION", str(e))


if __name__ == "__main__":
    mcp.run(transport="stdio")
