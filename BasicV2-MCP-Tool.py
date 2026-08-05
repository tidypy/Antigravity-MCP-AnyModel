import sys
import json
import time
import socket
import urllib.request
import urllib.error
import os
from typing import Any

API_URL = os.environ.get("MCP_API_URL", "")
API_KEY = os.environ.get("MCP_API_KEY", "")
MODEL_NAME = os.environ.get("MCP_MODEL_NAME", "")

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "fundamental-mcp-server"
SERVER_VERSION = "1.1.0"

SOFT_ERRORS = {
    "MISSING_API_KEY": ("API key is not configured.", "Configure MCP_API_KEY in your environment."),
    "TIMEOUT_EXCEEDED": ("Network read timed out on long stream.", "Retry or shorten the request."),
    "NETWORK_UNREACHABLE": ("DNS or socket error.", "Check local network connection."),
    "HTTP_401": ("Unauthorized request.", "Verify your API key."),
    "HTTP_429": ("Too many concurrent requests.", "Retry with backoff."),
    "HTTP_500": ("Provider service disruption.", "Retry later."),
    "HTTP_503": ("Provider service disruption.", "Retry later."),
    "KEY_COLLISION": ("Non-bijective key_map alias collision.", "Provide unique alias mappings in key_map."),
}

def make_error(reason_code: str, details: str | None = None) -> dict:
    explanation, action = SOFT_ERRORS.get(
        reason_code,
        ("Unknown error.", "Inspect configuration and retry."),
    )
    payload = {
        "reason_code": reason_code,
        "explanation": explanation,
        "action": action,
    }
    if details is not None:
        payload["details"] = details
    return payload

def respond(req_id: Any, result: Any = None, error: dict | None = None) -> None:
    if req_id is None:
        return
    msg = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def _strip_json_comments(text: str) -> str:
    # Minimal safe cleaner for config-like JSON payloads if needed.
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))

def _validate_payload(payload: Any) -> None:
    if payload is None:
        raise ValueError("payload is required")

def http_post_json(url: str, payload: Any, headers: dict | None = None, timeout: int = 25, retries: int = 3):
    headers = headers or {}
    _validate_payload(payload)
    body = json.dumps(payload).encode("utf-8")
    last_err = None

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read().decode("utf-8")
                return json.loads(data), None
        except socket.timeout as e:
            last_err = make_error("TIMEOUT_EXCEEDED", str(e))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", "ignore")
            if e.code == 401:
                last_err = make_error("HTTP_401", body_text)
            elif e.code == 429:
                last_err = make_error("HTTP_429", body_text)
            elif e.code == 500:
                last_err = make_error("HTTP_500", body_text)
            elif e.code == 503:
                last_err = make_error("HTTP_503", body_text)
            else:
                last_err = make_error(f"HTTP_{e.code}", body_text)
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last_err = make_error("NETWORK_UNREACHABLE", str(e))
        except Exception as e:
            last_err = make_error("NETWORK_UNREACHABLE", f"Unexpected error: {e}")

        if attempt < retries:
            time.sleep(min(2 * attempt, 5))

    return None, last_err

def to_toon_scalar(obj: Any) -> str:
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, float):
        if obj != obj:
            return "nan"
        if obj == float("inf"):
            return "inf"
        if obj == float("-inf"):
            return "-inf"
        return repr(obj)
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, str):
        if obj == "":
            return '""'
        if all(ch.isalnum() or ch in "_-./:" for ch in obj) and not obj[0].isdigit():
            return obj
        return json.dumps(obj, ensure_ascii=False)
    return json.dumps(obj, ensure_ascii=False)

def toon_encode(obj: Any) -> str:
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        return "\n".join(f"{to_toon_scalar(k)}: {toon_encode(v) if isinstance(v, (dict, list)) else to_toon_scalar(v)}" for k, v in obj.items())
    if isinstance(obj, list):
        if not obj:
            return "[]"
        return "\n".join(f"- {toon_encode(v) if isinstance(v, (dict, list)) else to_toon_scalar(v)}" for v in obj)
    return to_toon_scalar(obj)

def toon_decode(text: str) -> Any:
    # Lightweight decode: accept JSON first, otherwise return a structured error.
    s = text.strip()
    if not s:
        return {}
    if s[:1] in "{[":
        return json.loads(s)
    raise ValueError("TOON decoding not implemented in this minimal transport build")

dumps = toon_encode
loads = toon_decode

def initialize_result():
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
    }

def list_tools():
    return {
        "tools": [
            {
                "name": "send_request",
                "description": "Send a request to the configured backend service.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "payload": {"type": "object", "description": "Arbitrary JSON payload to send."},
                        "format": {"type": "string", "enum": ["json", "toon"], "default": "json"},
                    },
                    "required": ["payload"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "encode_toon",
                "description": "Encode JSON-compatible data to TOON.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object"},
                    },
                    "required": ["data"],
                    "additionalProperties": False,
                },
            },
        ]
    }

def call_tool(name: str, arguments: dict):
    if name == "send_request":
        if not API_URL:
            return {"isError": True, "content": [{"type": "text", "text": json.dumps(make_error("NETWORK_UNREACHABLE", "MCP_API_URL is not set."))}]}
        if not API_KEY:
            return {"isError": True, "content": [{"type": "text", "text": json.dumps(make_error("MISSING_API_KEY"))}]}

        payload = arguments.get("payload")
        fmt = arguments.get("format", "json")

        result, err = http_post_json(
            API_URL,
            payload,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                **({"X-Model": MODEL_NAME} if MODEL_NAME else {}),
            },
            timeout=25,
            retries=3,
        )

        if err is not None:
            return {"isError": True, "content": [{"type": "text", "text": json.dumps(err)}]}

        text = json.dumps(result, ensure_ascii=False) if fmt == "json" else toon_encode(result)
        return {"content": [{"type": "text", "text": text}]}

    if name == "encode_toon":
        return {"content": [{"type": "text", "text": toon_encode(arguments.get("data"))}]}

    return {"isError": True, "content": [{"type": "text", "text": json.dumps({"code": -32601, "message": f"Tool '{name}' not found"})}]}

def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"JSON Parse Error: {e}\n")
            sys.stderr.flush()
            continue

        req_id = req.get("id")
        method = req.get("method")

        if method == "initialize":
            respond(req_id, initialize_result())
        elif method and method.startswith("notifications/"):
            pass
        elif method == "tools/list":
            respond(req_id, list_tools())
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {}) or {}
            respond(req_id, call_tool(name, arguments))
        else:
            if req_id is not None:
                respond(req_id, error={"code": -32601, "message": f"Unhandled method: {method}"})

if __name__ == "__main__":
    main()
