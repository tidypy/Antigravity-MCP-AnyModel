import sys
import json
import os
import re
import urllib.request
import urllib.error

# Reconfigure stdout/stdin to UTF-8 on Windows for safe stdio transport
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_MODEL = "accounts/fireworks/routers/kimi-k3-fast"
DEFAULT_TIMEOUT_SEC = 25

# ==========================================
# TOON (Token-Oriented Object Notation) Engine
# ==========================================

_PRIMITIVE_LIKE = re.compile(r'-?\d+(\.\d+)?([eE][+-]?\d+)?|true|false|null')
_SPECIALS = set('",:\n\t\r[]{}')

def _encode_str(s):
    """Encodes a string into TOON, adding quotes if it contains specials, whitespace, or primitive keywords."""
    if not s:
        return '""'
    if s != s.strip() or _PRIMITIVE_LIKE.fullmatch(s) or any(c in _SPECIALS for c in s):
        esc = s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
        return f'"{esc}"'
    return s

def toon_encode(obj, indent=0):
    """
    Encodes Python dictionary/list/primitive into TOON format.
    Minimizes token usage using YAML-style indentation and CSV-style tabular headers for arrays.
    """
    prefix = "  " * indent
    if obj is None:
        return "null"
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    elif isinstance(obj, (int, float)):
        return str(obj)
    elif isinstance(obj, str):
        return _encode_str(obj)
    elif isinstance(obj, dict):
        if not obj:
            return ""
        lines = []
        for k, v in obj.items():
            encoded_key = _encode_str(k) if any(c in k for c in _SPECIALS) else k
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                cols = list(v[0].keys())
                if all(list(x.keys()) == cols for x in v):
                    col_str = ",".join(cols)
                    lines.append(f"{prefix}{encoded_key}[{len(v)}]{{{col_str}}}:")
                    for row in v:
                        row_vals = [toon_encode(row.get(c)) for c in cols]
                        lines.append(f"{prefix}  " + ",".join(row_vals))
                else:
                    lines.append(f"{prefix}{encoded_key}:")
                    for item in v:
                        lines.append(f"{prefix}  -")
                        lines.append(toon_encode(item, indent + 2))
            elif isinstance(v, list) and v and all(not isinstance(x, (dict, list)) for x in v):
                lines.append(f"{prefix}{encoded_key}[{len(v)}]: " + ",".join(toon_encode(x) for x in v))
            elif isinstance(v, dict):
                lines.append(f"{prefix}{encoded_key}:")
                sub = toon_encode(v, indent + 1)
                if sub:
                    lines.append(sub)
            elif isinstance(v, list):
                lines.append(f"{prefix}{encoded_key}:")
                for x in v:
                    lines.append(f"{prefix}  - " + toon_encode(x))
            else:
                lines.append(f"{prefix}{encoded_key}: {toon_encode(v)}")
        return "\n".join(lines)
    elif isinstance(obj, list):
        if not obj:
            return "[]"
        if all(isinstance(x, dict) for x in obj):
            cols = list(obj[0].keys())
            lines = [f"items[{len(obj)}]{{{','.join(cols)}}}:"]
            for row in obj:
                lines.append("  " + ",".join(toon_encode(row.get(c)) for c in cols))
            return "\n".join(lines)
        return ",".join(toon_encode(x) for x in obj)
    return _encode_str(str(obj))

def encode_toon(json_data, indent=0, **kwargs):
    """Wrapper that accepts JSON string or dict and returns TOON string."""
    if isinstance(json_data, str):
        obj = json.loads(json_data)
    else:
        obj = json_data
    return toon_encode(obj, indent=indent)

def _parse_val(val_str):
    """Parses a scalar TOON value string into a Python object."""
    val = val_str.strip()
    if val == "null" or val == "":
        return None
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        inner = val[1:-1]
        return inner.replace('\\"', '"').replace('\\\\', '\\').replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
    try:
        if "." in val or "e" in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val

def toon_decode(toon_str):
    """Decodes a TOON formatted string back into standard Python dictionary/list structures."""
    lines = [l for l in toon_str.splitlines() if l.strip()]
    if not lines:
        return {}
    
    root = {}
    stack = [(0, root)]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        
        while stack and stack[-1][0] > indent and len(stack) > 1:
            stack.pop()
            
        current_obj = stack[-1][1]
        
        tab_match = re.match(r"^([\w_]+)\[(\d+)\]\{([^}]+)\}:$", content)
        prim_arr_match = re.match(r"^([\w_]+)\[(\d+)\]:\s*(.*)$", content)
        kv_match = re.match(r"^([\w_]+):\s*(.*)$", content)
        
        if tab_match:
            key, count, cols_str = tab_match.groups()
            count = int(count)
            cols = [c.strip() for c in cols_str.split(",")]
            arr = []
            for _ in range(count):
                i += 1
                if i < len(lines):
                    row_line = lines[i].strip()
                    parts = [p.strip() for p in row_line.split(",")]
                    row_dict = {}
                    for c_idx, col in enumerate(cols):
                        row_dict[col] = _parse_val(parts[c_idx]) if c_idx < len(parts) else None
                    arr.append(row_dict)
            if isinstance(current_obj, dict):
                current_obj[key] = arr
        elif prim_arr_match:
            key, count, inline_vals = prim_arr_match.groups()
            vals = [_parse_val(v) for v in inline_vals.split(",")] if inline_vals else []
            if isinstance(current_obj, dict):
                current_obj[key] = vals
        elif kv_match:
            key, val_part = kv_match.groups()
            if not val_part:
                sub_dict = {}
                if isinstance(current_obj, dict):
                    current_obj[key] = sub_dict
                stack.append((indent + 2, sub_dict))
            else:
                if isinstance(current_obj, dict):
                    current_obj[key] = _parse_val(val_part)
        i += 1
        
    return root

def decode_toon(toon_str, **kwargs):
    """Wrapper that decodes TOON string and returns JSON string."""
    decoded = toon_decode(toon_str)
    return json.dumps(decoded, indent=2)

# ==========================================
# Fireworks AI LLM Service & MCP Logic
# ==========================================

def query_kimi(prompt, use_toon=False, timeout_seconds=None):
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    if not api_key or api_key == "YOUR_FIREWORKS_API_KEY":
        diag = {
            "status": "ERROR",
            "reason_code": "MISSING_API_KEY",
            "message": "FIREWORKS_API_KEY environment variable is not configured.",
            "action_recommendation": "Configure FIREWORKS_API_KEY in mcp_config.json or system environment."
        }
        return json.dumps(diag), True

    timeout = float(timeout_seconds) if timeout_seconds is not None else DEFAULT_TIMEOUT_SEC

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    final_prompt = prompt
    if use_toon:
        final_prompt = f"Please output your response using TOON (Token-Oriented Object Notation) compact format.\n\nUser Request:\n{prompt}"
        
    data = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": final_prompt}
        ]
    }
    
    req = urllib.request.Request(
        API_URL, 
        data=json.dumps(data).encode('utf-8'), 
        headers=headers, 
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            
            if "choices" in res_data and len(res_data["choices"]) > 0:
                return res_data["choices"][0]["message"]["content"], False
            elif "error" in res_data:
                diag = {
                    "status": "ERROR",
                    "reason_code": "API_ERROR",
                    "message": str(res_data["error"]),
                    "action_recommendation": "Verify API key and model parameters."
                }
                return json.dumps(diag), True
            else:
                diag = {
                    "status": "ERROR",
                    "reason_code": "UNEXPECTED_FORMAT",
                    "message": f"Unexpected response format: {json.dumps(res_data)}",
                    "action_recommendation": "Retry query."
                }
                return json.dumps(diag), True

    except urllib.error.HTTPError as e:
        diag = {
            "status": "ERROR",
            "reason_code": f"HTTP_{e.code}",
            "message": f"HTTP Error {e.code}: {e.reason}",
            "action_recommendation": "Check Fireworks AI service status."
        }
        return json.dumps(diag), True
    except urllib.error.URLError as e:
        reason_str = str(e.reason)
        if "timed out" in reason_str.lower() or "timeout" in reason_str.lower():
            reason_code = "TIMEOUT_EXCEEDED"
            action = "Request chunked generation or use 'Gemini step in'."
        else:
            reason_code = "NETWORK_UNREACHABLE"
            action = "Check network connection."
        diag = {
            "status": "ERROR",
            "reason_code": reason_code,
            "message": f"Network Error: {reason_str}",
            "action_recommendation": action
        }
        return json.dumps(diag), True
    except Exception as e:
        reason_str = str(e)
        if "timed out" in reason_str.lower() or "timeout" in reason_str.lower():
            reason_code = "TIMEOUT_EXCEEDED"
            action = "Request chunked generation or use 'Gemini step in'."
        else:
            reason_code = "UNKNOWN_ERROR"
            action = "Retry query."
        diag = {
            "status": "ERROR",
            "reason_code": reason_code,
            "message": f"Unexpected error: {reason_str}",
            "action_recommendation": action
        }
        return json.dumps(diag), True

def respond(response_id, result=None, error=None):
    if response_id is None:
        return
        
    response = {
        "jsonrpc": "2.0",
        "id": response_id
    }
    if error is not None:
        response["error"] = error
    else:
        response["result"] = result
        
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()

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
            respond(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "kimi-k3-fast-mcp-server",
                    "version": "1.2.0"
                }
            })
        elif method and method.startswith("notifications/"):
            pass
        elif method == "tools/list":
            respond(req_id, {
                "tools": [
                    {
                        "name": "query_kimi",
                        "description": "Send a prompt to the Kimi K3 Fast model on Fireworks AI for code generation and reasoning.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The detailed instructions or code query."
                                },
                                "use_toon": {
                                    "type": "boolean",
                                    "description": "Whether to request output formatted in TOON (Token-Oriented Object Notation)."
                                },
                                "timeout_seconds": {
                                    "type": "number",
                                    "description": "Custom timeout in seconds (default 25s)."
                                }
                            },
                            "required": ["prompt"]
                        }
                    },
                    {
                        "name": "encode_toon",
                        "description": "Convert JSON data into compact TOON format to save LLM tokens.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "json": {
                                    "type": "string",
                                    "description": "JSON string to encode into TOON format."
                                },
                                "json_data": {
                                    "type": "string",
                                    "description": "Alias for json argument."
                                }
                            }
                        }
                    },
                    {
                        "name": "decode_toon",
                        "description": "Convert TOON formatted string back into standard JSON format.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "toon": {
                                    "type": "string",
                                    "description": "TOON formatted string to decode."
                                },
                                "toon_data": {
                                    "type": "string",
                                    "description": "Alias for toon argument."
                                }
                            }
                        }
                    },
                    {
                        "name": "toon_encode",
                        "description": "Alias for encode_toon tool.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "json_data": {
                                    "type": "string",
                                    "description": "JSON string to encode into TOON format."
                                }
                            }
                        }
                    },
                    {
                        "name": "toon_decode",
                        "description": "Alias for decode_toon tool.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "toon_data": {
                                    "type": "string",
                                    "description": "TOON formatted string to decode."
                                }
                            }
                        }
                    }
                ]
            })
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {})
            
            if name == "query_kimi":
                prompt = arguments.get("prompt", "")
                use_toon = arguments.get("use_toon", False)
                timeout_sec = arguments.get("timeout_seconds", None)
                result_text, is_error = query_kimi(prompt, use_toon=use_toon, timeout_seconds=timeout_sec)
                
                payload = {
                    "content": [
                        {
                            "type": "text",
                            "text": result_text
                        }
                    ]
                }
                if is_error:
                    payload["isError"] = True
                respond(req_id, payload)
            elif name in ("encode_toon", "toon_encode"):
                json_str = arguments.get("json") or arguments.get("json_data") or "{}"
                try:
                    toon_result = encode_toon(json_str)
                    respond(req_id, {"content": [{"type": "text", "text": toon_result}]})
                except Exception as e:
                    respond(req_id, {"content": [{"type": "text", "text": f"Encoding Error: {str(e)}"}], "isError": True})
            elif name in ("decode_toon", "toon_decode"):
                toon_str = arguments.get("toon") or arguments.get("toon_data") or ""
                try:
                    json_result = decode_toon(toon_str)
                    respond(req_id, {"content": [{"type": "text", "text": json_result}]})
                except Exception as e:
                    respond(req_id, {"content": [{"type": "text", "text": f"Decoding Error: {str(e)}"}], "isError": True})
            else:
                respond(req_id, error={"code": -32601, "message": f"Method '{name}' not found"})
        else:
            if req_id is not None:
                respond(req_id, error={"code": -32601, "message": f"Unhandled method: {method}"})

if __name__ == "__main__":
    main()
