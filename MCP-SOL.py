import sys
import json
import os
import re
import urllib.request
import urllib.error

# Environment Variable support (fallback to default string if not set in environment)
API_KEY = os.environ.get("FIREWORKS_API_KEY", "YOUR_FIREWORKS_API_KEY")
API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL_NAME = "accounts/fireworks/routers/kimi-k3-fast"

def query_kimi(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    req = urllib.request.Request(
        API_URL, 
        data=json.dumps(data).encode('utf-8'), 
        headers=headers, 
        method="POST"
    )
    
    try:
        # Added explicit timeout (180 seconds / 3 minutes)
        with urllib.request.urlopen(req, timeout=180) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            
            if "choices" in res_data and len(res_data["choices"]) > 0:
                return res_data["choices"][0]["message"]["content"], False
            elif "error" in res_data:
                return f"API Error: {res_data['error']}", True
            else:
                return f"Unexpected response format: {json.dumps(res_data)}", True

    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode('utf-8')
            return f"HTTP Error {e.code}: {err_body}", True
        except Exception:
            return f"HTTP Error {e.code}: {e.reason}", True
    except urllib.error.URLError as e:
        return f"Network Error: {e.reason}", True
    except Exception as e:
        return f"Unexpected error: {str(e)}", True

def encode_toon(json_input, indent=2, delimiter=',', key_folding="off", flatten_depth=float('inf'), replacer=None):
    if isinstance(json_input, str):
        data = json.loads(json_input)
    else:
        data = json_input

    def fold_keys(obj, current_depth=0):
        if not isinstance(obj, dict) or current_depth >= flatten_depth:
            return obj
        res = {}
        for k, v in obj.items():
            v_folded = fold_keys(v, current_depth + 1) if isinstance(v, dict) else v
            if key_folding == "safe" and isinstance(v_folded, dict) and len(v_folded) == 1:
                sub_k, sub_v = list(v_folded.items())[0]
                res[f"{k}.{sub_k}"] = sub_v
            else:
                res[k] = v_folded
        return res

    if key_folding == "safe":
        data = fold_keys(data)

    def is_tabular_array(arr):
        return isinstance(arr, list) and len(arr) > 0 and all(isinstance(x, dict) for x in arr)

    def format_tabular(prefix, arr):
        if not arr:
            return f"{prefix}[0]{{}}:"
        all_keys = list(arr[0].keys())
        if replacer:
            replacer_set = set(str(r) for r in replacer)
            all_keys = [k for k in all_keys if str(k) in replacer_set]
        header = f"{prefix}[{len(arr)}]{{{','.join(all_keys)}}}:"
        rows = []
        for item in arr:
            row = delimiter.join(str(item.get(k, '')) for k in all_keys)
            rows.append(row)
        return header + "\n" + "\n".join(rows)

    if is_tabular_array(data):
        return format_tabular("", data)
    elif isinstance(data, dict):
        lines = []
        for k, v in data.items():
            if is_tabular_array(v):
                lines.append(format_tabular(k, v))
            else:
                lines.append(f"{k}: {json.dumps(v)}")
        return "\n".join(lines)
    
    return json.dumps(data, indent=indent)

def decode_toon(toon_input, strict=True, expand_paths="off", indent=2):
    lines = [line.strip() for line in toon_input.strip().splitlines() if line.strip()]
    if not lines:
        return json.dumps({}, indent=indent)

    header_pattern = re.compile(r"^([a-zA-Z0-9_\.]*)\[(\d+)\]\{([^\}]*)\}:$")

    result = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        match = header_pattern.match(line)
        if match:
            key, count_str, cols_str = match.groups()
            expected_count = int(count_str)
            cols = [c.strip() for c in cols_str.split(",") if c.strip()]
            
            rows = []
            i += 1
            delim = ","
            if i < len(lines):
                if "\t" in lines[i]:
                    delim = "\t"
                elif "|" in lines[i]:
                    delim = "|"

            while i < len(lines) and not header_pattern.match(lines[i]) and ":" not in lines[i]:
                row_values = lines[i].split(delim)
                obj = {}
                for idx, col in enumerate(cols):
                    val_str = row_values[idx].strip() if idx < len(row_values) else ""
                    if val_str.isdigit():
                        val = int(val_str)
                    elif val_str.replace('.', '', 1).isdigit() and val_str.count('.') == 1:
                        val = float(val_str)
                    elif val_str.lower() == 'true':
                        val = True
                    elif val_str.lower() == 'false':
                        val = False
                    elif val_str.lower() == 'null':
                        val = None
                    else:
                        val = val_str
                    obj[col] = val
                rows.append(obj)
                i += 1
            
            if strict and len(rows) != expected_count:
                raise ValueError(f"Strict validation failed: expected {expected_count} rows, got {len(rows)}")

            if key:
                result[key] = rows
            else:
                return json.dumps(rows, indent=indent)
        else:
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    result[k.strip()] = json.loads(v.strip())
                except Exception:
                    result[k.strip()] = v.strip()
            i += 1

    if expand_paths == "safe":
        expanded_result = {}
        for k, v in result.items():
            parts = k.split(".")
            curr = expanded_result
            for part in parts[:-1]:
                curr = curr.setdefault(part, {})
            curr[parts[-1]] = v
        result = expanded_result

    return json.dumps(result, indent=indent)

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
                    "version": "1.1.0"
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
                                }
                            },
                            "required": ["prompt"]
                        }
                    },
                    {
                        "name": "encode_toon",
                        "description": "Converts JSON data into the compact TOON format to save LLM tokens.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "json": {
                                    "type": "string",
                                    "description": "The JSON data (serialized as a string) to encode."
                                },
                                "indent": {
                                    "type": "number",
                                    "description": "Number of spaces for indentation.",
                                    "default": 2
                                },
                                "delimiter": {
                                    "type": "string",
                                    "description": "Delimiter for arrays/rows. Options: ',', '\\t', '|'.",
                                    "default": ","
                                },
                                "keyFolding": {
                                    "type": "string",
                                    "description": "Collapse single-key wrapper chains (e.g. a.b.c). Options: 'off', 'safe'.",
                                    "default": "off"
                                },
                                "flattenDepth": {
                                    "type": "number",
                                    "description": "Maximum depth to apply key folding."
                                },
                                "replacer": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Array of properties to include in the output."
                                }
                            },
                            "required": ["json"]
                        }
                    },
                    {
                        "name": "decode_toon",
                        "description": "Parses TOON formatted text back into standard JSON string.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "toon": {
                                    "type": "string",
                                    "description": "The TOON formatted string to decode."
                                },
                                "strict": {
                                    "type": "boolean",
                                    "description": "Enforce strict validation (e.g., checking declared array lengths).",
                                    "default": True
                                },
                                "expandPaths": {
                                    "type": "string",
                                    "description": "Reconstruct dotted keys into nested objects. Options: 'off', 'safe'.",
                                    "default": "off"
                                },
                                "indent": {
                                    "type": "number",
                                    "description": "Number of spaces for output JSON indentation.",
                                    "default": 2
                                }
                            },
                            "required": ["toon"]
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
                result_text, is_error = query_kimi(prompt)
                payload = {
                    "content": [{"type": "text", "text": result_text}]
                }
                if is_error:
                    payload["isError"] = True
                respond(req_id, payload)
            elif name == "encode_toon":
                json_data = arguments.get("json", "{}")
                indent = arguments.get("indent", 2)
                delimiter = arguments.get("delimiter", ",")
                key_folding = arguments.get("keyFolding", "off")
                flatten_depth = arguments.get("flattenDepth", float('inf'))
                replacer = arguments.get("replacer", None)

                try:
                    toon_output = encode_toon(
                        json_data, 
                        indent=indent, 
                        delimiter=delimiter, 
                        key_folding=key_folding, 
                        flatten_depth=flatten_depth, 
                        replacer=replacer
                    )
                    respond(req_id, {"content": [{"type": "text", "text": toon_output}]})
                except Exception as e:
                    respond(req_id, payload={"content": [{"type": "text", "text": f"Encoding Error: {str(e)}"}], "isError": True})
            elif name == "decode_toon":
                toon_data = arguments.get("toon", "")
                strict = arguments.get("strict", True)
                expand_paths = arguments.get("expandPaths", "off")
                indent = arguments.get("indent", 2)

                try:
                    json_output = decode_toon(
                        toon_data, 
                        strict=strict, 
                        expand_paths=expand_paths, 
                        indent=indent
                    )
                    respond(req_id, {"content": [{"type": "text", "text": json_output}]})
                except Exception as e:
                    respond(req_id, payload={"content": [{"type": "text", "text": f"Decoding Error: {str(e)}"}], "isError": True})
            else:
                respond(req_id, error={"code": -32601, "message": f"Method '{name}' not found"})
        else:
            if req_id is not None:
                respond(req_id, error={"code": -32601, "message": f"Unhandled method: {method}"})

if __name__ == "__main__":
    main()
