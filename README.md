# Antigravity MCP AnyModel Integration

[![Repository](https://img.shields.io/badge/GitHub-tidypy%2FAntigravity--MCP--AnyModel-blue?style=flat-square&logo=github)](https://github.com/tidypy/Antigravity-MCP-AnyModel.git)
[![Protocol](https://img.shields.io/badge/Protocol-MCP%20JSON--RPC%202.0-orange?style=flat-square)](https://modelcontextprotocol.io/)
[![Orchestrator](https://img.shields.io/badge/Orchestrator-Google%20Antigravity-4285F4?style=flat-square)](https://deepmind.google/)

An Model Context Protocol (MCP) bridge enabling **Google Antigravity** to seamlessly delegate code generation, heavy reasoning, and specialized tasks to third-party LLMs (such as **Kimi K3 Fast** hosted on Fireworks AI or other serverless BYOK providers).

---

## 📦 TOON Format Integration (Save ~35%–60% LLM Tokens)

This server includes native support for **TOON (Token-Optimized Object Notation)** via `encode_toon` and `decode_toon`.

### Why TOON?
When passing large tabular context (such as database rows, API responses, schema definitions, or file manifests) to LLMs like Kimi K3 or Gemini, standard JSON repeatedly bloats the prompt by duplicating object key names:
```json
{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
```

**TOON Format** compresses repetitive object arrays into a compact header and delimited row syntax:
```text
users[2]{id,name}:
1,Alice
2,Bob
```

### 📊 Token Reduction Benchmark
| Format | Syntax Example | Estimated Tokens | Token Savings |
| :--- | :--- | :---: | :---: |
| **Standard JSON** | `{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}` | ~38 tokens | Baseline |
| **TOON Format** | `users[2]{id,name}:1,Alice 2,Bob` | ~16 tokens | **~35% to 58% Savings** |

### Available TOON Tools:
- **`encode_toon`**: Converts JSON strings into TOON syntax. Supports custom delimiters (`,`, `\t`, `|`), property filtering (`replacer`), and single-key folding (`keyFolding="safe"`).
- **`decode_toon`**: Reconstructs TOON text back into standard formatted JSON objects.

---

## 🛠️ Quick Setup & Installation

### Step 1: Locate your MCP Customizations Folder
In Google Antigravity:
1. Go to **Settings** $\rightarrow$ **Customizations** $\rightarrow$ Click **Open MCP Folder**.
2. Alternatively, open your local configuration directory directly:
   `C:\Users\YOURUSERNAME\.gemini\config\`

### Step 2: Add the Python MCP Server Script
Paste the Python script into your configuration folder (e.g., `C:\Users\YOURUSERNAME\.gemini\config\MCP-SOL.py`).

<details>
<summary>📄 Click to view complete MCP Server Script (MCP-SOL.py)</summary>

```python
import sys
import json
import os
import re
import socket
import urllib.request
import urllib.error
from datetime import datetime

API_KEY = os.environ.get("FIREWORKS_API_KEY", "YOUR_FIREWORKS_API_KEY")
API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL_NAME = "accounts/fireworks/routers/kimi-k3-fast"
DEFAULT_TIMEOUT = int(os.environ.get("FIREWORKS_TIMEOUT", "25"))

def log_stderr(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sys.stderr.write(f"[{timestamp}] [MCP-SOL] {message}\n")
    sys.stderr.flush()

def query_kimi(prompt, timeout_seconds=None):
    timeout = timeout_seconds if timeout_seconds is not None else DEFAULT_TIMEOUT
    log_stderr(f"Initiating request to Fireworks AI (Model: {MODEL_NAME}, Timeout: {timeout}s)")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
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
                log_stderr("Request completed successfully.")
                return res_data["choices"][0]["message"]["content"], False
            elif "error" in res_data:
                err_msg = json.dumps(res_data["error"])
                log_stderr(f"API Error payload returned: {err_msg}")
                diag = format_diagnostic_error("FIREWORKS_API_ERROR", f"Fireworks API Error: {err_msg}")
                return diag, True
            else:
                diag = format_diagnostic_error("UNEXPECTED_RESPONSE_FORMAT", f"Unexpected format: {json.dumps(res_data)}")
                return diag, True

    except urllib.error.HTTPError as e:
        code = e.code
        try:
            err_body = e.read().decode('utf-8')
        except Exception:
            err_body = str(e.reason)
            
        reason_code = f"HTTP_{code}"
        if code == 401:
            reason_code = "UNAUTHORIZED_401"
        elif code == 429:
            reason_code = "RATE_LIMITED_429"
        elif code in (500, 502, 503, 504):
            reason_code = f"SERVERLESS_OVERLOAD_{code}"

        log_stderr(f"HTTPError {code}: {reason_code} - {err_body}")
        diag = format_diagnostic_error(reason_code, f"HTTP {code} ({e.reason}): {err_body}")
        return diag, True

    except urllib.error.URLError as e:
        if isinstance(e.reason, socket.timeout) or "timed out" in str(e.reason).lower():
            reason_code = "TIMEOUT_EXCEEDED"
            msg = f"Fireworks AI serverless API timed out after {timeout}s. Endpoint hung before returning tokens."
        else:
            reason_code = "NETWORK_UNREACHABLE"
            msg = f"Network connection failed: {e.reason}"
            
        log_stderr(f"URLError: {reason_code} - {msg}")
        diag = format_diagnostic_error(reason_code, msg, timeout_seconds=timeout)
        return diag, True

    except Exception as e:
        reason_code = "UNKNOWN_FAILURE"
        msg = f"Unexpected error: {str(e)}"
        log_stderr(f"Exception: {reason_code} - {msg}")
        diag = format_diagnostic_error(reason_code, msg)
        return diag, True

def format_diagnostic_error(reason_code, details, timeout_seconds=None):
    error_payload = {
        "status": "ERROR",
        "reason_code": reason_code,
        "details": details,
        "timestamp": datetime.now().isoformat(),
        "action_recommendation": "Fireworks serverless call stalled/failed. Switch to Orchestrator (Gemini step-in) or retry with smaller prompt context."
    }
    if timeout_seconds:
        error_payload["configured_timeout_seconds"] = timeout_seconds
    return json.dumps(error_payload, indent=2)

# (See MCP-SOL.py for complete TOON encode_toon & decode_toon implementations)
```
</details>

### Step 3: Configure `mcp_config.json`
Add your server configuration to `C:\Users\YOURUSERNAME\.gemini\config\mcp_config.json`:

```json
{
  "mcpServers": {
    "kimi-k3-fast": {
      "command": "python",
      "args": ["C:/Users/YOURUSERNAME/.gemini/config/MCP-SOL.py"],
      "env": {
        "FIREWORKS_API_KEY": "YOUR_FIREWORKS_API_KEY_HERE",
        "FIREWORKS_TIMEOUT": "25"
      }
    }
  }
}
```

---

## 💬 How to Use (Chat Invocation)

Simply invoke the tools in your Antigravity chat prompt:

> **User Prompt:**  
> *"Use encode_toon to compress this user table and send it to Kimi K3."*

Antigravity will automatically call `encode_toon`, compress the tabular JSON, and pass the reduced prompt to `query_kimi`.

---

## ⚡ Fast Timeout & Fallback Diagnostics

### Built-in 25-Second Timeout & Diagnostic Reason Codes
To prevent long 3-minute hangs when serverless API endpoints stall or overload, `MCP-SOL.py` includes a fast **25-second default timeout** (configurable via `FIREWORKS_TIMEOUT`).

When an API call stalls or errors out, the server immediately trips and returns machine-readable diagnostic details:

```json
{
  "status": "ERROR",
  "reason_code": "TIMEOUT_EXCEEDED",
  "details": "Fireworks AI serverless API timed out after 25s. Endpoint hung before returning tokens.",
  "timestamp": "2026-08-04T10:20:45",
  "configured_timeout_seconds": 25,
  "action_recommendation": "Fireworks serverless call stalled/failed. Switch to Orchestrator (Gemini step-in) or retry with smaller prompt context."
}
```

### Supported Reason Codes:
- `TIMEOUT_EXCEEDED`: API hung beyond configured timeout limit (25s).
- `UNAUTHORIZED_401`: Invalid API Key.
- `RATE_LIMITED_429`: Fireworks rate limit or quota exceeded.
- `SERVERLESS_OVERLOAD_500_503_504`: Serverless model router or gateway overloaded.
- `NETWORK_UNREACHABLE`: Connection dropped or DNS failure.

---

## 🛡️ Operational Rules Enforced

1. **Strict Kimi K3 Delegation:**
   Antigravity will **NEVER** silently override Kimi K3 unless explicitly instructed or when an explicit diagnostic error payload is returned.
2. **Immediate Error & Fallback Reporting:**
   If `query_kimi` times out or fails, Antigravity receives the exact `reason_code` within 25 seconds and asks for retry confirmation or Gemini step-in.
3. **Gemini Step-In Requires Explicit Approval:**
   Gemini will take over code generation only when you explicitly state **"Gemini step in"**.

---

## 📄 License
MIT License. Created for use with Google Antigravity & Model Context Protocol.
