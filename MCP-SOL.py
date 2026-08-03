import sys
import json
import os
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
                    "version": "1.0.0"
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
            else:
                respond(req_id, error={"code": -32601, "message": f"Method '{name}' not found"})
        else:
            if req_id is not None:
                respond(req_id, error={"code": -32601, "message": f"Unhandled method: {method}"})

if __name__ == "__main__":
    main()
