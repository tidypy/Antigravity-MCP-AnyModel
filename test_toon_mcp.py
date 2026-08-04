import subprocess
import json
import unittest
import os

class TestMCPServerTOON(unittest.TestCase):
    def setUp(self):
        env = os.environ.copy()
        env["FIREWORKS_API_KEY"] = "mock_key_for_unit_tests"
        self.proc = subprocess.Popen(
            ["python", "MCP-SOL.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

    def tearDown(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()

    def send_rpc(self, req):
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        return json.loads(line)

    def test_initialize_and_tools_list(self):
        init_res = self.send_rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(init_res["id"], 1)
        self.assertEqual(init_res["result"]["serverInfo"]["name"], "kimi-k3-fast-mcp-server")

        tools_res = self.send_rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = tools_res["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        self.assertIn("query_kimi", tool_names)
        self.assertIn("encode_toon", tool_names)
        self.assertIn("decode_toon", tool_names)

    def test_encode_toon_tool_call(self):
        self.send_rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        
        sample_json = json.dumps({"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]})
        call_res = self.send_rpc({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "encode_toon",
                "arguments": {
                    "json": sample_json
                }
            }
        })
        text_content = call_res["result"]["content"][0]["text"]
        self.assertIn("users[2]{id,name}:", text_content)
        self.assertIn("1,Alice", text_content)

    def test_decode_toon_tool_call(self):
        self.send_rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        
        sample_toon = "users[2]{id,name}:\n1,Alice\n2,Bob"
        call_res = self.send_rpc({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "decode_toon",
                "arguments": {
                    "toon": sample_toon
                }
            }
        })
        text_content = call_res["result"]["content"][0]["text"]
        data = json.loads(text_content)
        self.assertEqual(len(data["users"]), 2)
        self.assertEqual(data["users"][0]["name"], "Alice")

if __name__ == "__main__":
    unittest.main()
