#!/usr/bin/env python3
"""Fetch mem0 MCP server handshake result (initialize + tools/list + prompts/list + prompts/get).

Default: prints the final handshake JSON to stdout. Verbose progress goes to stderr.
Use --output/-o to write the JSON to a file instead.
"""
import argparse
import json
import os
import re
import sys
from typing import Any

MCP_URL = "https://mcp.mem0.ai/mcp/"


def parse_sse(text: str) -> list[dict]:
    """Parse SSE (Server-Sent Events) text into a list of JSON-RPC messages."""
    messages = []
    for block in re.split(r"\n\n+", text.strip()):
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("data: "):
                data_lines.append(line[6:])
        if data_lines:
            for data in data_lines:
                try:
                    messages.append(json.loads(data))
                except json.JSONDecodeError:
                    pass
    return messages


def jsonrpc_request(method: str, params: dict | None = None, request_id: int | str | None = 1) -> dict:
    body = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        body["id"] = request_id
    if params is not None:
        body["params"] = params
    return body


def post_sse(client: Any, headers: dict, body: dict) -> tuple[int, dict, str]:
    """POST JSON-RPC, handle SSE response. Returns (status, headers, raw_text)."""
    resp = client.post(MCP_URL, json=body, headers=headers)
    return resp.status_code, dict(resp.headers), resp.text


def main() -> dict:
    parser = argparse.ArgumentParser(
        description="Fetch mem0 MCP server handshake and output the JSON result.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Write the final handshake JSON to this file instead of stdout.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Print full request/response details to stderr.",
    )
    args = parser.parse_args()

    mem0_api_key = os.environ.get("MEM0_API_KEY", "")
    if not mem0_api_key:
        print("Error: MEM0_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    try:
        import httpx
    except ImportError:
        print("Error: httpx is required to fetch the MCP handshake.", file=sys.stderr)
        sys.exit(1)

    def log(*a, **kw):
        if args.verbose:
            print(*a, file=sys.stderr, **kw)

    headers = {
        "Authorization": f"Token {mem0_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    client = httpx.Client(timeout=30, follow_redirects=True)

    # ── Step 1: initialize ──────────────────────────────────────────
    log(">>> initialize ... ", end="")
    init_req = jsonrpc_request(
        "initialize",
        {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "mem0-mcp-handshake-script", "version": "1.0.0"},
        },
        request_id=1,
    )
    log(f"\n{json.dumps(init_req, indent=2)}")

    status, resp_headers, body = post_sse(client, headers, init_req)
    mcp_session_id = resp_headers.get("mcp-session-id", "")
    if mcp_session_id:
        headers["Mcp-Session-Id"] = mcp_session_id
    log(f"HTTP {status}  session={mcp_session_id or 'N/A'}")
    init_msgs = parse_sse(body)
    init_result = next((msg.get("result", {}) for msg in init_msgs if "result" in msg), {})
    log(json.dumps(init_msgs, indent=2))

    if status != 200:
        print(f"ERROR: HTTP {status}", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: notifications/initialized ──────────────────────────
    log(">>> notifications/initialized ... ", end="")
    notif = jsonrpc_request("notifications/initialized", request_id=None)
    status, _, body = post_sse(client, headers, notif)
    log(f"HTTP {status}")

    # ── Step 3: tools/list ─────────────────────────────────────────
    log(">>> tools/list ... ", end="")
    tools_req = jsonrpc_request("tools/list", {}, request_id=2)
    log(f"\n{json.dumps(tools_req, indent=2)}")

    status, _, body = post_sse(client, headers, tools_req)
    tools_msgs = parse_sse(body)
    tools_result = {}
    for msg in tools_msgs:
        if "result" in msg:
            tools_result = msg
    log(json.dumps(tools_msgs, indent=2))

    if not tools_result:
        print(f"ERROR: no result in tools/list response\n{body[:2000]}", file=sys.stderr)
        sys.exit(1)
    tools = tools_result.get("result", {}).get("tools", [])
    log(f"HTTP {status}  ({len(tools)} tools)")

    # ── Step 4: prompts/list ───────────────────────────────────────
    log(">>> prompts/list ... ", end="")
    prompts_list_req = jsonrpc_request("prompts/list", {}, request_id=3)
    log(f"\n{json.dumps(prompts_list_req, indent=2)}")

    status, _, body = post_sse(client, headers, prompts_list_req)
    prompts = []
    prompts_msgs = parse_sse(body)
    for msg in prompts_msgs:
        if "result" in msg:
            prompts = msg["result"].get("prompts", [])
    log(json.dumps(prompts_msgs, indent=2))
    log(f"HTTP {status}  ({len(prompts)} prompts)")

    # ── Step 5: prompts/get for each prompt ────────────────────────
    all_prompts: list[dict] = []
    for i, p in enumerate(prompts):
        prompt_name = p["name"]
        log(f">>> prompts/get ({prompt_name}) ... ", end="")
        get_req = jsonrpc_request("prompts/get", {"name": prompt_name}, request_id=4 + i)
        status, _, body = post_sse(client, headers, get_req)
        get_msgs = parse_sse(body)
        for msg in get_msgs:
            if "result" in msg:
                msg["result"]["name"] = prompt_name
                all_prompts.append(msg["result"])
        log(f"HTTP {status}")

    # ── Build & output ─────────────────────────────────────────────
    full_handshake = {
        "server_info": init_result.get("serverInfo", {}),
        "protocolVersion": init_result.get("protocolVersion", "2025-11-25"),
        "tools": tools,
        "prompts": all_prompts,
    }

    json_output = json.dumps(full_handshake, indent=2, ensure_ascii=False)

    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(json_output)
        print(f"Saved to: {out_path}", file=sys.stderr)
    else:
        print(json_output)

    log(f"\nTools: {len(tools)}  |  Prompts: {len(all_prompts)}")
    for t in tools:
        log(f"  tool: {t['name']}")
    for p in all_prompts:
        log(f"  prompt: {p.get('name', '?')}")

    return full_handshake


if __name__ == "__main__":
    main()
