#!/usr/bin/env python3
"""Smoke-test POST /chat on the running server.py FastAPI app.

Examples:
  python test_chat_endpoint.py
  python test_chat_endpoint.py --base-url http://127.0.0.1:8000
  python test_chat_endpoint.py --require-answer
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: int = 10) -> tuple[int, Any]:
    """POST a JSON payload and return either decoded JSON or the raw response body."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def get_json(url: str, timeout: int = 10) -> tuple[int, Any]:
    """GET a JSON endpoint and return the decoded response when possible."""
    req = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def main() -> int:
    """Run a small contract test against the live `/chat` endpoint."""
    parser = argparse.ArgumentParser(description="Test /chat endpoint behavior on server.py")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL where server.py is running.")
    parser.add_argument(
        "--require-answer",
        action="store_true",
        help="Fail if valid question does not return 200 with an 'answer'.",
    )
    args = parser.parse_args()

    chat_url = f"{args.base_url.rstrip('/')}/chat"
    openapi_url = f"{args.base_url.rstrip('/')}/openapi.json"
    failures: list[str] = []
    deps_unavailable = False

    # Contract precheck: server.py should define POST /chat with JSON body.
    try:
        status, schema = get_json(openapi_url)
    except urllib.error.URLError as e:
        print(f"FAIL: Could not connect to {openapi_url}: {e}")
        return 1

    if status != 200 or not isinstance(schema, dict):
        print(f"FAIL: Could not read OpenAPI schema from {openapi_url}. Status={status}, body={schema}")
        return 1

    chat_post = ((schema.get("paths") or {}).get("/chat") or {}).get("post")
    if not isinstance(chat_post, dict):
        failures.append("OpenAPI has no POST /chat route.")
    else:
        has_request_body = "requestBody" in chat_post
        if not has_request_body:
            failures.append(
                "POST /chat does not use a JSON request body in this running app. "
                "That does not match server.py (expects {'question': ...} in body)."
            )

    # Case 1: Valid question
    status, body = post_json(chat_url, {"question": "How do I limit LED current?"})

    if status == 200:
        if not isinstance(body, dict) or "answer" not in body:
            failures.append(f"Valid question returned 200 but missing 'answer': {body}")
    elif status == 503:
        msg = "Valid question returned 503 (Supabase/LLM not initialized)."
        deps_unavailable = True
        if args.require_answer:
            failures.append(msg)
        else:
            print(f"WARN: {msg}")
    elif status == 500:
        msg = "Valid question returned 500 (runtime dependency error while answering)."
        if args.require_answer:
            failures.append(f"{msg} body={body}")
        else:
            print(f"WARN: {msg} body={body}")
    else:
        failures.append(f"Valid question expected 200/503/500, got {status}: {body}")

    # Case 2: Empty string question
    status, body = post_json(chat_url, {"question": "   "})
    if deps_unavailable:
        expected_msgs = {
            "Supabase client not initialized; set SUPABASE_KEY.",
            "LLM client not initialized.",
        }
        if status != 503 or not isinstance(body, dict) or body.get("detail") not in expected_msgs:
            failures.append(
                "Empty question expected 503 because dependencies are unavailable, "
                f"got {status}: {body}"
            )
    else:
        if status != 400 or body != {"detail": "Question cannot be empty."}:
            failures.append(f"Empty question expected 400 + exact detail, got {status}: {body}")

    # Case 3: Missing required field
    status, body = post_json(chat_url, {})
    if status != 422:
        failures.append(f"Missing question expected 422, got {status}: {body}")

    if failures:
        print("FAIL: /chat endpoint checks failed")
        for f in failures:
            print(f"- {f}")
        return 1

    print("PASS: /chat endpoint checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
