from __future__ import annotations

import argparse
import json

import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive terminal client for POST /chat or POST /chat/{lab_number}."
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument(
        "--lab-number",
        type=int,
        default=None,
        help="Optional lab number. If omitted, the server will infer it from the question or search all labs.",
    )
    p.add_argument(
        "--question",
        default=None,
        help="Optional one-shot question. If omitted, starts interactive mode.",
    )
    p.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds.")
    return p.parse_args()


def _pretty(obj: object) -> str:
    try:
        return json.dumps(obj, indent=2)
    except Exception:
        return str(obj)


def _prompt_lab_number() -> int | None:
    while True:
        try:
            raw = input("Lab number (press Enter to auto-detect): ").strip()
        except EOFError:
            return None
        if not raw:
            return None
        try:
            lab_number = int(raw)
        except ValueError:
            print("Enter a positive integer or press Enter to skip.")
            continue
        if lab_number <= 0:
            print("Enter a positive integer or press Enter to skip.")
            continue
        return lab_number


def _ask(base_url: str, question: str, timeout: int, lab_number: int | None = None) -> tuple[bool, str]:
    path = f"/chat/{lab_number}" if lab_number is not None else "/chat"
    r = requests.post(
        f"{base_url.rstrip('/')}{path}",
        json={"question": question},
        timeout=timeout,
    )
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        return False, f"HTTP {r.status_code}\n{_pretty(detail)}"

    data = r.json()
    if not isinstance(data, dict) or "answer" not in data:
        return False, f"Unexpected response shape:\n{_pretty(data)}"
    return True, str(data["answer"])


def main() -> int:
    args = parse_args()
    base = args.base_url.rstrip("/")
    lab_number = args.lab_number

    try:
        if args.question:
            ok, text = _ask(base, args.question, args.timeout, lab_number=lab_number)
            if ok:
                print(text)
                return 0
            print(text)
            return 1

        print("Student Chat Mode")
        if lab_number is None:
            lab_number = _prompt_lab_number()
        if lab_number is None:
            print("Lab: auto-detect")
        else:
            print(f"Lab: {lab_number}")
        print("Type your question and press Enter.")
        print("Commands: /quit, /exit")

        while True:
            try:
                q = input("\nYou: ").strip()
            except EOFError:
                print()
                return 0
            if not q:
                continue
            if q.lower() in {"/quit", "/exit", "quit", "exit"}:
                return 0

            ok, text = _ask(base, q, args.timeout, lab_number=lab_number)
            if ok:
                print(f"Assistant: {text}")
            else:
                print(f"Assistant error: {text}")

    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except requests.RequestException as e:
        print(f"Connection error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
