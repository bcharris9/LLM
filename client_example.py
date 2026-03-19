"""Example client that exercises the catalog and debug endpoints."""

from __future__ import annotations

import argparse
import json
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the demo debug client."""
    p = argparse.ArgumentParser(description="Example client for the Circuit Debug API")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--circuit", default=None, help="Optional exact circuit name. If omitted, uses the first circuit.")
    p.add_argument(
        "--payload-file",
        default=None,
        help="Optional path to a JSON file containing an exact POST /debug payload. If provided, this is sent as-is.",
    )
    p.add_argument(
        "--demo-use-golden-values",
        action="store_true",
        help="Auto-fill node voltages with golden values from /circuits/{name}/nodes (demo only).",
    )
    p.add_argument("--demo-offset-node", default=None, help="Optional node to perturb in demo mode (e.g. N001).")
    p.add_argument("--demo-offset-volts", type=float, default=0.5)
    p.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Submit with strict validation (recommended). Use --no-strict for partial input testing.",
    )
    return p.parse_args()


def pretty(obj: Any) -> str:
    """Pretty-print JSON payloads for terminal output."""
    return json.dumps(obj, indent=2)


def main() -> int:
    """Fetch a circuit, build a demo payload, and submit it to `/debug`."""
    args = parse_args()
    base = args.base_url.rstrip("/")

    r = requests.get(f"{base}/circuits", timeout=30)
    r.raise_for_status()
    circuits_doc = r.json()
    print("GET /circuits")
    print(pretty(circuits_doc))

    circuits = circuits_doc.get("circuits", [])
    if not circuits:
        raise RuntimeError("No circuits returned by API")
    payload: dict[str, Any] | None = None
    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise RuntimeError("--payload-file must contain a JSON object")
        if not payload.get("circuit_name"):
            raise RuntimeError("--payload-file JSON must include circuit_name")
        circuit_name = str(payload["circuit_name"])
    else:
        circuit_name = args.circuit or circuits[0]

    r = requests.get(f"{base}/circuits/{circuit_name}/nodes", timeout=30)
    r.raise_for_status()
    nodes_doc = r.json()
    print("\nGET /circuits/{name}/nodes")
    print(pretty(nodes_doc))

    if payload is None:
        payload = {
            "circuit_name": circuit_name,
            "node_voltages": {},
            "source_currents": {},
            "strict": bool(args.strict),
        }

        if args.demo_use_golden_values:
            for item in nodes_doc.get("nodes", []):
                if item.get("golden_value") is not None:
                    payload["node_voltages"][item["node_name"]] = float(item["golden_value"])
            for item in nodes_doc.get("source_currents", []):
                if item.get("golden_value") is not None:
                    payload["source_currents"][item["source_name"]] = float(item["golden_value"])
            if args.demo_offset_node and args.demo_offset_node in payload["node_voltages"]:
                payload["node_voltages"][args.demo_offset_node] += float(args.demo_offset_volts)
        else:
            for item in nodes_doc.get("nodes", []):
                if item.get("golden_value") is not None:
                    payload["node_voltages"][item["node_name"]] = float(item["golden_value"])
                else:
                    payload["node_voltages"][item["node_name"]] = 0.0
            if args.demo_offset_node and args.demo_offset_node in payload["node_voltages"]:
                payload["node_voltages"][args.demo_offset_node] += float(args.demo_offset_volts)
    else:
        payload.setdefault("strict", bool(args.strict))

    print("\nPOST /debug payload")
    print(pretty(payload))

    r = requests.post(f"{base}/debug", json=payload, timeout=60)
    r.raise_for_status()
    debug_doc = r.json()
    print("\nPOST /debug response")
    print(pretty(debug_doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
