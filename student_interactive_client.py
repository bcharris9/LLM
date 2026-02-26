from __future__ import annotations

import argparse
import json
import re
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Interactive student client for Circuit Debug API. "
            "Fetches required nodes for a circuit and prompts for measurements one at a time."
        )
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--circuit", default=None, help="Optional exact circuit name to skip circuit selection.")
    p.add_argument(
        "--ask-source-currents",
        action="store_true",
        help="Also prompt for optional source currents one at a time.",
    )
    p.add_argument(
        "--show-golden",
        action="store_true",
        help="Show golden values in prompts (useful for instructor demo, not student use).",
    )
    p.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        default=True,
        help="Submit with strict=false (not recommended; allows missing nodes).",
    )
    p.add_argument(
        "--save-payload",
        default=None,
        help="Optional path to save the submitted payload JSON.",
    )
    return p.parse_args()


def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2)


def fetch_json(url: str, timeout: int = 30) -> dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return data


def choose_circuit(base: str, preselected: str | None) -> tuple[str, list[str]]:
    circuits_doc = fetch_json(f"{base}/circuits")
    circuits = circuits_doc.get("circuits", [])
    if not isinstance(circuits, list) or not circuits:
        raise RuntimeError("No circuits returned by API")

    if preselected:
        if preselected not in circuits:
            raise RuntimeError(f"Unknown circuit '{preselected}'. Use /circuits to list valid names.")
        return preselected, circuits

    def lab_key(name: str) -> str:
        m = re.match(r"(?i)^(lab\d+)", name.strip())
        if m:
            return f"Lab{m.group(1)[3:]}"
        return "Other"

    def lab_sort_key(lab: str) -> tuple[int, str]:
        m = re.match(r"^Lab(\d+)$", lab)
        if m:
            return (0, f"{int(m.group(1)):04d}")
        return (1, lab.lower())

    by_lab: dict[str, list[str]] = {}
    for name in circuits:
        by_lab.setdefault(lab_key(str(name)), []).append(str(name))
    for lab in by_lab:
        by_lab[lab].sort(key=lambda s: s.lower())

    labs = sorted(by_lab.keys(), key=lab_sort_key)

    print("Available labs:")
    for i, lab in enumerate(labs, start=1):
        print(f"  {i:>3}. {lab} ({len(by_lab[lab])} circuits)")

    selected_lab: str | None = None
    while True:
        raw = input("\nChoose a lab by number or name (example: 4 or Lab4): ").strip()
        if not raw:
            continue
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(labs):
                selected_lab = labs[idx - 1]
                break
            # Also accept bare lab numbers like "4" -> "Lab4"
            lab_name = f"Lab{idx}"
            if lab_name in by_lab:
                selected_lab = lab_name
                break
            print(f"Invalid lab selection. Enter 1-{len(labs)} or a valid lab name.")
            continue
        normalized = raw.strip()
        if re.fullmatch(r"(?i)lab\d+", normalized):
            normalized = f"Lab{int(re.sub(r'(?i)^lab', '', normalized))}"
        elif re.fullmatch(r"\d+", normalized):
            normalized = f"Lab{int(normalized)}"
        if normalized in by_lab:
            selected_lab = normalized
            break
        print("Invalid lab name. Try again.")

    assert selected_lab is not None
    lab_circuits = by_lab[selected_lab]
    print(f"\nCircuits in {selected_lab}:")
    for i, name in enumerate(lab_circuits, start=1):
        print(f"  {i:>3}. {name}")

    while True:
        raw = input("\nChoose a circuit by number or exact name: ").strip()
        if not raw:
            continue
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(lab_circuits):
                return lab_circuits[idx - 1], circuits
            print(f"Invalid number. Enter 1-{len(lab_circuits)}.")
            continue
        if raw in lab_circuits:
            return raw, circuits
        print(f"Invalid circuit name for {selected_lab}. Try again.")


def _parse_float_input(raw: str) -> float:
    return float(raw.strip())


def prompt_measurements(
    label: str,
    items: list[dict[str, Any]],
    key_name: str,
    *,
    show_golden: bool,
    allow_skip: bool,
) -> dict[str, float]:
    values: dict[str, float] = {}
    if not items:
        return values

    print(f"\nEnter {label} one at a time.")
    print("Commands: 'back' = previous item, 'quit' = exit" + (", 'skip' = skip item" if allow_skip else ""))

    i = 0
    while i < len(items):
        item = items[i]
        name = str(item.get(key_name))
        prompt = f"[{i + 1}/{len(items)}] {name}"
        if show_golden and item.get("golden_value") is not None:
            prompt += f" (golden {item['golden_value']})"
        prompt += ": "

        raw = input(prompt).strip()
        if not raw:
            continue
        lower = raw.lower()
        if lower in {"quit", "exit", "q"}:
            raise KeyboardInterrupt()
        if lower == "back":
            if i > 0:
                prev_item = items[i - 1]
                prev_name = str(prev_item.get(key_name))
                values.pop(prev_name, None)
                i -= 1
            else:
                print("Already at the first item.")
            continue
        if allow_skip and lower == "skip":
            i += 1
            continue
        try:
            values[name] = _parse_float_input(raw)
            i += 1
        except ValueError:
            print("Invalid number. Enter a numeric value in volts/amps (examples: 5, -0.23, 1.2e-3).")

    return values


def print_node_checklist(nodes_doc: dict[str, Any]) -> None:
    circuit_name = nodes_doc.get("circuit_name", "<unknown>")
    print(f"\nSelected circuit: {circuit_name}")
    print(f"Required nodes: {nodes_doc.get('node_count', 0)}")
    print(f"Optional source currents: {nodes_doc.get('source_current_count', 0)}")

    nodes = nodes_doc.get("nodes", [])
    if nodes:
        print("\nNode list:")
        for item in nodes:
            print(f"  - {item.get('node_name')} ({item.get('measurement_key')})")

    srcs = nodes_doc.get("source_currents", [])
    if srcs:
        print("\nSource current list (optional):")
        for item in srcs:
            print(f"  - {item.get('source_name')} ({item.get('measurement_key')})")


def main() -> int:
    args = parse_args()
    base = args.base_url.rstrip("/")

    try:
        health = fetch_json(f"{base}/health")
        print("API health:")
        print(pretty(health))

        circuit_name, _ = choose_circuit(base, args.circuit)
        nodes_doc = fetch_json(f"{base}/circuits/{circuit_name}/nodes")
        print_node_checklist(nodes_doc)

        nodes = nodes_doc.get("nodes", [])
        srcs = nodes_doc.get("source_currents", [])

        node_voltages = prompt_measurements(
            "node voltages (V, measured relative to ground)",
            nodes if isinstance(nodes, list) else [],
            "node_name",
            show_golden=args.show_golden,
            allow_skip=False,
        )

        source_currents: dict[str, float] = {}
        if args.ask_source_currents and isinstance(srcs, list) and srcs:
            print("\nSource currents are optional. You can type 'skip' for any source you did not measure.")
            source_currents = prompt_measurements(
                "source currents (A)",
                srcs,
                "source_name",
                show_golden=args.show_golden,
                allow_skip=True,
            )
        elif isinstance(srcs, list) and srcs:
            yn = input("\nDo you want to enter source currents too? [y/N]: ").strip().lower()
            if yn in {"y", "yes"}:
                print("You can type 'skip' for any source you did not measure.")
                source_currents = prompt_measurements(
                    "source currents (A)",
                    srcs,
                    "source_name",
                    show_golden=args.show_golden,
                    allow_skip=True,
                )

        defaults = nodes_doc.get("golden_defaults", {}) if isinstance(nodes_doc.get("golden_defaults"), dict) else {}
        payload: dict[str, Any] = {
            "circuit_name": circuit_name,
            "node_voltages": node_voltages,
            "source_currents": source_currents,
            "temp": float(defaults.get("temp", 27.0)),
            "tnom": float(defaults.get("tnom", 27.0)),
            "strict": bool(args.strict),
        }

        print("\nPayload to submit:")
        print(pretty(payload))
        submit = input("\nSubmit to /debug now? [Y/n]: ").strip().lower()
        if submit in {"n", "no"}:
            print("Submission cancelled.")
            return 0

        if args.save_payload:
            with open(args.save_payload, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved payload: {args.save_payload}")

        r = requests.post(f"{base}/debug", json=payload, timeout=120)
        r.raise_for_status()
        result = r.json()
        print("\nAPI response:")
        print(pretty(result))
        return 0

    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 130
    except requests.HTTPError as e:
        detail = None
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text if e.response is not None else str(e)
        print("\nAPI error:")
        print(pretty(detail) if isinstance(detail, dict) else str(detail))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
