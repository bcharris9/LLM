from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_PARENT = SCRIPT_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

try:
    from LLM import llm_knn_helpers as helpers
except ModuleNotFoundError as e:
    if e.name != "LLM":
        raise
    import llm_knn_helpers as helpers  # type: ignore[no-redef]


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_DIR = Path("circuit_debug_api/assets_hybrid/adapter")
DEFAULT_KNN_REF = Path("circuit_debug_api/assets_hybrid/knn_ref_train_instruct.jsonl")
DEFAULT_CATALOG = Path("circuit_debug_api/assets/circuit_catalog.json")
DEFAULT_REPO_OUTPUT_ROOT = REPO_ROOT / "circuit_debug_api/packaged_reports"
DEFAULT_REPORT_GLOB = str(DEFAULT_REPO_OUTPUT_ROOT / "qwen15b*_report.json")


def _metric_from_report(report_payload: dict, metric: str):
    value = report_payload
    for part in metric.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _read_report_metric(path: Path, metric: str) -> float | None:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return _metric_from_report(payload, metric)


def _find_report_paths(report_glob: str) -> list[Path]:
    p = Path(report_glob)
    if any(ch in report_glob for ch in ("*", "?", "[")):
        if p.is_absolute():
            return sorted(p.parent.glob(p.name))
        return sorted(Path.cwd().glob(report_glob))
    if p.exists():
        return [p]
    return []


def _numeric_step_from_path(path: Path) -> int | None:
    m = re.search(r"checkpoint-(\d+)$", path.as_posix())
    return int(m.group(1)) if m else None


def _resolve_adapter_payload(adapter_root: Path) -> Path | None:
    if not adapter_root.is_dir():
        return None
    direct = adapter_root / "adapter_config.json"
    if direct.exists():
        return adapter_root

    ckpts = [
        ck
        for ck in adapter_root.glob("checkpoint-*")
        if ck.is_dir() and (ck / "adapter_config.json").exists()
    ]
    if not ckpts:
        return None

    ckpts.sort(key=lambda p: _numeric_step_from_path(p) or 0, reverse=True)
    return ckpts[0]


def _iter_adapter_candidates(roots: list[Path]):
    for root in roots:
        if not root.exists():
            continue
        direct_payload = _resolve_adapter_payload(root)
        if direct_payload is not None:
            yield root, direct_payload
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            payload = _resolve_adapter_payload(d)
            if payload is None:
                continue
            if not (d.name.startswith("qwen15b_") or d.name == "adapter"):
                continue
            yield d, payload


def _name_similarity(target: str, candidate: str) -> tuple[int, int]:
    t = set(target.split("_"))
    c = set(candidate.split("_"))
    overlap = len(t.intersection(c))
    # Prefer candidates with similarly-sized token sets.
    size_penalty = -abs(len(t) - len(c))
    return overlap, size_penalty


def _select_best_report_adapter(
    report_glob: str,
    candidate_roots: list[Path],
    metric: str,
) -> tuple[Path, Path, float] | None:
    report_paths = _find_report_paths(report_glob)
    if not report_paths:
        return None

    adapters = list(_iter_adapter_candidates(candidate_roots))
    if not adapters:
        return None

    best: tuple[float, int, int, int, Path, Path] | None = None
    # tuple fields: metric, overlap, step_bonus, size_penalty, report_path, payload_path

    for rpt in report_paths:
        metric_value = _read_report_metric(rpt, metric)
        if metric_value is None:
            continue

        stem = rpt.stem.removesuffix("_report")
        requested_ckpt: int | None = None
        m = re.search(r"_ckpt(\d+)", stem)
        if m:
            requested_ckpt = int(m.group(1))

        # try exact stem directory first
        direct = rpt.parent / stem
        direct_payload = _resolve_adapter_payload(direct)
        if direct_payload:
            overlap, size_penalty = _name_similarity(stem, direct.name)
            rec = (metric_value, overlap, 100, size_penalty, rpt, direct_payload)
            if best is None or rec > best:
                best = rec
            continue

        # otherwise fuzzy match qwen15b directories
        base_prefix = stem.split(f"_ckpt{requested_ckpt}")[0] if requested_ckpt else stem
        for candidate_name, payload in adapters:
            overlap, size_penalty = _name_similarity(base_prefix, candidate_name.name)
            if overlap == 0:
                if len(adapters) != 1:
                    continue
                overlap = 1
                size_penalty = 0

            step_bonus = 0
            candidate_step = _numeric_step_from_path(payload)
            if requested_ckpt is not None:
                if candidate_step == requested_ckpt:
                    step_bonus = 50
                elif candidate_step is not None:
                    # small penalty for distant checkpoint mismatch
                    step_bonus = -abs(candidate_step - requested_ckpt) // 50

            rec = (metric_value, overlap, step_bonus, size_penalty, rpt, payload)
            if best is None or rec > best:
                best = rec

    if best is None:
        return None

    return best[4], best[5], float(best[0])


def _copy_adapter(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_file(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def _read_instruction_from_jsonl(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            return str(row.get("instruction", "")).strip()
    return ""


def _portable_path_str(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("/", "\\")
    except ValueError:
        return str(path).replace("/", "\\")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LLM+KNN hybrid runtime assets for LLM")
    p.add_argument("--api-dir", type=Path, default=SCRIPT_DIR)
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    p.add_argument("--knn-ref-file", type=Path, default=DEFAULT_KNN_REF)
    p.add_argument("--catalog-file", type=Path, default=DEFAULT_CATALOG)
    p.add_argument(
        "--auto-pick-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select the best adapter from available eval reports.",
    )
    p.add_argument(
        "--candidate-roots",
        nargs="+",
        type=Path,
        default=[Path("circuit_debug_api/assets_hybrid")],
        help="Roots that contain qwen15b* adapter candidates and eval reports.",
    )
    p.add_argument(
        "--report-glob",
        type=str,
        default=DEFAULT_REPORT_GLOB,
        help="Glob used to discover qwen15b eval reports for auto-selection.",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="exact_match.pct",
        help="Metric path in report JSON used for ranking candidates.",
    )
    p.add_argument("--include-lab-id-in-prompt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--response-style", choices=["faulttype_diag_fix", "diag_fix"], default="faulttype_diag_fix")
    p.add_argument("--knn-k", type=int, default=1)
    p.add_argument("--knn-alpha", type=float, default=1.0)
    p.add_argument("--knn-weighted-vote", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--knn-standardize", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--knn-eps", type=float, default=1e-9)
    p.add_argument("--use-prerules", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--measurement-stat-mode", choices=["full", "max_only", "max_rms"], default="max_only")
    p.add_argument("--prefer-voltage-keys", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--voltage-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-measurements", type=int, default=24)
    p.add_argument("--max-deltas", type=int, default=24)
    p.add_argument("--instruction", type=str, default="")
    return p.parse_args()


def _portable_path(path: Path, *, base: Path | None = None) -> str:
    p = Path(path)
    if base is not None:
        try:
            p = p.relative_to(base)
        except ValueError:
            pass
    return p.as_posix()


def main() -> int:
    args = parse_args()
    api_dir = args.api_dir
    hybrid_dir = api_dir / "assets_hybrid"
    hybrid_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = args.adapter_dir
    selected_report: Path | None = None
    selected_metric: float | None = None

    if args.auto_pick_best:
        auto = _select_best_report_adapter(
            report_glob=args.report_glob,
            candidate_roots=[Path(r) for r in args.candidate_roots],
            metric=args.metric,
        )
        if auto is not None:
            selected_report, adapter_dir, selected_metric = auto
            print(
                f"Auto-selected adapter={adapter_dir} using report={selected_report.name} "
                f"metric={args.metric}={selected_metric:.6f}"
            )
        else:
            print("Auto-selection found no compatible model candidates. Falling back to --adapter-dir.")

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter dir: {adapter_dir}")
    if not args.knn_ref_file.exists():
        raise FileNotFoundError(f"Missing KNN reference file: {args.knn_ref_file}")
    if not args.catalog_file.exists():
        raise FileNotFoundError(f"Missing catalog file (build tabular assets first): {args.catalog_file}")

    adapter_payload_dir = _resolve_adapter_payload(adapter_dir)
    if adapter_payload_dir is None:
        raise FileNotFoundError(f"Invalid adapter source (no LoRA payload found): {adapter_dir}")

    adapter_dest = hybrid_dir / "adapter"
    _copy_adapter(adapter_payload_dir, adapter_dest)

    knn_ref_dest = hybrid_dir / "knn_ref_train_instruct.jsonl"
    _copy_file(args.knn_ref_file, knn_ref_dest)

    ref_rows = helpers.load_jsonl(knn_ref_dest)
    knn_index = helpers.build_knn_index(ref_rows)
    knn_index_dest = hybrid_dir / "knn_index.joblib"
    joblib.dump(knn_index, knn_index_dest)

    instruction = args.instruction.strip() or _read_instruction_from_jsonl(knn_ref_dest)
    config = {
        "backend": "llm_knn_hybrid",
        "model_name": args.model_name,
        "adapter_dir": _portable_path(adapter_dest, base=api_dir),
        "knn_ref_file": _portable_path(knn_ref_dest, base=api_dir),
        "knn_index_file": _portable_path(knn_index_dest, base=api_dir),
        "catalog_file": _portable_path(args.catalog_file, base=api_dir),
        "response_style": args.response_style,
        "decode_mode": "score_classes_knn",
        "knn_k": int(args.knn_k),
        "knn_alpha": float(args.knn_alpha),
        "knn_weighted_vote": bool(args.knn_weighted_vote),
        "knn_standardize": bool(args.knn_standardize),
        "knn_eps": float(args.knn_eps),
        "use_prerules": bool(args.use_prerules),
        "include_lab_id_in_prompt": bool(args.include_lab_id_in_prompt),
        "measurement_stat_mode": args.measurement_stat_mode,
        "prefer_voltage_keys": bool(args.prefer_voltage_keys),
        "voltage_only": bool(args.voltage_only),
        "max_measurements": int(args.max_measurements),
        "max_deltas": int(args.max_deltas),
        "instruction": instruction,
        "auto_selected": bool(args.auto_pick_best and selected_report is not None),
        "auto_selected_report": _portable_path_str(selected_report),
        "auto_selected_metric": args.metric,
        "auto_selected_metric_value": selected_metric,
    }
    config_path = hybrid_dir / "hybrid_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Wrote adapter: {adapter_dest}")
    print(f"Wrote knn ref: {knn_ref_dest}")
    print(f"Wrote knn index: {knn_index_dest}")
    print(f"Wrote config: {config_path}")
    print(f"KNN rows={len(ref_rows)} vectors={len(knn_index.get('vectors', []))} features={len(knn_index.get('keys', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
