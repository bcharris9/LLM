from __future__ import annotations

import argparse
import json
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
DEFAULT_ADAPTER_DIR = Path("pipeline/out/qwen15b_alllabs_v2_lora_e1_len384_gapfix_v1")
DEFAULT_KNN_REF = Path("pipeline/out_one_lab_all_v2_train/merged_finetune_currents_noid_v2/train_instruct_gapfix_v1.jsonl")
DEFAULT_CATALOG = SCRIPT_DIR / "assets" / "circuit_catalog.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LLM+KNN hybrid runtime assets for LLM")
    p.add_argument("--api-dir", type=Path, default=SCRIPT_DIR)
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    p.add_argument("--knn-ref-file", type=Path, default=DEFAULT_KNN_REF)
    p.add_argument("--catalog-file", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--include-lab-id-in-prompt", action=argparse.BooleanOptionalAction, default=False)
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
def _copy_adapter(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)

    def _ignore(_dir: str, names: list[str]):
        return [n for n in names if n.startswith("checkpoint-")]

    shutil.copytree(src, dst, ignore=_ignore)


def _read_instruction_from_jsonl(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            return str(row.get("instruction", "")).strip()
    return ""


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

    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter dir: {args.adapter_dir}")
    if not args.knn_ref_file.exists():
        raise FileNotFoundError(f"Missing KNN reference file: {args.knn_ref_file}")
    if not args.catalog_file.exists():
        raise FileNotFoundError(f"Missing catalog file (build tabular assets first): {args.catalog_file}")

    adapter_dest = hybrid_dir / "adapter"
    _copy_adapter(args.adapter_dir, adapter_dest)

    knn_ref_dest = hybrid_dir / "knn_ref_train_instruct.jsonl"
    shutil.copy2(args.knn_ref_file, knn_ref_dest)

    # Prebuild KNN index for faster API startup.
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
