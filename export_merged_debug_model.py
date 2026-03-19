#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

try:
    from LLM import llm_knn_helpers as helpers
    from LLM.hybrid_runtime import _resolve_config_path
except ModuleNotFoundError as e:
    if e.name != "LLM":
        raise
    import llm_knn_helpers as helpers  # type: ignore[no-redef]
    from hybrid_runtime import _resolve_config_path  # type: ignore[no-redef]


def _resolve_dtype(dtype_name: str, *, auto_dtype: torch.dtype) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name == "auto":
        return auto_dtype
    return mapping[dtype_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the debug LoRA adapter into the base model and export a standalone model directory."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the merged model and tokenizer will be written.",
    )
    parser.add_argument(
        "--config-path",
        default="assets_hybrid/hybrid_config.json",
        help="Hybrid config used to discover the default base model and adapter path.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Override the base model name or local path. Defaults to CIRCUIT_DEBUG_BASE_MODEL or hybrid_config.json.",
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Override the adapter directory. Defaults to the adapter_dir entry from hybrid_config.json.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used during merge.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Torch dtype used while loading the model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_dir = Path(__file__).resolve().parent
    hybrid_assets_dir = api_dir / "assets_hybrid"
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = api_dir / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Hybrid config not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_model = (
        args.base_model
        or os.environ.get("CIRCUIT_DEBUG_BASE_MODEL")
        or config.get("model_name")
    )
    if not base_model:
        raise ValueError("Could not determine the base model name.")

    adapter_dir = _resolve_config_path(
        args.adapter_dir or config.get("adapter_dir"),
        hybrid_assets_dir=hybrid_assets_dir,
        api_dir=api_dir,
    )
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

    auto_device, auto_dtype = helpers.choose_device()
    device = auto_device if args.device == "auto" else args.device
    dtype = _resolve_dtype(args.dtype, auto_dtype=auto_dtype)

    tokenizer = helpers.AutoTokenizer.from_pretrained(
        str(base_model),
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = helpers.load_model(str(base_model), adapter_dir, device, dtype)
    if not hasattr(model, "merge_and_unload"):
        raise TypeError("Loaded model does not support merge_and_unload().")

    merged = model.merge_and_unload()
    merged.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": str(base_model),
        "adapter_dir": str(adapter_dir),
        "device": device,
        "dtype": str(dtype),
        "config_path": str(config_path),
    }
    (output_dir / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Merged model written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
