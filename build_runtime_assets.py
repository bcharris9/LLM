from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
import sys

import joblib
import numpy as np
from xgboost import XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_PARENT = SCRIPT_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

try:
    from LLM.runtime import (
        PAIR_MC_PO,
        build_circuit_catalog,
        build_feature_dict_from_measurements,
        family_id,
    )
except ModuleNotFoundError as e:
    if e.name != "LLM":
        raise
    from runtime import (  # type: ignore[no-redef]
        PAIR_MC_PO,
        build_circuit_catalog,
        build_feature_dict_from_measurements,
        family_id,
    )


DEFAULT_TRAIN_FILES = [
    "pipeline/out_one_lab_all_v2_train/merged_finetune_currents_labid_v1/train_instruct.jsonl",
    "pipeline/out_one_lab_all_v2_train/merged_finetune_currents_labid_v1/val_instruct.jsonl",
    "pipeline/out_one_lab_all_v3_train/merged_finetune_currents_labid_v3_targeted/train_instruct.jsonl",
    "pipeline/out_one_lab_all_v3_train/merged_finetune_currents_labid_v3_targeted/val_instruct.jsonl",
    "pipeline/out_one_lab_all_v4_pairfocus_train/merged_finetune_currents_labid_v4_pairfocus/train_instruct.jsonl",
    "pipeline/out_one_lab_all_v4_pairfocus_train/merged_finetune_currents_labid_v4_pairfocus/val_instruct.jsonl",
    "pipeline/out_one_lab_all_v5_pairfocus_train/merged_finetune_currents_labid_v5_pairfocus/train_instruct.jsonl",
    "pipeline/out_one_lab_all_v5_pairfocus_train/merged_finetune_currents_labid_v5_pairfocus/val_instruct.jsonl",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build runtime assets for LLM")
    p.add_argument("--api-dir", type=Path, default=SCRIPT_DIR)
    p.add_argument(
        "--source-bundle",
        type=Path,
        default=Path("pipeline/out/tabular_fault_head_union_v2_v3_v4_v5_xgb_pair_bundle.joblib"),
    )
    p.add_argument("--golden-root", type=Path, default=Path("pipeline/out_one_lab_all_v2_train"))
    p.add_argument(
        "--train-file",
        dest="train_files",
        action="append",
        help="Optional override/additional merged train/val instruct jsonl files. Can be repeated.",
    )
    p.add_argument("--skip-family-pair-models", action="store_true")
    return p.parse_args()


def _parse_instruct_row(row: dict) -> tuple[str, str, dict[str, float], dict[str, float]]:
    import re

    text = row["input"]
    out = row["output"]
    lab_m = re.search(r"^Lab:\s*(.+)$", text, re.M)
    fault_m = re.search(r"^FaultType:\s*(\w+)", out, re.M)
    if not lab_m or not fault_m:
        raise ValueError("Could not parse instruct row")
    circuit_name = lab_m.group(1).strip()
    fault_type = fault_m.group(1).strip()

    def parse_line_map(header: str) -> dict[str, float]:
        m = re.search(rf"^{header}:\s*(.+)$", text, re.M)
        out_map: dict[str, float] = {}
        if not m:
            return out_map
        for part in m.group(1).split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip().lower()
            try:
                out_map[k] = float(v.strip())
            except ValueError:
                continue
        return out_map

    measured = parse_line_map("Measured")
    deltas_raw = parse_line_map("DeltasVsGolden")
    deltas: dict[str, float] = {}
    for k, v in deltas_raw.items():
        if k.endswith("_delta"):
            deltas[k[: -len("_delta")]] = v

    golden: dict[str, float] = {}
    for k, mv in measured.items():
        if k in deltas:
            golden[k] = mv - deltas[k]

    return circuit_name, fault_type, measured, golden


def _train_family_pair_models(
    bundle_path: Path,
    train_files: list[Path],
    out_path: Path,
) -> int:
    bundle = joblib.load(bundle_path)
    vectorizer = bundle["vectorizer"]
    report = dict(bundle.get("report") or {})
    pair_params = dict(report.get("pair_params") or {})
    if not pair_params:
        pair_params = {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.05,
            "colsample_bytree": 0.9,
        }

    X_rows = []
    y_rows: list[str] = []
    labs: list[str] = []
    for path in train_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            circuit_name, fault_type, measured, golden = _parse_instruct_row(row)
            if fault_type not in PAIR_MC_PO:
                continue
            feats = build_feature_dict_from_measurements(
                circuit_name=circuit_name,
                measured=measured,
                golden=golden,
                sim_success=True,
            )
            X_rows.append(feats)
            y_rows.append(fault_type)
            labs.append(circuit_name)

    if not X_rows:
        joblib.dump({}, out_path)
        return 0

    X = vectorizer.transform(X_rows)
    y = np.array([1 if y == "pin_open" else 0 for y in y_rows], dtype=int)
    fam_counts = Counter(family_id(l) for l in labs)

    models: dict[str, XGBClassifier] = {}
    for fam, cnt in fam_counts.items():
        if cnt < 80:
            continue
        idx = np.array([family_id(l) == fam for l in labs], dtype=bool)
        if int(idx.sum()) < 40:
            continue
        y_f = y[idx]
        if len(set(y_f.tolist())) < 2:
            continue

        model = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            random_state=42,
            n_jobs=4,
            subsample=1.0,
            n_estimators=max(200, int(pair_params.get("n_estimators", 400)) // 2),
            max_depth=max(3, int(pair_params.get("max_depth", 3)) - 1),
            learning_rate=max(0.02, float(pair_params.get("learning_rate", 0.05))),
            colsample_bytree=min(1.0, float(pair_params.get("colsample_bytree", 0.9))),
        )
        model.fit(X[idx], y_f)
        models[fam] = model

    joblib.dump(models, out_path)
    return len(models)


def _collect_sorted_fault_classes(train_files: list[Path]) -> list[str]:
    labels: set[str] = set()
    for path in train_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            _, fault_type, _, _ = _parse_instruct_row(row)
            labels.add(fault_type)
    return sorted(labels)


def main() -> int:
    args = parse_args()
    api_dir = args.api_dir
    assets_dir = api_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    if not args.source_bundle.exists():
        raise FileNotFoundError(f"Missing source bundle: {args.source_bundle}")

    model_bundle_dest = assets_dir / "model_bundle.joblib"
    shutil.copy2(args.source_bundle, model_bundle_dest)

    catalog = build_circuit_catalog(args.golden_root)
    catalog_path = assets_dir / "circuit_catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    config = {
        "pair_classes": list(PAIR_MC_PO),
        "pair_threshold": float(joblib.load(model_bundle_dest).get("report", {}).get("pair_threshold", 0.5)),
        "source_bundle": str(args.source_bundle).replace("/", "\\"),
        "golden_root": str(args.golden_root).replace("/", "\\"),
        "family_pair_models_enabled": not args.skip_family_pair_models,
    }

    family_model_count = 0
    family_pair_models_path = assets_dir / "family_pair_models.joblib"
    train_files = [Path(p) for p in (args.train_files or DEFAULT_TRAIN_FILES)]
    config["class_names_sorted"] = _collect_sorted_fault_classes(train_files)
    if not args.skip_family_pair_models:
        missing = [str(p) for p in train_files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing train files for family pair models: {missing}")
        family_model_count = _train_family_pair_models(
            bundle_path=model_bundle_dest,
            train_files=train_files,
            out_path=family_pair_models_path,
        )
    else:
        joblib.dump({}, family_pair_models_path)

    config["family_pair_model_count"] = family_model_count
    config["train_files"] = [str(p).replace("/", "\\") for p in train_files]
    (assets_dir / "runtime_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Wrote: {model_bundle_dest}")
    print(f"Wrote: {catalog_path} (circuits={catalog['circuit_count']})")
    print(f"Wrote: {family_pair_models_path} (family_models={family_model_count})")
    print(f"Wrote: {assets_dir / 'runtime_config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
