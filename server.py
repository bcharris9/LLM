from __future__ import annotations

import os
import subprocess
import json
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from circuit_debug_api.hybrid_runtime import CircuitDebugHybridRuntime
from circuit_debug_api.runtime import CircuitDebugRuntime


API_DIR = Path(__file__).resolve().parent
ASSETS_DIR = API_DIR / "assets"
HYBRID_DIR = API_DIR / "assets_hybrid"
HYBRID_CONFIG_PATH = HYBRID_DIR / "hybrid_config.json"
PACKAGED_GOLDEN_ROOT = API_DIR / "packaged_golden_root"


class DebugRequest(BaseModel):
    circuit_name: str = Field(..., description="Exact circuit name from GET /circuits")
    node_voltages: dict[str, float] = Field(
        default_factory=dict,
        description="Map of node name -> measured voltage (V). Example keys: N001, N002, VCC, -VCC, VOUT.",
    )
    source_currents: dict[str, float] = Field(
        default_factory=dict,
        description="Optional map of voltage source name -> measured current (A). Example key: V1.",
    )
    measurement_overrides: dict[str, float] = Field(
        default_factory=dict,
        description="Advanced: direct measurement_key -> value overrides (e.g. v_n001_max).",
    )
    temp: float | None = Field(default=27.0, description="Temperature feature (degC).")
    tnom: float | None = Field(default=27.0, description="Nominal temperature feature (degC).")
    strict: bool = Field(default=True, description="Fail if not all listed nodes are provided.")


class HealthResponse(BaseModel):
    ok: bool
    backend: str
    circuits: int
    family_pair_models: int
    pair_threshold: float
    selected_model_report: str | None = None
    selected_model_metric: str | None = None
    selected_model_metric_value: float | None = None


class ModelInfoResponse(BaseModel):
    ok: bool
    backend: str
    adapter_dir: str | None = None
    knn_ref_file: str | None = None
    knn_index_file: str | None = None
    model_name: str
    selected_model_report: str | None = None
    selected_model_metric: str | None = None
    selected_model_metric_value: float | None = None


def _load_hybrid_config() -> dict[str, Any] | None:
    if not HYBRID_CONFIG_PATH.exists():
        return None
    return json.loads(HYBRID_CONFIG_PATH.read_text(encoding="utf-8"))


def _refresh_hybrid_assets() -> None:
    builder = API_DIR / "build_hybrid_assets.py"
    if not builder.exists():
        raise FileNotFoundError(f"Missing builder: {builder}")

    cfg = _load_hybrid_config() or {}
    bool_flags = {
        "include-lab-id-in-prompt": cfg.get("include_lab_id_in_prompt", False),
        "prefer-voltage-keys": cfg.get("prefer_voltage_keys", True),
        "voltage-only": cfg.get("voltage_only", False),
        "knn-weighted-vote": cfg.get("knn_weighted_vote", True),
        "knn-standardize": cfg.get("knn_standardize", False),
        "use-prerules": cfg.get("use_prerules", True),
    }

    cmd = [sys.executable, str(builder)]
    if "measurement_stat_mode" in cfg:
        cmd += ["--measurement-stat-mode", str(cfg["measurement_stat_mode"])]
    if "response_style" in cfg:
        cmd += ["--response-style", str(cfg["response_style"])]
    if "knn_k" in cfg:
        cmd += ["--knn-k", str(int(cfg["knn_k"]))]
    if "knn_alpha" in cfg:
        cmd += ["--knn-alpha", str(float(cfg["knn_alpha"]))]
    if "knn_eps" in cfg:
        cmd += ["--knn-eps", str(float(cfg["knn_eps"]))]
    if "knn_weighted_vote" in cfg:
        cmd += ["--knn-weighted-vote" if bool_flags["knn-weighted-vote"] else "--no-knn-weighted-vote"]
    if "knn_standardize" in cfg:
        cmd += ["--knn-standardize" if bool_flags["knn-standardize"] else "--no-knn-standardize"]
    if "use_prerules" in cfg:
        cmd += ["--use-prerules" if bool_flags["use-prerules"] else "--no-use-prerules"]
    if "include_lab_id_in_prompt" in cfg:
        cmd += [
            "--include-lab-id-in-prompt"
            if bool_flags["include-lab-id-in-prompt"]
            else "--no-include-lab-id-in-prompt"
        ]
    if "prefer_voltage_keys" in cfg:
        cmd += [
            "--prefer-voltage-keys"
            if bool_flags["prefer-voltage-keys"]
            else "--no-prefer-voltage-keys"
        ]
    if "voltage_only" in cfg:
        cmd += ["--voltage-only" if bool_flags["voltage-only"] else "--no-voltage-only"]
    if "max_measurements" in cfg:
        cmd += ["--max-measurements", str(int(cfg["max_measurements"]))]
    if "max_deltas" in cfg:
        cmd += ["--max-deltas", str(int(cfg["max_deltas"]))]

    subprocess.run(cmd, check=True, cwd=str(API_DIR.parent), capture_output=True, text=True)


@lru_cache(maxsize=1)
def get_runtime() -> Any:
    hybrid_dir = HYBRID_DIR
    hybrid_cfg = HYBRID_CONFIG_PATH
    if hybrid_cfg.exists():
        return CircuitDebugHybridRuntime(
            catalog_path=ASSETS_DIR / "circuit_catalog.json",
            hybrid_assets_dir=hybrid_dir,
            auto_build_catalog_from=PACKAGED_GOLDEN_ROOT if PACKAGED_GOLDEN_ROOT.exists() else None,
        )
    return CircuitDebugRuntime(
        model_bundle_path=ASSETS_DIR / "model_bundle.joblib",
        circuit_catalog_path=ASSETS_DIR / "circuit_catalog.json",
        family_pair_models_path=ASSETS_DIR / "family_pair_models.joblib",
        config_path=ASSETS_DIR / "runtime_config.json",
    )


app = FastAPI(
    title="Circuit Debug API",
    version="1.0.0",
    description=(
        "FastAPI wrapper for the LTSpice-trained circuit fault classifier. "
        "Provides circuit catalog, node schema, and debugging inference from measured node voltages/currents."
    ),
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    rt = get_runtime()
    selected_report = None
    selected_metric = None
    selected_metric_value = None
    if isinstance(rt, CircuitDebugHybridRuntime):
        cfg = _load_hybrid_config()
        if cfg:
            selected_report = cfg.get("auto_selected_report")
            selected_metric = cfg.get("auto_selected_metric")
            selected_metric_value = cfg.get("auto_selected_metric_value")
    backend = "llm_knn_hybrid" if rt.__class__.__name__.endswith("HybridRuntime") else "tabular_xgboost"
    return HealthResponse(
        ok=True,
        backend=backend,
        circuits=len(rt.list_circuits()),
        family_pair_models=len(rt.family_pair_models),
        pair_threshold=float(rt.pair_threshold),
        selected_model_report=selected_report,
        selected_model_metric=selected_metric,
        selected_model_metric_value=selected_metric_value,
    )


@app.get("/model", response_model=ModelInfoResponse)
def model() -> ModelInfoResponse:
    rt = get_runtime()
    backend = "llm_knn_hybrid" if rt.__class__.__name__.endswith("HybridRuntime") else "tabular_xgboost"
    if backend != "llm_knn_hybrid":
        return ModelInfoResponse(
            ok=True,
            backend=backend,
            adapter_dir=None,
            knn_ref_file=None,
            knn_index_file=None,
            model_name="N/A",
        )

    cfg = _load_hybrid_config() or {}
    effective_model_name = str(os.environ.get("CIRCUIT_DEBUG_BASE_MODEL", cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")))
    return ModelInfoResponse(
        ok=True,
        backend=backend,
        adapter_dir=cfg.get("adapter_dir"),
        knn_ref_file=cfg.get("knn_ref_file"),
        knn_index_file=cfg.get("knn_index_file"),
        model_name=effective_model_name,
        selected_model_report=cfg.get("auto_selected_report"),
        selected_model_metric=cfg.get("auto_selected_metric"),
        selected_model_metric_value=cfg.get("auto_selected_metric_value"),
    )


@app.post("/admin/refresh-model", include_in_schema=True)
def refresh_model() -> dict[str, Any]:
    _refresh_hybrid_assets()
    get_runtime.cache_clear()
    return {
        "ok": True,
        "message": "Hybrid assets rebuilt using best report and runtime reloaded.",
        "model": model().dict(),
    }


@app.get("/circuits")
def list_circuits() -> dict[str, Any]:
    rt = get_runtime()
    names = rt.list_circuits()
    return {"count": len(names), "circuits": names}


@app.get("/circuits/{circuit_name}/nodes")
def get_circuit_nodes(circuit_name: str) -> dict[str, Any]:
    rt = get_runtime()
    if not rt.has_circuit(circuit_name):
        raise HTTPException(status_code=404, detail=f"Unknown circuit: {circuit_name}")
    spec = rt.circuit_spec(circuit_name)
    return {
        "circuit_name": circuit_name,
        "node_count": len(spec.get("nodes", [])),
        "nodes": spec.get("nodes", []),
        "source_current_count": len(spec.get("source_currents", [])),
        "source_currents": spec.get("source_currents", []),
        "golden_defaults": spec.get("golden_defaults", {}),
        "notes": {
            "recommended": "Provide all listed nodes in POST /debug for best accuracy. Source currents are optional but improve accuracy."
        },
    }


@app.post("/debug")
def debug_circuit(req: DebugRequest) -> dict[str, Any]:
    rt = get_runtime()
    if not rt.has_circuit(req.circuit_name):
        raise HTTPException(status_code=404, detail=f"Unknown circuit: {req.circuit_name}")
    try:
        result = rt.predict_fault(
            circuit_name=req.circuit_name,
            node_voltages=req.node_voltages,
            source_currents=req.source_currents,
            measurement_overrides=req.measurement_overrides,
            temp=req.temp,
            tnom=req.tnom,
            strict=req.strict,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
    return result.to_dict()
