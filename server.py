from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from LLM.hybrid_runtime import CircuitDebugHybridRuntime
    from LLM.runtime import CircuitDebugRuntime
except ModuleNotFoundError as e:
    if e.name != "LLM":
        raise
    from hybrid_runtime import CircuitDebugHybridRuntime  # type: ignore[no-redef]
    from runtime import CircuitDebugRuntime  # type: ignore[no-redef]


API_DIR = Path(__file__).resolve().parent
ASSETS_DIR = API_DIR / "assets"


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
    strict: bool = Field(default=False, description="Fail if not all listed nodes are provided.")


class HealthResponse(BaseModel):
    ok: bool
    backend: str
    circuits: int
    family_pair_models: int
    pair_threshold: float


@lru_cache(maxsize=1)
def get_runtime() -> Any:
    hybrid_dir = API_DIR / "assets_hybrid"
    hybrid_cfg = hybrid_dir / "hybrid_config.json"
    if hybrid_cfg.exists():
        return CircuitDebugHybridRuntime(
            catalog_path=ASSETS_DIR / "circuit_catalog.json",
            hybrid_assets_dir=hybrid_dir,
            auto_build_catalog_from=Path("pipeline/out_one_lab_all_v2_train"),
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
    backend = "llm_knn_hybrid" if rt.__class__.__name__.endswith("HybridRuntime") else "tabular_xgboost"
    return HealthResponse(
        ok=True,
        backend=backend,
        circuits=len(rt.list_circuits()),
        family_pair_models=len(rt.family_pair_models),
        pair_threshold=float(rt.pair_threshold),
    )


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
