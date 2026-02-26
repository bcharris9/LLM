# Circuit Debug API

FastAPI wrapper for your LTSpice-trained circuit fault system.

Default backend:

- **LLM + KNN hybrid** (Qwen LoRA adapter + KNN class priors)
- Uses the same hybrid scoring path as `pipeline/test_lora_model.py` (`score_classes_knn`)

Fallback backend (if hybrid assets are not present):

- Tabular XGBoost classifier

Endpoints:

- `GET /circuits` : list all supported (golden) circuit names
- `GET /circuits/{circuit_name}/nodes` : list required node names (plus optional source current names)
- `POST /debug` : submit measured values and receive a predicted fault class + diagnosis/fix text

## Directory Contents

- `server.py` : FastAPI app and endpoints
- `runtime.py` : model loading, feature engineering, inference logic
- `build_runtime_assets.py` : packages tabular model + catalog assets
- `build_hybrid_assets.py` : packages LoRA adapter + KNN reference/index assets for hybrid API mode
- `client_example.py` : example client hitting all endpoints
- `requirements.txt` : Python deps for API + client
- `run_api.ps1` : PowerShell start script
- `assets/` : tabular assets + circuit catalog
- `assets_hybrid/` : hybrid assets (LoRA adapter copy, KNN ref/index, hybrid config)

## Install Dependencies

```powershell
.\\.venv312\\Scripts\\python.exe -m pip install -r .\\circuit_debug_api\\requirements.txt
```

## Build Runtime Assets (one-time or after model updates)

```powershell
.\\.venv312\\Scripts\\python.exe .\\circuit_debug_api\\build_runtime_assets.py
```

## Build Hybrid Assets (LLM + KNN)

This copies the selected LoRA adapter into the API directory and prebuilds a KNN index from the training instruct JSONL.

```powershell
.\\.venv312\\Scripts\\python.exe .\\circuit_debug_api\\build_hybrid_assets.py
```

## Run the API

```powershell
.\\.venv312\\Scripts\\python.exe -m uvicorn circuit_debug_api.server:app --host 127.0.0.1 --port 8000
```

Or:

```powershell
powershell -ExecutionPolicy Bypass -File .\\circuit_debug_api\\run_api.ps1
```

`run_api.ps1` will auto-build both tabular and hybrid assets if they are missing.

## Example Client (uses all endpoints)

```powershell
.\\.venv312\\Scripts\\python.exe .\\circuit_debug_api\\client_example.py --demo-use-golden-values --demo-offset-node N001 --demo-offset-volts 0.5
```

Notes:

- Demo mode is only to show endpoint usage. Exact golden values are not a real fault case.
- For best accuracy, provide all nodes from `GET /circuits/{name}/nodes`.
- Supplying source currents (if available) improves accuracy.
- `GET /health` reports which backend is active (`llm_knn_hybrid` or `tabular_xgboost`).

## POST /debug Request Shape

```json
{
  "circuit_name": "Lab1_1_0",
  "node_voltages": {
    "N001": 9.0,
    "N002": 5.0
  },
  "source_currents": {
    "V1": -0.00185
  },
  "strict": false
}
```
