# Circuit Debug API

FastAPI wrapper for your LTSpice-trained circuit fault system.

This folder is now packaged so it can run on its own after you upload just `circuit_debug_api/`.
The only major dependency that is not vendored into Git is the Qwen base model itself. By default
the API will download/load `Qwen/Qwen2.5-1.5B-Instruct` from Hugging Face. If you want to use a
local snapshot instead, set:

```powershell
$env:CIRCUIT_DEBUG_BASE_MODEL = "C:\\path\\to\\Qwen2.5-1.5B-Instruct"
```

Default backend:

- **LLM + KNN hybrid** (Qwen LoRA adapter + KNN class priors)
- Uses the same hybrid scoring path as `pipeline/test_lora_model.py` (`score_classes_knn`)

Fallback backend (if hybrid assets are not present):

- Tabular XGBoost classifier

Endpoints:

- `POST /chat` : ask lab-manual questions and let the server infer or auto-select the lab context
- `POST /chat/{lab_number}` : ask lab-manual questions for a specific lab
- `GET /circuits` : list all supported (golden) circuit names
- `GET /circuits/{circuit_name}/nodes` : list required node names (plus optional source current names)
- `POST /debug` : submit measured values and receive a predicted fault class + diagnosis/fix text

## Directory Contents

- `server.py` : FastAPI app and endpoints
- `runtime.py` : model loading, feature engineering, inference logic
- `build_runtime_assets.py` : packages tabular model + catalog assets
- `build_hybrid_assets.py` : packages LoRA adapter + KNN reference/index assets for hybrid API mode
- `client_example.py` : example client hitting all endpoints
- `chat_terminal_client.py` : interactive terminal chat client for `POST /chat` or `POST /chat/{lab_number}`
- `student_interactive_client.py` : interactive terminal client that prompts for node values one at a time
- `test_chat_endpoint.py` : Python smoke test for `POST /chat`
- `smoke_test_api.ps1` : PowerShell smoke test for API startup + `/debug` client flow
- `demo_payloads/` : ready-to-submit real simulated measurement payloads (for reproducible demos)
- `requirements.txt` : Python deps for API + client
- `make_venv.sh` : Bash script to create `.venv312` and install deps
- `run_api.ps1` : PowerShell start script (Windows)
- `assets/` : tabular assets + circuit catalog
- `assets_hybrid/` : hybrid assets (LoRA adapter copy, KNN ref/index, hybrid config)
- `packaged_golden_root/` : local copy of the golden measurement files used by the packaged catalog
- `packaged_reports/` : local copy of the selected best-model eval report used for auto-pick metadata

## Install Dependencies

```powershell
.\\.venv312\\Scripts\\python.exe -m pip install -r .\\requirements.txt
```

## Build Runtime Assets (one-time or after model updates)

The default rebuild path is now self-contained and uses the packaged files already inside
`circuit_debug_api/`.

```powershell
.\\.venv312\\Scripts\\python.exe .\\build_runtime_assets.py
```

## Build Hybrid Assets (LLM + KNN)

This copies the selected LoRA adapter into the API directory and prebuilds a KNN index from the training instruct JSONL.
This `LLM/` demo folder already includes prebuilt hybrid assets in `assets_hybrid/`, so you can skip this step.

The builder now defaults to the packaged adapter, packaged KNN reference file, and packaged report
inside `circuit_debug_api/`. It does not need the rest of the repo for a normal rebuild.

```powershell
.\\.venv312\\Scripts\\python.exe .\\build_hybrid_assets.py
```

Force a specific adapter (disable auto-pick):

```powershell
.\\.venv312\\Scripts\\python.exe .\\circuit_debug_api\\build_hybrid_assets.py `
  --auto-pick-best False `
  --adapter-dir .\\assets_hybrid\\adapter
```

## Run the API

```powershell
.\\.venv312\\Scripts\\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Or (for a fresh best-model selection from the latest reports):

```powershell
powershell -ExecutionPolicy Bypass -File .\\circuit_debug_api\\run_api.ps1 -RefreshModel
```

`run_api.ps1` will auto-build both tabular and hybrid assets if they are missing and uses auto-pick for the best hybrid model.

## Bash / Linux/macOS Quickstart

If you are using Bash (macOS/Linux, or Git Bash/WSL), these commands assume you already activated the venv:
`source .venv312/bin/activate`

Create the virtual environment (recommended):

```bash
./make_venv.sh
```

Install dependencies:

```bash
python -m pip install -r ./requirements.txt
```

Build runtime assets:

```bash
python ./build_runtime_assets.py
```

Skip this step if `./assets/model_bundle.joblib` already exists (it does in this demo folder).

Build hybrid assets:

```bash
python ./build_hybrid_assets.py
```

Skip this step if `./assets_hybrid/hybrid_config.json` already exists (it does in this demo folder).

Run the API:

```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Example client:

```bash
python ./client_example.py --demo-use-golden-values --demo-offset-node N001 --demo-offset-volts 0.5
```

Interactive student client:

```bash
python ./student_interactive_client.py
```

Interactive chat client:

```bash
python ./chat_terminal_client.py --base-url http://127.0.0.1:8000
```

Query endpoints from Bash:

```bash
curl -s http://127.0.0.1:8000/circuits | jq
curl -s http://127.0.0.1:8000/circuits/Lab9_2/nodes | jq
```

Submit a JSON payload from Bash:

```bash
curl -s http://127.0.0.1:8000/debug \
  -H 'Content-Type: application/json' \
  --data @./student_lab9_2_payload.json | jq
```

## Example Client (uses all endpoints)

```powershell
.\\.venv312\\Scripts\\python.exe .\\client_example.py --demo-use-golden-values --demo-offset-node N001 --demo-offset-volts 0.5
```

## Interactive Student Client (one measurement at a time)

This client first prompts the student to choose a lab, then a circuit within that lab, then prompts for measurements one by one.

```powershell
.\\.venv312\\Scripts\\python.exe .\\student_interactive_client.py
```

Flow in terminal:

1. Choose lab (example: `Lab4` or `4`)
2. Choose circuit from that lab
3. Enter each node voltage one at a time
4. Optionally enter source currents
5. Submit and receive diagnosis

Optional flags:

- `--circuit Lab9_2` : skip circuit selection prompt
- `--ask-source-currents` : also prompt for optional source currents
- `--save-payload .\\student_payload.json` : save the submitted payload
- `--show-golden` : instructor/demo mode only
- `--no-strict` : allow missing nodes (not recommended)

## Interactive Chat Client (terminal Q&A)

This client lets a student type a question in the terminal and prints the chat answer. In interactive mode it prompts for an optional lab number once; if you skip it, the server will infer or auto-select the lab.

```powershell
.\\.venv312\\Scripts\\python.exe .\\chat_terminal_client.py --base-url http://127.0.0.1:8000
```

Optional one-shot question:

```powershell
.\\.venv312\\Scripts\\python.exe .\\chat_terminal_client.py `
  --base-url http://127.0.0.1:8000 `
  --lab-number 1 `
  --question "What does Lab 1 procedure require?"
```

Exit commands in interactive mode: `/quit`, `/exit`.

## Chat Endpoint Test (server.py)

Run the chat endpoint smoke test:

```powershell
.\\.venv312\\Scripts\\python.exe .\\test_chat_endpoint.py --base-url http://127.0.0.1:8000
```

Strict mode (require valid-question call to return `200` with `answer`):

```powershell
.\\.venv312\\Scripts\\python.exe .\\test_chat_endpoint.py `
  --base-url http://127.0.0.1:8000 `
  --require-answer
```

This test checks:

- `POST /chat` exists and uses a JSON request body
- valid `{"question":"..."}` request
- empty-question behavior
- missing required `question` field (`422`)

## Full API Smoke Test (PowerShell)

`smoke_test_api.ps1` starts `uvicorn`, waits for `/health`, runs `client_example.py`, and prints server log tails:

```powershell
powershell -ExecutionPolicy Bypass -File .\\smoke_test_api.ps1
```

## Specific Circuit Demo (real simulated case)

This uses a real simulated variant from `Lab9_2` (`Lab9_2__v0022`) and submits the measured node voltages/source currents to the API.

```powershell
.\\.venv312\\Scripts\\python.exe .\\client_example.py `
  --payload-file .\\demo_payloads\\Lab9_2__v0022_request.json
```

Reference metadata / expected injected fault for that demo case:

- `demo_payloads/Lab9_2__v0022_expected.json`

## Additional Real Demo Payloads (held-out simulated eval rows)

These are extra reproducible demos extracted from held-out simulated eval rows and aligned to a saved hybrid eval run.

- Index: `demo_payloads/demo_index.json`
- Each demo has:
  - `*_request.json` (send to `POST /debug`)
  - `*_expected.json` (saved target label / provenance)

Included classes:

- `param_drift`
- `resistor_value_swap`
- `resistor_wrong_value`
- `missing_component`
- `short_between_nodes`
- `swapped_nodes`
- `pin_open`

Run any one demo:

```powershell
.\\.venv312\\Scripts\\python.exe .\\client_example.py `
  --payload-file .\\demo_payloads\\evalrow_0001__Lab1_2A_2_0__param_drift_request.json
```

Run another demo:

```powershell
.\\.venv312\\Scripts\\python.exe .\\client_example.py `
  --payload-file .\\demo_payloads\\evalrow_0049__lab4_task2_part1_-3__short_between_nodes_request.json
```

## Student Breadboard Workflow (exact endpoint flow)

This is the intended real use path when a student has breadboard measurements.

### 1) Start the API

```powershell
.\\.venv312\\Scripts\\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

### 2) Get the valid circuit names (pick the golden circuit the student is building)

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/circuits | ConvertTo-Json -Depth 5
```

### 3) Get the exact node names required for that circuit

Example for `Lab9_2`:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/circuits/Lab9_2/nodes | ConvertTo-Json -Depth 10
```

Use the returned `nodes[].node_name` list as the measurement checklist.

### 4) Take measurements on the breadboard (what to enter)

- Measure each listed node voltage relative to the breadboard ground node.
- Enter values using the exact node names from the API.
- If the circuit is time-varying, this API currently expects the same convention as training: `*_max` features (peak/max values).
- `source_currents` are optional, but if you can measure them they usually improve accuracy.
- Keep `temp` and `tnom` at defaults unless you intentionally changed them.

### 5) Fill a payload JSON file with the student measurements

Example (`student_lab9_2_payload.json`):

```json
{
  "circuit_name": "Lab9_2",
  "node_voltages": {
    "N001": 0.95,
    "N002": 0.18,
    "N003": 5.0,
    "N004": 1.0,
    "N005": 0.18,
    "N006": -5.0
  },
  "source_currents": {},
  "temp": 27.0,
  "tnom": 27.0,
  "strict": true
}
```

Notes:

- `strict: true` will fail if any required node is missing (recommended for students).
- If you do not know source currents, leave `source_currents` as `{}`.

### 6) Submit the measurements for debugging

Using the example client:

```powershell
.\\.venv312\\Scripts\\python.exe .\\client_example.py `
  --payload-file .\\student_lab9_2_payload.json
```

Using the interactive student client (recommended for manual breadboard entry):

```powershell
.\\.venv312\\Scripts\\python.exe .\\student_interactive_client.py `
  --circuit Lab9_2 `
  --ask-source-currents `
  --save-payload .\\student_lab9_2_payload.json
```

Or directly with PowerShell:

```powershell
$body = Get-Content .\\student_lab9_2_payload.json -Raw
Invoke-RestMethod -Uri http://127.0.0.1:8000/debug -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 10
```

### 7) Read the response

Main fields:

- `fault_type` : predicted fault class
- `confidence` : model confidence (not a guarantee)
- `diagnosis` / `fix` : human-readable guidance
- `missing_required_nodes` : nodes you forgot to provide (when `strict=false`)

Notes:

- Demo mode is only to show endpoint usage. Exact golden values are not a real fault case.
- `--payload-file` mode is the recommended way to demo a specific real simulated case.
- For best accuracy, provide all nodes from `GET /circuits/{name}/nodes`.
- Supplying source currents (if available) improves accuracy.
- `GET /health` reports which backend is active (`llm_knn_hybrid` or `tabular_xgboost`).
- `GET /model` returns the exact selected adapter paths and eval report metrics used to pick the model.

To force a fresh best-model selection and reload after new training/eval artifacts are produced:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/admin/refresh-model -Method Post | ConvertTo-Json -Depth 12
```

Or restart with refresh enabled:

```powershell
powershell -ExecutionPolicy Bypass -File .\\circuit_debug_api\\run_api.ps1 -RefreshModel
```

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
  "strict": true
}
```
