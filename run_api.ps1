$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Resolve-Path (Join-Path $Root ".."))

$Py = ".\\.venv312\\Scripts\\python.exe"
if (-not (Test-Path $Py)) {
  $Py = "python"
}

if (-not (Test-Path ".\\circuit_debug_api\\assets\\model_bundle.joblib")) {
  Write-Host "Assets not found. Building runtime assets..."
  & $Py .\\circuit_debug_api\\build_runtime_assets.py
}

# Build hybrid assets (LoRA adapter + KNN reference/index) if missing.
if (-not (Test-Path ".\\circuit_debug_api\\assets_hybrid\\hybrid_config.json")) {
  Write-Host "Hybrid assets not found. Building hybrid assets..."
  & $Py .\\circuit_debug_api\\build_hybrid_assets.py
}

& $Py -m uvicorn circuit_debug_api.server:app --host 127.0.0.1 --port 8000
