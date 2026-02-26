$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Resolve-Path (Join-Path $Root ".."))

$py = ".\\.venv312\\Scripts\\python.exe"
if (-not (Test-Path $py)) {
  throw "Missing venv python: $py"
}

$logDir = ".\\circuit_debug_api\\_smoke_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "uvicorn.out.log"
$stderr = Join-Path $logDir "uvicorn.err.log"
if (Test-Path $stdout) { Remove-Item $stdout -Force }
if (Test-Path $stderr) { Remove-Item $stderr -Force }

$p = Start-Process -FilePath $py `
  -ArgumentList @("-m","uvicorn","circuit_debug_api.server:app","--host","127.0.0.1","--port","8000") `
  -WorkingDirectory (Get-Location).Path `
  -PassThru `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr

try {
  $ready = $false
  for ($i=0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 1
    try {
      $h = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5
      $ready = $true
      Write-Host "HEALTH:"
      $h | ConvertTo-Json -Depth 8
      break
    } catch {
      if ($p.HasExited) {
        throw "uvicorn exited early with code $($p.ExitCode)"
      }
    }
  }

  if (-not $ready) {
    throw "API did not become ready in time."
  }

  & $py .\circuit_debug_api\client_example.py `
    --base-url http://127.0.0.1:8000 `
    --circuit Lab1_1_0 `
    --demo-use-golden-values `
    --demo-offset-node N001 `
    --demo-offset-volts 0.5
  $clientExit = $LASTEXITCODE
  Write-Host "CLIENT_EXIT=$clientExit"

  Write-Host "SERVER_STDOUT_TAIL:"
  if (Test-Path $stdout) { Get-Content $stdout -Tail 80 }
  Write-Host "SERVER_STDERR_TAIL:"
  if (Test-Path $stderr) { Get-Content $stderr -Tail 120 }

  if ($clientExit -ne 0) {
    throw "Client example failed with exit code $clientExit"
  }
}
finally {
  if ($p -and -not $p.HasExited) {
    Stop-Process -Id $p.Id -Force
    Start-Sleep -Milliseconds 500
  }
}

