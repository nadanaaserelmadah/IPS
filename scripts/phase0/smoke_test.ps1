# Phase 0: Quick training smoke test (dummy_vio, no KITTI)
$ErrorActionPreference = "Stop"

$IpsRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$ViftRoot = Join-Path $IpsRoot "third_party\VIFT"
$VenvPython = Join-Path $ViftRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtualenv missing. Run: .\scripts\phase0\setup_env.ps1"
}

Push-Location $ViftRoot
try {
    $env:PROJECT_ROOT = $ViftRoot
    Write-Host "PROJECT_ROOT=$env:PROJECT_ROOT"
    Write-Host "Running dummy_vio smoke test (2 epochs, CPU)..."
    & $VenvPython src\train.py `
        experiment=dummy_vio `
        trainer=cpu `
        trainer.max_epochs=2 `
        trainer.min_epochs=1 `
        data.batch_size=2 `
        test=False
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Smoke test FAILED (exit code $LASTEXITCODE). See docs\SETUP.md troubleshooting (PyTorch WinError 126)."
    }
    Write-Host ""
    Write-Host "Smoke test PASSED."
} finally {
    Pop-Location
}
