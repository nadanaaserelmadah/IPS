# Phase 0: Create venv and install VIFT dependencies
$ErrorActionPreference = "Stop"

$IpsRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$ViftRoot = Join-Path $IpsRoot "third_party\VIFT"
$VenvPython = Join-Path $ViftRoot ".venv\Scripts\python.exe"
$Requirements = Join-Path $ViftRoot "requirements.txt"
$EnvReport = Join-Path $IpsRoot "docs\phase0_environment.txt"

if (-not (Test-Path $ViftRoot)) {
    Write-Error "VIFT not found at $ViftRoot. Run download or clone first."
}

# Find Python launcher
$Py = Get-Command py -ErrorAction SilentlyContinue
if (-not $Py) {
    Write-Error "Python launcher 'py' not found. Install Python 3.10+ from python.org"
}

Write-Host "IPS root:  $IpsRoot"
Write-Host "VIFT root: $ViftRoot"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment..."
    & py -3 -m venv (Join-Path $ViftRoot ".venv")
}

Write-Host "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing requirements (this may take several minutes)..."
& $VenvPython -m pip install -r $Requirements

Write-Host "Reinstalling PyTorch CPU wheels (more reliable on Windows)..."
& $VenvPython -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Host "Writing environment report..."
$lines = @(
    "Phase 0 environment report",
    "Generated: $(Get-Date -Format o)",
    "IPS_ROOT=$IpsRoot",
    "VIFT_ROOT=$ViftRoot",
    ""
)
& $VenvPython --version 2>&1 | ForEach-Object { $lines += "Python: $_" }
$lines += ""
foreach ($pkg in @("torch", "torchvision", "lightning", "hydra-core", "omegaconf")) {
    $v = & $VenvPython -m pip show $pkg 2>$null | Select-String "^Version:"
    if ($v) { $lines += "$pkg $($v.ToString().Replace('Version: ', ''))" }
}
try {
    $cuda = & $VenvPython -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>&1
    $lines += $cuda
} catch {
    $lines += "CUDA check failed: $_"
}
$lines | Set-Content -Path $EnvReport -Encoding UTF8

Write-Host ""
Write-Host "Setup complete."
Write-Host "  venv:    $VenvPython"
Write-Host "  report:  $EnvReport"
Write-Host ""
Write-Host "Next: .\scripts\phase0\smoke_test.ps1"
