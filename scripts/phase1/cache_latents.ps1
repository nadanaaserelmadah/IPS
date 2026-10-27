# Cache EuRoC latents (pretrained encoder + GPU recommended)
$ErrorActionPreference = "Stop"

$IpsRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$ViftRoot = Join-Path $IpsRoot "third_party\VIFT"
$VenvPython = Join-Path $ViftRoot ".venv\Scripts\python.exe"
$Pretrained = Join-Path $ViftRoot "pretrained_models\vf_512_if_256_3e-05.model"

if (-not (Test-Path $Pretrained)) {
    Write-Error "Missing pretrained model. See docs/SETUP.md"
}

Push-Location (Join-Path $ViftRoot "data")
try {
    $env:PROJECT_ROOT = $ViftRoot
    & $VenvPython latent_euroc_caching.py
    & $VenvPython latent_euroc_val_caching.py
} finally {
    Pop-Location
}

Write-Host "Latents saved under third_party\VIFT\data\euroc_latent_data\"
