# Phase 0: Cache KITTI latents (requires KITTI + pretrained model + CUDA)
$ErrorActionPreference = "Stop"

$IpsRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$ViftRoot = Join-Path $IpsRoot "third_party\VIFT"
$VenvPython = Join-Path $ViftRoot ".venv\Scripts\python.exe"
$Pretrained = Join-Path $ViftRoot "pretrained_models\vf_512_if_256_3e-05.model"
$KittiSeq = Join-Path $ViftRoot "data\kitti_data\sequences\00"

if (-not (Test-Path $VenvPython)) { Write-Error "Run setup_env.ps1 first." }
if (-not (Test-Path $Pretrained)) { Write-Error "Missing: $Pretrained" }
if (-not (Test-Path $KittiSeq)) { Write-Error "Missing KITTI. Run download_kitti.ps1" }

Push-Location (Join-Path $ViftRoot "data")
try {
    $env:PROJECT_ROOT = $ViftRoot
    Write-Host "Caching training latents (GPU)..."
    & $VenvPython latent_caching.py
    Write-Host "Caching validation latents..."
    & $VenvPython latent_val_caching.py
    Write-Host "Done. Check data\kitti_latent_data\"
} finally {
    Pop-Location
}
