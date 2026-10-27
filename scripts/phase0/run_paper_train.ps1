# Phase 0: Train VIFT paper config (requires latents + GPU recommended)
$ErrorActionPreference = "Stop"

param(
    [string]$Trainer = "gpu",
    [int]$MaxEpochs = 0  # 0 = use yaml default (200)
)

$IpsRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$ViftRoot = Join-Path $IpsRoot "third_party\VIFT"
$VenvPython = Join-Path $ViftRoot ".venv\Scripts\python.exe"
$Pretrained = Join-Path $ViftRoot "pretrained_models\vf_512_if_256_3e-05.model"
$LatentTrain = Join-Path $ViftRoot "data\kitti_latent_data\train_10"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Run setup_env.ps1 first."
}
if (-not (Test-Path $Pretrained)) {
    Write-Error "Missing pretrained model: $Pretrained`nDownload from Visual-Selective-VIO Google Drive (see docs\SETUP.md)."
}
if (-not (Test-Path $LatentTrain)) {
    Write-Error "Missing latent data: $LatentTrain`nRun latent_caching.py after KITTI download."
}

Push-Location $ViftRoot
try {
    $env:PROJECT_ROOT = $ViftRoot
    $args = @(
        "src\train.py",
        "experiment=latent_kitti_vio_paper",
        "trainer=$Trainer"
    )
    if ($MaxEpochs -gt 0) {
        $args += "trainer.max_epochs=$MaxEpochs"
    }
    Write-Host "Running: python $($args -join ' ')"
    & $VenvPython @args
} finally {
    Pop-Location
}
