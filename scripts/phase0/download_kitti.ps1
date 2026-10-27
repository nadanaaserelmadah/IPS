# Phase 0: Download KITTI Odometry color + poses (large download)
$ErrorActionPreference = "Stop"

$IpsRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$DataDir = Join-Path $IpsRoot "third_party\VIFT\data"
$KittiRoot = Join-Path $DataDir "kitti_data"

New-Item -ItemType Directory -Force -Path $KittiRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $KittiRoot "sequences") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $KittiRoot "poses") | Out-Null

$ColorUrl = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip"
$PosesUrl = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip"
$ColorZip = Join-Path $DataDir "data_odometry_color.zip"
$PosesZip = Join-Path $DataDir "data_odometry_poses.zip"

Write-Host "KITTI download target: $KittiRoot"
Write-Host "WARNING: color zip is ~22 GB. Ensure enough disk space and bandwidth."
Write-Host ""

if (-not (Test-Path $ColorZip)) {
    Write-Host "Downloading color images..."
    Invoke-WebRequest -Uri $ColorUrl -OutFile $ColorZip -UseBasicParsing
} else {
    Write-Host "Color zip already exists, skipping download."
}

if (-not (Test-Path $PosesZip)) {
    Write-Host "Downloading poses..."
    Invoke-WebRequest -Uri $PosesUrl -OutFile $PosesZip -UseBasicParsing
} else {
    Write-Host "Poses zip already exists, skipping download."
}

Write-Host "Extracting color..."
Expand-Archive -Path $ColorZip -DestinationPath $DataDir -Force
# KITTI zip extracts to dataset/sequences
$ExtractedSeq = Join-Path $DataDir "dataset\sequences"
if (Test-Path $ExtractedSeq) {
    Copy-Item -Path "$ExtractedSeq\*" -Destination (Join-Path $KittiRoot "sequences") -Recurse -Force
    Remove-Item -Recurse -Force (Join-Path $DataDir "dataset")
}

Write-Host "Extracting poses..."
Expand-Archive -Path $PosesZip -DestinationPath $DataDir -Force
$ExtractedPoses = Join-Path $DataDir "dataset\poses"
if (Test-Path $ExtractedPoses) {
    Copy-Item -Path "$ExtractedPoses\*" -Destination (Join-Path $KittiRoot "poses") -Force
    Remove-Item -Recurse -Force (Join-Path $DataDir "dataset")
}

# Remove sequences 11-21 per VIFT data_prep.sh
foreach ($i in 11..21) {
    $seqPath = Join-Path $KittiRoot "sequences\$('{0:D2}' -f $i)"
    if (Test-Path $seqPath) {
        Remove-Item -Recurse -Force $seqPath
        Write-Host "Removed sequence $i"
    }
}

Write-Host ""
Write-Host "KITTI layout ready under: $KittiRoot"
Write-Host "Next: download pretrained model, then run latent_caching.py (see docs\SETUP.md)"
