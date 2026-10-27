# Download KITTI IMU .mat files (Visual-Selective-VIO) into VIFT data tree
$ErrorActionPreference = "Stop"
$IpsRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Python = Join-Path $IpsRoot "third_party\VIFT\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

& $Python (Join-Path $PSScriptRoot "download_kitti_imus.py") @args
