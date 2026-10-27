# Download EuRoC MAV sequences for Phase 1 (machine hall + vicon room subsets)
$ErrorActionPreference = "Stop"

$IpsRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$EurocRoot = Join-Path $IpsRoot "third_party\VIFT\data\euroc"
New-Item -ItemType Directory -Force -Path $EurocRoot | Out-Null

# ASL EuRoC dataset base URLs
$Base = "https://projects.asl.ethz.ch/datasets/datasets/euroc"

$Sequences = @(
    @{ Name = "MH_01_easy"; Url = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip" },
    @{ Name = "MH_02_easy"; Url = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip" },
    @{ Name = "MH_05_difficult"; Url = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_05_difficult/MH_05_difficult.zip" },
    @{ Name = "V1_01_easy"; Url = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip" },
    @{ Name = "V1_02_medium"; Url = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip" },
    @{ Name = "V2_02_medium"; Url = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_02_medium/V2_02_medium.zip" }
)

foreach ($seq in $Sequences) {
    $dest = Join-Path $EurocRoot $seq.Name
    if (Test-Path (Join-Path $dest "cam0\data.csv")) {
        Write-Host "Skip $($seq.Name) (already present)"
        continue
    }
    $zip = Join-Path $EurocRoot "$($seq.Name).zip"
    Write-Host "Downloading $($seq.Name)..."
    Invoke-WebRequest -Uri $seq.Url -OutFile $zip -UseBasicParsing
    Write-Host "Extracting $($seq.Name)..."
    Expand-Archive -Path $zip -DestinationPath $EurocRoot -Force
    Remove-Item $zip
}

Write-Host ""
Write-Host "EuRoC ready under: $EurocRoot"
Write-Host "Next: py scripts\phase1\verify_gt_ate.py"
