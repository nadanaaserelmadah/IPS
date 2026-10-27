# Test EuRoC dataloader (requires downloaded data)
$ErrorActionPreference = "Stop"

$IpsRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$ViftRoot = Join-Path $IpsRoot "third_party\VIFT"
$VenvPython = Join-Path $ViftRoot ".venv\Scripts\python.exe"
$EurocRoot = Join-Path $ViftRoot "data\euroc"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Run phase0 setup_env.ps1 first."
}
if (-not (Test-Path (Join-Path $EurocRoot "MH_01_easy\cam0\data.csv"))) {
    Write-Error "EuRoC not found. Run download_euroc.ps1 first."
}

Push-Location $ViftRoot
try {
    $env:PROJECT_ROOT = $ViftRoot
    & $VenvPython -c @"
from src.data.components.EuRoC_dataset import EuRoC
from src.utils import custom_transform
t = custom_transform.Compose([
    custom_transform.ToTensor(),
    custom_transform.Resize((256, 512)),
])
ds = EuRoC(root='data/euroc', sequences=['MH_01_easy'], transform=t, sequence_length=11)
print(ds)
(x, y) = ds[0]
imgs, imus, rot, w = x
print('imgs', imgs.shape, 'imus', imus.shape, 'gts', y.shape, 'rot', rot)
print('OK dataloader sample 0')
"@
} finally {
    Pop-Location
}
