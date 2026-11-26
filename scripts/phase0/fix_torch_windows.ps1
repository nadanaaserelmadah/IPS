# Fix PyTorch WinError 126 on Windows (missing DLL dependencies)
$ErrorActionPreference = "Stop"

$VenvPython = "$PSScriptRoot\..\..\third_party\VIFT\.venv\Scripts\python.exe"
$VenvPip = "$PSScriptRoot\..\..\third_party\VIFT\.venv\Scripts\pip.exe"

Write-Host "=== PyTorch Windows repair ==="
Write-Host ""
Write-Host "1. Install Microsoft Visual C++ 2015-2022 Redistributable (x64):"
Write-Host "   https://aka.ms/vs/17/release/vc_redist.x64.exe"
Write-Host ""
Write-Host "2. Reinstall CPU PyTorch wheels..."
& $VenvPip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Host ""
Write-Host "3. Testing import..."
& $VenvPython -c "import torch; print('OK torch', torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Still failing. Try:"
    Write-Host "  - Reboot after installing VC++ redistributable"
    Write-Host "  - Use Python 3.11 venv instead of 3.13"
    Write-Host "  - Install via conda: conda install pytorch torchvision cpuonly -c pytorch"
    exit 1
}
Write-Host "PyTorch is working."
