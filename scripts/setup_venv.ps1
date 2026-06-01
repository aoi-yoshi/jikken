# Create project-local .venv and install requirements (Windows PowerShell).
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File scripts/setup_venv.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$VenvPip = Join-Path $Root ".venv\Scripts\pip.exe"

if (-not (Test-Path $VenvPy)) {
    Write-Host "Creating venv at $Root\.venv ..."
    python -m venv (Join-Path $Root ".venv")
}

Write-Host "Upgrading pip ..."
& $VenvPy -m pip install -U pip

Write-Host "Installing requirements (this may take a while) ..."
& $VenvPip install -r (Join-Path $Root "requirements.txt")

Write-Host "Done. Use this interpreter for all steps:"
Write-Host "  $VenvPy"
