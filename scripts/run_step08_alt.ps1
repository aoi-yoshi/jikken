$ErrorActionPreference = "Continue"
$PY = "C:\Users\aoi7y\miniconda3\envs\flvl\python.exe"
$ROOT = "c:\Python\実験_修正"
Set-Location $ROOT

Write-Host "=== start lpred at $(Get-Date) ==="
& $PY scripts\step08_alt_modes.py --mode lpred 2>&1
$ec1 = $LASTEXITCODE
Write-Host "=== lpred exit=$ec1 at $(Get-Date) ==="

if ($ec1 -eq 0) {
    Write-Host "=== start lpred_grad at $(Get-Date) ==="
    & $PY scripts\step08_alt_modes.py --mode lpred_grad 2>&1
    $ec2 = $LASTEXITCODE
    Write-Host "=== lpred_grad exit=$ec2 at $(Get-Date) ==="
} else {
    Write-Host "=== skip lpred_grad due to lpred failure (exit=$ec1) ==="
}

Write-Host "=== all done at $(Get-Date) ==="
