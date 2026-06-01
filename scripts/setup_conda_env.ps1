# Create conda env `flvl` (python=3.10) and install per the thesis plan.
# Run:  powershell -ExecutionPolicy Bypass -File scripts/setup_conda_env.ps1
$ErrorActionPreference = "Stop"

$CondaBat = "$env:USERPROFILE\miniconda3\condabin\conda.bat"
if (-not (Test-Path $CondaBat)) {
    Write-Error "conda.bat not found at $CondaBat. Install Miniconda first."
}

Write-Host "Creating env flvl (python=3.10) ..."
& $CondaBat create -y -n flvl python=3.10
if ($LASTEXITCODE -ne 0) { throw "conda create failed" }

$EnvPy  = "$env:USERPROFILE\miniconda3\envs\flvl\python.exe"
$EnvPip = "$env:USERPROFILE\miniconda3\envs\flvl\Scripts\pip.exe"
if (-not (Test-Path $EnvPy)) {
    throw "Env python not found at $EnvPy"
}

Write-Host "Upgrading pip ..."
& $EnvPy -m pip install -U pip

# RTX 5050 (Blackwell, sm_120) requires CUDA 12.8 PyTorch wheels.
Write-Host "Installing PyTorch (cu128) ..."
& $EnvPip install --index-url https://download.pytorch.org/whl/cu128 `
    torch torchvision torchaudio

Write-Host "Installing transformers / accelerate / peft / datasets / flwr / opencv / avalanche ..."
& $EnvPip install `
    "transformers>=4.49.0" `
    "accelerate>=0.33.0" `
    "peft>=0.12.0" `
    "datasets" `
    "safetensors>=0.4.0" `
    "Pillow>=10.0.0" `
    "PyYAML>=6.0.0" `
    "tqdm>=4.66.0" `
    "scikit-learn>=1.4.0" `
    "pandas>=2.2.0" `
    "fsspec>=2023.1.0,<=2026.2.0" `
    "flwr>=1.8.0" `
    "protobuf>=4.25.0" `
    "opencv-python>=4.9.0.80" `
    "avalanche-lib"

Write-Host "Verifying ..."
& $EnvPy -c "import torch, transformers, peft, flwr; import avalanche; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('transformers', transformers.__version__); print('peft', peft.__version__); print('flwr', flwr.__version__); print('avalanche', avalanche.__version__)"

Write-Host ""
Write-Host "Done. To activate manually:"
Write-Host "  $env:USERPROFILE\miniconda3\condabin\conda.bat activate flvl"
Write-Host "Or just use the env python directly:"
Write-Host "  $EnvPy scripts\step00_env_check.py"
