# Sigmoid Binary Classification System - Setup Script for Windows PowerShell
# This script automates the entire setup process

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "Sigmoid Binary Classification System - Setup" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if UV is installed
Write-Host "[Step 1/4] Checking for UV..." -ForegroundColor Green
try {
    $uvVersion = uv --version
    Write-Host "✓ UV found: $uvVersion" -ForegroundColor Green
}
catch {
    Write-Host "✗ UV not found. Installing UV..." -ForegroundColor Yellow
    pip install uv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed to install UV. Please install manually: pip install uv" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Step 2: Create virtual environment
Write-Host "[Step 2/4] Creating virtual environment..." -ForegroundColor Green
if (Test-Path ".\.venv") {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}
else {
    uv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

Write-Host ""

# Step 3: Activate virtual environment
Write-Host "[Step 3/4] Activating virtual environment..." -ForegroundColor Green
try {
    & .\.venv\Scripts\Activate.ps1
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
}
catch {
    Write-Host "⚠ Could not activate with script, trying alternative method..." -ForegroundColor Yellow
    $env:Path = "$PWD\.venv\Scripts;$env:Path"
    Write-Host "✓ Virtual environment path added" -ForegroundColor Green
}

Write-Host ""

# Step 4: Install dependencies
Write-Host "[Step 4/4] Installing dependencies from requirements.txt..." -ForegroundColor Green
Write-Host "Installing: numpy, pandas, matplotlib, tabulate" -ForegroundColor Cyan
uv pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "Setup Complete! ✓" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the program, use:" -ForegroundColor White
Write-Host "  python main.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "The venv is already activated in this terminal session." -ForegroundColor Cyan
Write-Host ""
