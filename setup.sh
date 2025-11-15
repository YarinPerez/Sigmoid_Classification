#!/bin/bash
# Sigmoid Binary Classification System - Setup Script for Linux/macOS
# This script automates the entire setup process

echo "=================================================="
echo "Sigmoid Binary Classification System - Setup"
echo "=================================================="
echo ""

# Step 1: Check if UV is installed
echo "[Step 1/4] Checking for UV..."
if command -v uv &> /dev/null; then
    echo "✓ UV found: $(uv --version)"
else
    echo "✗ UV not found. Installing UV..."
    pip install uv
    if [ $? -ne 0 ]; then
        echo "✗ Failed to install UV. Please install manually: pip install uv"
        exit 1
    fi
fi

echo ""

# Step 2: Create virtual environment
echo "[Step 2/4] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "✓ Virtual environment already exists"
else
    uv venv
    if [ $? -ne 0 ]; then
        echo "✗ Failed to create virtual environment"
        exit 1
    fi
    echo "✓ Virtual environment created"
fi

echo ""

# Step 3: Activate virtual environment
echo "[Step 3/4] Activating virtual environment..."
source .venv/bin/activate
if [ $? -eq 0 ]; then
    echo "✓ Virtual environment activated"
else
    echo "⚠ Warning: Could not activate virtual environment"
fi

echo ""

# Step 4: Install dependencies
echo "[Step 4/4] Installing dependencies from requirements.txt..."
echo "Installing: numpy, pandas, matplotlib, tabulate"
uv pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "✗ Failed to install dependencies"
    exit 1
fi
echo "✓ Dependencies installed successfully"

echo ""
echo "=================================================="
echo "Setup Complete! ✓"
echo "=================================================="
echo ""
echo "To run the program, use:"
echo "  python main.py"
echo ""
echo "The venv is already activated in this terminal session."
echo ""
