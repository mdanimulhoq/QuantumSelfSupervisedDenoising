#!/bin/bash
# N2LN-QEM Setup Script (TDD §10.3)

set -e

echo "========================================="
echo "N2LN-QEM Setup Script"
echo "========================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install package
echo "Installing N2LN-QEM..."
pip install -e .

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw
mkdir -p checkpoints
mkdir -p experiments/*/plots

# Set seeds
echo "Setting seeds..."
python -c "from src.utils.seeding import set_seed; set_seed(42); print('Seed set')"

echo "========================================="
echo "✅ Setup complete!"
echo "Run: make reproduce"
echo "========================================="
