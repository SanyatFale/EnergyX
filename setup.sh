#!/bin/bash
# TinyTS-Scientist — Setup Script
# Usage: bash setup.sh

set -e

echo "TinyTS-Scientist Setup"
echo "======================"
echo ""

# ── Check Python version ──────────────────────────────────────
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
major=$(echo "$python_version" | cut -d. -f1)
minor=$(echo "$python_version" | cut -d. -f2)
if [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -lt 10 ]; }; then
    echo "ERROR: Python 3.10+ required (found $python_version)"
    exit 1
fi
echo "  Found Python $python_version"
echo ""

# ── Create virtual environment ────────────────────────────────
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  Virtual environment already exists. Skipping."
else
    python3 -m venv venv
    echo "  Created venv/"
fi
echo ""

# ── Install dependencies ──────────────────────────────────────
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Dependencies installed."
echo ""

# ── Create .env from example ──────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
    echo "  Edit .env to set your CEREBRAS_API_KEY (or switch to Ollama)."
else
    echo ".env already exists. Skipping."
fi
echo ""

# ── Done ──────────────────────────────────────────────────────
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. Edit .env — set CEREBRAS_API_KEY or LLM_PROVIDER=ollama"
echo "  3. streamlit run app.py"
echo ""
