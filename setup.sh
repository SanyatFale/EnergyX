#!/bin/bash
# Setup script for TinyTS-Scientist

set -e

echo "🚀 TinyTS-Scientist Setup"
echo "=========================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate and install dependencies
echo "Installing dependencies..."
echo "This may take a few minutes..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
pip install -q langchain>=0.3.0 \
    langchain-community>=0.3.0 \
    langgraph>=0.2.0 \
    langchain-ollama>=0.2.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    statsmodels>=0.14.0 \
    scikit-learn>=1.3.0 \
    lightgbm>=4.0.0 \
    torch>=2.0.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    plotly>=5.14.0 \
    pyarrow>=12.0.0 \
    fastparquet>=2023.4.0 \
    pydantic>=2.0.0 \
    pydantic-settings>=2.0.0 \
    python-dotenv>=1.0.0 \
    rich>=13.0.0 \
    typer>=0.9.0

echo "✓ Dependencies installed"
echo ""

# Create .env file
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created"
else
    echo "⚠️  .env file already exists. Skipping."
fi
echo ""

# Generate sample data
echo "Generating sample data..."
python examples/generate_sample_data.py
echo "✓ Sample data generated"
echo ""

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Make sure Ollama is running:"
echo "   ollama serve"
echo ""
echo "3. Pull the model (in another terminal):"
echo "   ollama pull llama3.2:3b"
echo ""
echo "4. Run the example:"
echo "   python examples/basic_usage.py"
echo ""

