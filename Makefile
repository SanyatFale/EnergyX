.PHONY: install setup test run-example generate-data clean help

help:
	@echo "TinyTS-Scientist - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install dependencies with Poetry"
	@echo "  make setup          - Complete setup (install + generate data)"
	@echo ""
	@echo "Data:"
	@echo "  make generate-data  - Generate sample datasets"
	@echo ""
	@echo "Run:"
	@echo "  make run-example    - Run basic usage example"
	@echo "  make test           - Run tests (when implemented)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove generated files and outputs"
	@echo ""

install:
	@echo "Installing dependencies with Poetry..."
	poetry install
	@echo "✓ Dependencies installed"

setup: install
	@echo "Setting up environment..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "✓ Created .env file"; fi
	@echo "Generating sample data..."
	poetry run python examples/generate_sample_data.py
	@echo ""
	@echo "✓ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Make sure Ollama is running: ollama serve"
	@echo "  2. Pull the model: ollama pull llama3.2:3b"
	@echo "  3. Run example: make run-example"

generate-data:
	@echo "Generating sample datasets..."
	poetry run python examples/generate_sample_data.py

run-example:
	@echo "Running basic usage example..."
	@echo "Note: This will prompt for human approval during execution"
	@echo ""
	poetry run python examples/basic_usage.py

test:
	@echo "Running tests..."
	poetry run pytest tests/ -v

clean:
	@echo "Cleaning up..."
	rm -rf outputs/*
	rm -rf data/raw/*.csv
	rm -rf data/processed/*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleanup complete"

lint:
	@echo "Running linters..."
	poetry run black tinyts/ examples/ --check
	poetry run ruff check tinyts/ examples/

format:
	@echo "Formatting code..."
	poetry run black tinyts/ examples/
	poetry run ruff check tinyts/ examples/ --fix

