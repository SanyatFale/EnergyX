.PHONY: setup run cli test clean help

help:
	@echo "TinyTS-Scientist"
	@echo ""
	@echo "  make setup     Create venv and install dependencies"
	@echo "  make run       Launch Streamlit UI"
	@echo "  make cli       Run CLI forecast (edit args below)"
	@echo "  make test      Run tests"
	@echo "  make clean     Remove outputs and caches"
	@echo ""

setup:
	bash setup.sh

run:
	. venv/bin/activate && streamlit run app.py

cli:
	. venv/bin/activate && python -m tinyts.cli run \
		"data/raw/building_energy_data (copy 1)_elec.csv" \
		--time timestamp --target meter_reading \
		--query "forecast next 3 days and explain"

test:
	. venv/bin/activate && pytest tests/ -v

clean:
	rm -rf outputs/*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned."
