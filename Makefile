.PHONY: setup test lint format run-dashboard build clean

VENV_BIN = .venv/bin
PYTHON = $(VENV_BIN)/python
PYTEST = $(VENV_BIN)/pytest
FLAKE8 = $(VENV_BIN)/flake8
BLACK = $(VENV_BIN)/black
STREAMLIT = $(VENV_BIN)/streamlit

setup:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install pre-commit pytest flake8 black
	pre-commit install

test:
	$(PYTHON) -m pytest

lint:
	$(FLAKE8) src tests
	$(BLACK) --check src tests

format:
	$(BLACK) src tests

run-dashboard:
	$(STREAMLIT) run src/explainable_aml/dashboard/app.py

build:
	docker build -t explainable-aml .

clean:
	rm -rf build/ dist/ *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
