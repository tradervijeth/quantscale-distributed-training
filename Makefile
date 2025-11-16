# QuantScale Makefile
#
# Common development tasks for QuantScale distributed training framework
#
# Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
# Licensed under the Apache License 2.0
# Contact: finance@vijeth.com

.PHONY: help install install-dev install-test clean test test-cov lint format check benchmark demo docker docker-run

# Default target
.DEFAULT_GOAL := help

# Python and pip
PYTHON := python3
PIP := $(PYTHON) -m pip

# Directories
SRC_DIR := quantscale
TEST_DIR := tests
SCRIPTS_DIR := scripts
EXAMPLES_DIR := examples

help: ## Show this help message
	@echo "QuantScale - Distributed Training Framework for Financial ML"
	@echo "Author: Vithushan Jeyapahan <finance@vijeth.com>"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in development mode
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .
	@echo "✓ QuantScale installed successfully"

install-dev: ## Install with development dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .
	$(PIP) install -r requirements-dev.txt
	@echo "✓ Development dependencies installed"

install-test: ## Install with test dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .
	$(PIP) install -r requirements-test.txt
	@echo "✓ Test dependencies installed"

install-all: ## Install all dependencies (dev + test)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .
	$(PIP) install -r requirements-dev.txt -r requirements-test.txt
	@echo "✓ All dependencies installed"

clean: ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .eggs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*~' -delete
	@echo "✓ Cleaned build artifacts and cache files"

test: ## Run tests with pytest
	pytest $(TEST_DIR)/ -v

test-cov: ## Run tests with coverage report
	pytest $(TEST_DIR)/ -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "✓ Coverage report generated in htmlcov/index.html"

test-fast: ## Run tests in parallel
	pytest $(TEST_DIR)/ -v -n auto

test-watch: ## Run tests in watch mode
	pytest-watch $(TEST_DIR)/ -v

format: ## Format code with black and isort
	black $(SRC_DIR)/ $(TEST_DIR)/ $(SCRIPTS_DIR)/ $(EXAMPLES_DIR)/
	isort $(SRC_DIR)/ $(TEST_DIR)/ $(SCRIPTS_DIR)/ $(EXAMPLES_DIR)/
	@echo "✓ Code formatted"

lint: ## Run linters (flake8, mypy)
	@echo "Running flake8..."
	flake8 $(SRC_DIR)/ $(TEST_DIR)/ $(SCRIPTS_DIR)/ --max-line-length=100 --extend-ignore=E203,W503
	@echo "Running mypy..."
	mypy $(SRC_DIR)/ --ignore-missing-imports --no-strict-optional
	@echo "✓ Linting passed"

check: format lint test ## Run format, lint, and test

type-check: ## Run type checking with mypy
	mypy $(SRC_DIR)/ --ignore-missing-imports --no-strict-optional

security: ## Run security checks
	@echo "Running bandit..."
	bandit -r $(SRC_DIR)/ -ll
	@echo "Running safety..."
	safety check
	@echo "✓ Security checks passed"

pre-commit-install: ## Install pre-commit hooks
	pre-commit install
	@echo "✓ Pre-commit hooks installed"

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

benchmark: ## Run performance benchmarks
	$(PYTHON) $(SCRIPTS_DIR)/benchmark.py --model transformer --mode both
	@echo "✓ Benchmark completed - results saved to benchmark_results.csv"

benchmark-gpu: ## Run GPU scaling benchmarks
	$(PYTHON) $(SCRIPTS_DIR)/benchmark.py --gpu-scaling
	@echo "✓ GPU scaling benchmark completed"

demo: ## Run end-to-end demo
	$(PYTHON) $(EXAMPLES_DIR)/demo.py
	@echo "✓ Demo completed"

example-basic: ## Run basic usage example
	$(PYTHON) $(EXAMPLES_DIR)/basic_usage.py

train-single: ## Train model on single GPU (requires data)
	$(PYTHON) $(SCRIPTS_DIR)/train_single.py --train-data data/train.parquet --val-data data/val.parquet

train-distributed: ## Train with distributed training (4 GPUs)
	torchrun --nproc_per_node=4 $(SCRIPTS_DIR)/train_distributed.py --train-data data/train.parquet --val-data data/val.parquet

docker: ## Build Docker image
	docker build -t quantscale:latest .
	@echo "✓ Docker image built: quantscale:latest"

docker-run: ## Run Docker container interactively
	docker run --rm -it --gpus all -v $(PWD)/data:/workspace/data quantscale:latest

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d
	@echo "✓ Services started:"
	@echo "  - Jupyter: http://localhost:8888"
	@echo "  - TensorBoard: http://localhost:6006"
	@echo "  - MLflow: http://localhost:5000"

docker-compose-down: ## Stop all docker-compose services
	docker-compose down

build: ## Build package distribution
	$(PYTHON) -m build
	@echo "✓ Package built in dist/"

publish-test: build ## Publish to TestPyPI
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	$(PYTHON) -m twine upload dist/*

docs: ## Build documentation (if Sphinx is set up)
	@echo "Documentation build not yet implemented"

version: ## Show version information
	@$(PYTHON) -c "import quantscale; print(f'QuantScale v{quantscale.__version__}')"
	@$(PYTHON) --version
	@$(PIP) --version

info: ## Show environment information
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo "PyTorch: $$($(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "CUDA Available: $$($(PYTHON) -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
	@echo "GPU Count: $$($(PYTHON) -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'N/A')"

ci: ## Run CI checks locally (format, lint, test, security)
	@echo "Running CI checks..."
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) test-cov
	@$(MAKE) security
	@echo "✓ All CI checks passed"

all: clean install-all check benchmark ## Clean, install, check, and benchmark

# Development shortcuts
dev: install-dev pre-commit-install ## Set up development environment
	@echo "✓ Development environment ready"

quick-test: ## Quick test run (no coverage)
	pytest $(TEST_DIR)/ -v --tb=short -x

# Utility targets
lines: ## Count lines of code
	@echo "Lines of code:"
	@find $(SRC_DIR) -name '*.py' | xargs wc -l | tail -1

check-deps: ## Check for outdated dependencies
	$(PIP) list --outdated

update-deps: ## Update dependencies
	$(PIP) install --upgrade -r requirements.txt -r requirements-dev.txt -r requirements-test.txt
