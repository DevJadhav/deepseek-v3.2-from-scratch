.PHONY: help check install train test clean

# Default target
help:
	@echo "DeepSeek-From-Scratch Makefile"
	@echo "=============================="
	@echo "Available commands:"
	@echo "  make check    - Check documentation (README.md, TINY_MODEL_TRAINING.md) for broken links"
	@echo "  make install  - Install dependencies using uv"
	@echo "  make train    - Run tiny model training (local MLX)"
	@echo "  make test     - Run Python tests"
	@echo "  make clean    - Remove build artifacts and cache"

check:
	@echo "Checking documentation..."
	@uv run python scripts/check_docs.py README.md TINY_MODEL_TRAINING.md

install:
	@echo "Installing dependencies..."
	@uv sync

train:
	@echo "Running tiny model training..."
	@uv run python scripts/train_tiny.py --max-steps 100 --quick-test

test:
	@echo "Running tests..."
	@uv run pytest deepseek-from-scratch-python/tests

clean:
	@echo "Cleaning up..."
	@rm -rf .pytest_cache
	@rm -rf __pycache__
	@find . -type d -name "__pycache__" -exec rm -rf {} +
