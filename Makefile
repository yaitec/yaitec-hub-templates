.PHONY: help lint fix format all

help:
	@echo "Usage: make [target]"
	@echo "Available targets:"
	@echo "  lint    - Run Ruff lint check"
	@echo "  fix     - Auto-fix lint issues with Ruff"
	@echo "  format  - Format code using Ruff"
	@echo "  all     - Run lint, fix, and format sequentially"

lint:
	uv run ruff check . --output-format=github

fix:
	uv run ruff check . --output-format=github --fix

format:
	uv run ruff format .

all: lint fix format