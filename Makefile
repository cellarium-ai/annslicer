.PHONY: install lint typecheck benchmark

install:
	pip install --upgrade pip
	pip install -e ".[dev]"

lint:
	ruff check --fix src/ tests/ benchmarks/
	ruff format src/ tests/ benchmarks/

typecheck:
	mypy src/annslicer

benchmark:
	pytest benchmarks/ --benchmark-only -v -s
