.PHONY: install lint typecheck benchmark build-check

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

build-check:
	@echo "--- Building sdist and wheel ---"
	python -m build --outdir /tmp/annslicer-dist-check
	@echo "--- Checking distributions ---"
	twine check /tmp/annslicer-dist-check/*
	@rm -rf /tmp/annslicer-dist-check
	@echo "--- Build check passed ---"
