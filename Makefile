.PHONY: install lint typecheck benchmark build-check

install:
	pip install --upgrade pip
	pip install -e ".[dev]"

lint:
	ruff check --fix src/ tests/ benchmarks/
	ruff format src/ tests/ benchmarks/

typecheck:
	mypy src/annslicer

# Usage: make benchmark [INPUT=path/to/file.h5ad] [PYTEST_ARGS="--bench-jobs 1,4,8 ..."]
# (see CONTRIBUTING.md; `pytest benchmarks/ --help` lists the --bench-* options)
INPUT ?=
PYTEST_ARGS ?=

benchmark:
	pytest benchmarks/ --benchmark-only -v -s $(if $(INPUT),--bench-input "$(INPUT)") $(PYTEST_ARGS)

build-check:
	@echo "--- Building sdist and wheel ---"
	python -m build --outdir /tmp/annslicer-dist-check
	@echo "--- Checking distributions ---"
	twine check /tmp/annslicer-dist-check/*
	@rm -rf /tmp/annslicer-dist-check
	@echo "--- Build check passed ---"
