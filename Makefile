.PHONY: install test lint bench examples clean all

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/ examples/

bench:
	python benchmarks/gpu_scaling.py

examples:
	python examples/taylor_green.py
	python examples/compressible_hit.py

clean:
	rm -rf __pycache__ .pytest_cache src/*.egg-info
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

all: install test lint
