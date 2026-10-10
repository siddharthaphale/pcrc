.PHONY: test-cpu test-gpu test-all

# Install the desired extras first (see README); preserve them during tests.
test-cpu:
	uv run --no-sync pytest tests/cpu/ -v --tb=short

test-gpu:
	uv run --no-sync pytest tests/gpu/ -m gpu -v --tb=short

test-all: test-cpu test-gpu
