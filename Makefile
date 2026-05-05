.PHONY: setup test test-fast smoke-baseline smoke-ldc clean

setup:
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -x --no-header -q

smoke-baseline:
	python -m src.train --config-name baseline_scan_smoke

smoke-ldc:
	python -m src.train --config-name ldc_v2_scan_smoke

clean:
	rm -rf outputs/ multirun/ wandb/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
