.PHONY: help install test conformance governance stats seal verify items clean

help:
	@echo "install       editable install with dev extras"
	@echo "test          full test suite"
	@echo "conformance   Phase-0 fixtures only (tests/conformance)"
	@echo "governance    seal / blind / rerun-policy tests"
	@echo "stats         decision / bootstrap / power tests on synthetic data"
	@echo "seal          regenerate FREEZE-1.json"
	@echo "verify        verify FREEZE-1.json against the working tree"

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest tests -q

conformance:
	python -m pytest tests/conformance -q

governance:
	python -m pytest tests/governance -q

stats:
	python -m pytest tests/stats -q

seal:
	python -m wda.governance.seal write --stage freeze1

verify:
	python -m wda.governance.seal verify --stage freeze1

items:
	python -m wda.items.generator --split explore --n 40 --out runs/phase0/items_explore.jsonl

clean:
	python -c "import shutil,pathlib;[shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
