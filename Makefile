.PHONY: venv install lint test draws quality

venv:
	python -m venv .venv

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	ruff check .
	pytest -q

quality: test

draws:
	python -m euromillions.get_draws --out data/euromillions.csv --append
