.PHONY: venv install lint test draws

venv:
	python -m venv .venv

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -q

draws:
	python -m euromillions.get_draws --out data/euromillions.csv --append
