.PHONY: venv install lint test draws quality branch-hmm-v3

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

branch-hmm-v3:
	python -m euromillions.branch_hmm_v3 --source real --history data/euromillions.csv --out-dir outputs/euromillions/branch_hmm_v3
