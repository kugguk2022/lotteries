# Lotteries

[![CI](https://github.com/kugguk2022/lotteries/actions/workflows/ci.yml/badge.svg)](https://github.com/kugguk2022/lotteries/actions/workflows/ci.yml)

Lottery data playground for EuroMillions, Totoloto, and EuroDreams. The goal is to collect clean datasets, explore statistical signals, and prototype ranking models. Everything here is research-focused; use it responsibly.

## Features

- Fetch EuroMillions draws with retry, caching, deduplication, and 5s network timeouts.
- Pandera schema keeps draw columns consistent and type-safe.
- Ticket helper validates/scores EuroMillions 5-number + 2-star guesses.
- Existing labs (`grok.py`, `roi.py`, etc.) for experimentation with each lottery.
- Developer tooling via `pyproject.toml`, Ruff, pytest, and optional Makefile helpers.

## Project Layout

```text
lotteries/
|-- euromillions/
|   |-- get_draws.py        # Normalized EuroMillions fetcher (CLI entrypoint)
|   |-- schema.py           # Pandera schema + helpers
|   |-- grok.py             # EuroMillions modelling lab
|   |-- roi.py              # ROI backtesting sandbox (planned)
|-- eurodreams/             # EuroDreams analysis scripts
|-- totoloto/               # Totoloto analysis scripts
|-- data/                   # Local datasets (gitignored)
|-- Makefile                # Optional developer shortcuts
|-- pyproject.toml          # Project metadata and dependencies
|-- requirements.txt        # Editable-install helper
|-- README.md               # This document
```

## Getting Started

```bash
git clone https://github.com/kugguk2022/lotteries
cd lotteries

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install -U pip
pip install -e ".[dev]"
```

### Useful Commands

```bash
# Run Ruff lint
ruff check .

# Execute tests
pytest -q

# Fetch EuroMillions draws into data/euromillions.csv
python -m euromillions.get_draws --out data/euromillions.csv --append
```

## EuroMillions `get_draws`

Fetches historical EuroMillions draws from the MerseyWorld CSV endpoint, caches responses locally, retries transient failures, and normalizes the output. The script deduplicates on `draw_date` and validates via Pandera before writing a CSV.

```bash
python -m euromillions.get_draws --out data/euromillions.csv --append
python -m euromillions.get_draws --from 2023-01-01 --to 2024-12-31 --out data/euromillions_2023_2024.csv
```

### Checking a EuroMillions Ticket

```python
from euromillions import EuroMillionsGuess, evaluate_guess
from euromillions.get_draws import normalize

csv_text = (
    "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Lucky Star1,Lucky Star2\n"
    "2024-01-12,5,18,22,37,45,4,9\n"
)
draw = normalize(csv_text).iloc[0]

guess = EuroMillionsGuess(balls=[5, 18, 22, 37, 45], stars=[4, 9])
ball_hits, star_hits = evaluate_guess(draw, guess)

print(f"Matched {ball_hits} numbers and {star_hits} stars")
```

### Sample CSV Output

```csv
draw_date,ball_1,ball_2,ball_3,ball_4,ball_5,star_1,star_2
2024-01-02,1,2,3,4,5,1,2
2024-01-09,6,7,8,9,10,3,4
2024-01-16,11,12,13,14,15,5,6
```

### Output Schema

- `draw_date`: ISO `YYYY-MM-DD`
- `ball_1` .. `ball_5`: integers 1-50
- `star_1`, `star_2`: integers 1-12

### Validation

`euromillions/schema.py` defines the canonical Pandera schema. `validate_df` coerces `draw_date` to timezone-naive timestamps and enforces number ranges (1-50 for balls, 1-12 for stars). Tests cover both the schema and CSV normalization pipeline.

## EuroMillions ROI (Planned)

**Not implemented yet -- CLI will error if run.**

`euromillions/roi.py` will host walk-forward bankroll simulations, EV gating, and ticket ranking. The CLI entry point will be exposed once the module is production-ready.

## Testing

Pytest coverage lives in `tests/` for schema validation, CSV normalization, and ticket scoring. Suites run offline; install dev extras with `pip install -e ".[dev]"` before executing `pytest`.

## Legacy R Scripts

R notebooks and `.r` files remain for historical reference but are deprecated. Prefer the Python pipelines when adding new work.

## Contributing

Pull requests are welcome. Please keep experiments isolated, document inputs/outputs, and add unit tests when introducing new behaviours.

## Disclaimer

This repository is for research and education only. Lotteries remain games of chance; nothing here is financial advice.

## License

[MIT](LICENSE)

##Citation
If you use this work in your research, please cite:

@software{Lotteries2025,
  -title={Inference in European Lotteries},
  -author={kugguk2022},
  -year={2025},
  -url={https://github.com/kugguk2022/lotteries}
}
