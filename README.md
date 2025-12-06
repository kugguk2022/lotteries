# Lotteries

[![CI status](https://github.com/kugguk2022/lotteries/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kugguk2022/lotteries/actions/workflows/ci.yml)

Lottery data playground for EuroMillions, Totoloto, and EuroDreams. The repo ships a small typed public API plus labs for modelling, bankroll experiments, and scraping. Everything is research-focused; use it responsibly.

## Quickstart

```bash
git clone https://github.com/kugguk2022/lotteries
cd lotteries
python -m venv .venv
.\.venv\Scripts\activate    # Windows
# or: source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"

# Quality gate (lint + tests)
make test     # or: ruff check . && pytest -q

# 1) Fetch EuroMillions history (cached and normalized)
python -m euromillions.get_draws --out data/euromillions.csv --append

# 2) Generate frequency-weighted candidate tickets
python -m euromillions.infer --history data/euromillions.csv --n 10 --out runs/euromillions_candidates.csv

# 3) (Optional) Run all lotteries end-to-end
python run_all.py --n-candidates 200
```

## Use Cases
- Keep a validated EuroMillions history CSV up to date with retry/caching.
- Inspect number/“lucky star” distributions for quick EDA.
- Generate baseline candidate tickets using smoothed frequency sampling.
- Extend the labs (`grok.py`, `roi.py`) for custom modelling experiments.
 - Fetch and benchmark Totoloto and EuroDreams datasets alongside EuroMillions.

## EuroMillions `get_draws`

Fetches historical EuroMillions draws from the MerseyWorld CSV endpoint, caches responses locally, retries transient failures, and normalizes the output. The script deduplicates on `draw_date` and validates before writing a CSV.

```bash
python -m euromillions.get_draws --out data/euromillions.csv --append
python -m euromillions.get_draws --from 2023-01-01 --to 2024-12-31 --out data/euromillions_2023_2024.csv
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

`euromillions/schema.py` enforces the canonical column set, coerces `draw_date` to timezone-naive timestamps, and checks ranges (1-50 for balls, 1-12 for stars). Tests cover both the schema and CSV normalization pipeline.

## EuroMillions Inference (Baseline)

`euromillions/infer.py` provides a light, frequency-weighted baseline generator inspired by the original `grok.py` experiment. It reads historical draws, builds smoothed number frequencies, and samples tickets without replacement.

```bash
python -m euromillions.infer --history data/euromillions.csv --n 10 --out runs/euromillions_candidates.csv
```

- `--history`: normalized CSV from `euromillions.get_draws`
- `--n`: number of candidate tickets to generate (default 10)
- `--smoothing`: additive smoothing applied to frequencies (default 1.0)
- `--seed`: optional seed for reproducibility

## EuroMillions ROI (Planned)

**Not implemented yet -- CLI will error if run.**

`euromillions/roi.py` will host walk-forward bankroll simulations, EV gating, and ticket ranking. The CLI entry point will be exposed once the module is production-ready.

## Testing

```bash
pytest -q
```

Tests include schema/normalization checks and a baseline comparison that ensures the frequency-weighted sampler outperforms a uniform random picker on a biased dataset.

## Legacy R Scripts

R notebooks and `.r` files remain for historical reference but are deprecated. Prefer the Python pipelines when adding new work.

## Totoloto (lab)

Totoloto fetcher and parsing utilities (`totoloto/`). Heuristics may need refresh if upstream HTML changes.

```bash
python totoloto/totoloto_get_draws.py --out data/totoloto.csv
python totoloto/totoloto_get_draws.py --out data/totoloto_2015_2025.csv --start-year 2015 --end-year 2025
```

## EuroDreams (lab)

EuroDreams draw fetchers (`eurodreams/`) for annuity-style analysis. HTML may drift; keep scripts isolated from the stable API.

```bash
python eurodreams/eurodreams_get_draws.py --out data/eurodreams_all.csv
python eurodreams/eurodreams_get_draws.py --out data/eurodreams_2023_2025.csv --start-year 2023 --end-year 2025
```

## Run-All Orchestrator

`run_all.py` pulls history for EuroMillions, Totoloto, and EuroDreams (via the existing fetchers), generates frequency-weighted candidates for each, and reports whether the frequency sampler beats a uniform random baseline on a simple “top-bin hit rate” metric with a permutation-test p-value.

```bash
python run_all.py --n-candidates 200 --permutation-iters 500 --smoothing 1.0
```

- Fetches to `data/{lottery}.csv` by default (reuses cached files if present).
- Writes candidates to `runs/{lottery}_candidates.csv`.
- Prints mean scores for frequency vs random and an approximate p-value; values << 0.05 indicate the frequency sampler captures bias beyond uniform chance on the observed history (not forward-looking predictiveness).

## Contributing
- Open a PR or Discussion for new lotteries, docs, or modelling experiments.
- Keep experiments isolated, document inputs/outputs, and add tests for new behaviours.
- Tag releases when milestones land so downstream users can pin versions.

## License

[MIT](LICENSE)
