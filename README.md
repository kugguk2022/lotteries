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
# If sources are flaky, reuse local data instead of failing
python -m euromillions.get_draws --out data/euromillions.csv --allow-stale
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

Resilience notes:

- Fetcher accepts header-less CSV payloads and trims malformed source headers.
- `--allow-stale` reuses (in order) an existing `--out` file, the bundled `euromillions/euromillions_2016_2025.csv`, or the tiny sample `data/examples/euromillions_sample.csv` when all network sources fail.

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
- If a fetch fails (network or source drift) but a local CSV already exists, it will reuse the local copy; otherwise the run stops for that lottery.

## Contributing

- Open a PR or Discussion for new lotteries, docs, or modelling experiments.
- Keep experiments isolated, document inputs/outputs, and add tests for new behaviours.
- Tag releases when milestones land so downstream users can pin versions.

## License

[MIT](LICENSE)

## Architecture & Logic

The repository implements a multi-stage pipeline for lottery analysis, capable of running end-to-end for EuroMillions, and partially for others.

### The Pipeline Steps

1.  **Fetch (`get_draws`)**: Downloads historical draw data from various online sources, normalizes it, and saves it to a CSV file. It supports caching and fallbacks to local data if the network or source is unavailable.
2.  **Lotto Lab (`lottolab.py`)** _(EuroMillions only)_: A comprehensive research lab that runs an "Agent" (forecaster), a "Discriminator" (pair co-occurrence), a "Grok" model (tiny transformer), and an RL Mixer. It produces detailed analysis and plots.
3.  **Features (`phase2_sobol.py`)**: Extracts advanced features from the draw history, specifically calculating "Point of Interest" (POI) metrics based on pair co-occurrences and time-based features (Euler phi).
4.  **Grok (`grok.py`)**: Trains a dual-input Transformer model on the extracted features (`g` sequence vs `poi` sequence) to learn patterns and predict future "interest" scores.
5.  **Tickets (`phase2_sobol.py`)**: Generates candidate tickets using Sobol low-discrepancy sequences combined with combinadic unranking. This ensures tickets cover the combinations space more evenly than random sampling.
6.  **Infer (`infer.py`)**: A baseline generator that creates tickets based on simple frequency-weighted sampling of historical draws.

### Quickstart Batch Scripts

We provide one-click batch scripts for Windows to run the entire pipeline for each lottery.

#### EuroMillions

Runs the full 6-stage pipeline.

```cmd
start_euromillions.bat
```

_Outputs:_ `outputs/euromillions/`

#### Totoloto

Runs a 5-stage pipeline (skips `lottolab`).

```cmd
start_totoloto.bat
```

_Outputs:_ `outputs/totoloto/`

#### EuroDreams

Runs Fetch and Infer stages.
_Note: Advanced Sobol/Grok stages are currently skipped due to 6-ball incompatibility._

```cmd
start_eurodreams.bat
```

_Outputs:_ `outputs/eurodreams/`
