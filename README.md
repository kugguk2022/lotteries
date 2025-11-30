# Lotteries

[![CI status](https://github.com/kugguk2022/lotteries/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kugguk2022/lotteries/actions/workflows/ci.yml)

Lottery data playground for EuroMillions, Totoloto, and EuroDreams. The repo ships a small typed public API plus a set of labs for modelling, bankroll experiments, and scraping. Everything is research-focused; use it responsibly.

## Quickstart

```bash
git clone https://github.com/kugguk2022/lotteries
cd lotteries
python -m venv .venv
.\.venv\Scripts\activate    # Windows
# or: source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"

# Quality gate
make test

# Fetch EuroMillions history and append to a cached CSV
python -m euromillions.get_draws --out data/euromillions.csv --append
```

## Motivation & Roadmap

- **Stable** [`euromillions/get_draws.py`](euromillions/get_draws.py): signal detection + clean data ingest (retry, cache, normalize); [`euromillions/guess.py`](euromillions/guess.py) for ticket scoring; [`euromillions/schema.py`](euromillions/schema.py) for typed draws. Research hooks: signal detection, schema-first feature building.
- **Lab** [`euromillions/roi.py`](euromillions/roi.py): EV gating and walk-forward bankroll simulation before ranking tickets.
- **Lab** [`totoloto/`](totoloto): Portuguese Totoloto scraping + robustness checks across lotteries.
- **Lab** [`eurodreams/`](eurodreams): EuroDreams fetchers to study annuity-style payouts and draw drift.
- **Lab** [`euromillions_agent/`](euromillions_agent): agent/discriminator/grok/mixer stack for richer modelling.
- **Lab** [`grok.py`](grok.py): standalone transformer sandbox.
- **Deprecated** legacy R files (`*.r`) kept only for provenance.
- Releases: tag milestones like `v0.1` (stable EuroMillions fetch + schema + scoring) so downstream users can pin versions and cite specific URIs.

## Typed Public API (Python)

- `euromillions.EuroMillionsGuess`: immutable ticket with validation and sorting.
- `euromillions.evaluate_guess`: returns `(ball_hits, star_hits)` against a normalized draw.
- `euromillions.normalize`: CSV text -> validated `DataFrame` (Pandera schema enforced).
- `euromillions.load_history`: convenience reader for CSVs produced by this repo.
Everything else is internal or experimental; docstrings call out lab status explicitly.

## Folder Quickstarts

- `euromillions/`: `python -m euromillions.get_draws --out data/euromillions.csv --append`; then score tickets:
  ```python
  from euromillions import EuroMillionsGuess, evaluate_guess, load_history
  df = load_history("data/euromillions.csv")
  guess = EuroMillionsGuess([1, 2, 20, 30, 40], [3, 11])
  print(evaluate_guess(df.iloc[-1], guess))
  ```
- `totoloto/`: `python totoloto/totoloto_get_draws.py --out data/totoloto.csv --start-year 2015 --end-year 2025`; JSON: `--format json --out data/totoloto.json`.
- `eurodreams/`: `python eurodreams/eurodreams_get_draws.py --out data/eurodreams_all.csv --start-year 2023 --end-year 2025`; pick a source with `--source irish|euro|lottery_ie`.

## EuroMillions `get_draws` (CLI + API)

Arguments

| Flag | Purpose | Default |
| --- | --- | --- |
| `--from`, `--to` | Inclusive date window (`YYYY-MM-DD`) | None (full history) |
| `--out` | Output CSV path (required) | – |
| `--append` | Dedup + append to existing CSV | False |
| `--cache-dir` | Override cache root (hash-named CSV blobs) | `.cache/euromillions` |
| `--no-cache` | Bypass cache read/write | Uses cache |
| `--quiet` | Suppress summary print | False |

Error modes: `FetchError` on network failures, `ContentTypeError` if the remote endpoint returns a non-text payload, `NormalizationError` when the CSV is missing expected columns after renaming, and Pandera validation errors for out-of-range values. Cache files are keyed by URL + parameters and stored under `cache_dir/digest.csv`; set `EUROMILLIONS_CACHE_DIR` to change the location globally.

Recipes

- Full history CSV: `python -m euromillions.get_draws --out data/euromillions.csv --append`
- Year-limited: `python -m euromillions.get_draws --from 2023-01-01 --to 2024-12-31 --out data/euromillions_2023_2024.csv`
- Incremental nightly refresh (keeps cache warm): `python -m euromillions.get_draws --from 2024-01-01 --out data/euromillions.csv --append --cache-dir .cache/euromillions`
- In-memory use (no write): `from euromillions.get_draws import fetch_and_normalize; df = fetch_and_normalize().dataframe`

## Labs

- `grok.py`: transformer-based dual-sequence lab; run `python grok.py` after placing data in `data/`. See `labs/README.md` for inputs/outputs and baseline reproduction notes.
- `euromillions_agent/`: agent + discriminator + grok + CEM mixer; entrypoints documented in `euromillions_agent/README.md`. Treat outputs as experimental and keep PRs isolated.
- `euromillions/roi.py`: labelled as a lab for EV gating and walk-forward bankroll simulations; not wired into the CLI yet.

## Examples & Docs

- `examples/euromillions_hello_world.ipynb`: fetch history, compute marginal number/star frequencies, score synthetic tickets, and plot simple distributions.
- `examples/README.md`: notebook roadmap and short how-to-run notes.
- `tests/README.md`: what the suite covers and how long it takes.

## Research Ideas

- Frequency baselines: marginal counts + recency decay vs. naive uniform draws.
- Signal detection: chi-square or permutation tests on co-occurrence matrices to spot weak structure.
- EV gating: simple prize-table EV vs. bankroll simulations before ranking tickets.
- Walk-forward bankroll: rolling windows with risk caps to mimic “live” play.
- Bayesian sanity checks: posterior over hit rates with conjugate priors; flag drift in draws.

## Contributing & Community

Open a PR or start a Discussion if you want to add new lotteries, tighten docs, or port labs. See `CONTRIBUTING.md` for “good first experiments” and documentation tasks. Please keep experiments isolated, document inputs/outputs, and add tests for new behaviours. Tag releases when milestones land so downstream users can pin versions and cite specific URIs.

## License

[MIT](LICENSE)

## Citation

If you use this work in your research, please cite:
```
@software{Lotteries2025,
  -title={Inference in European Lotteries},
  -author={kugguk2022},
  -year={2025},
  -url={https://github.com/kugguk2022/lotteries}
}
```
