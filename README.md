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

## EuroMillions `get_draws`

Fetches historical EuroMillions draws from the MerseyWorld CSV endpoint, caches responses locally, retries transient failures, and normalizes the output. The script deduplicates on `draw_date` and validates via Pandera before writing a CSV.

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

`euromillions/schema.py` defines the canonical Pandera schema. `validate_df` coerces `draw_date` to timezone-naive timestamps and enforces number ranges (1-50 for balls, 1-12 for stars). Tests cover both the schema and CSV normalization pipeline.

## EuroMillions ROI (Planned)

**Not implemented yet -- CLI will error if run.**

`euromillions/roi.py` will host walk-forward bankroll simulations, EV gating, and ticket ranking. The CLI entry point will be exposed once the module is production-ready.

## Testing

Minimal smoke tests live in `tests/`. They do not require network access and focus on schema and normalization behaviour. Extend with dataset-specific fixtures as new functionality lands.

## Legacy R Scripts

R notebooks and `.r` files remain for historical reference but are deprecated. Prefer the Python pipelines when adding new work.

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
