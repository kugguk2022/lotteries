# 🎲 Lotteries — analysis & guessing toolkit

Small research playground for lottery data (EuroMillions, Totoloto, and EuroDreams/"Edreams").  
Focus: clean datasets, quick feature engineering, sanity/randomness checks, and baseline models for ranking number combinations. **This is research code—use responsibly.**

---

## 🔧 Issues Fixed (Recent Update)

✅ **Deprecated pandas method**: Fixed `fillna(method='ffill')` → `fillna().ffill()`  
✅ **Hard-coded Linux paths**: Replaced with relative/flexible paths that work on Windows  
✅ **Missing dependencies**: Added comprehensive `requirements.txt`  
✅ **File path issues**: Added error handling for missing data files  
✅ **Cross-platform compatibility**: Fixed PowerShell command compatibility  
✅ **Code maintainability**: Improved error handling and documentation  

## ✨ What's new
- **2025‑10‑01** — Major code fixes and cross-platform compatibility improvements
- **2025‑08‑22** — Added docs & examples for `euromillions/get_draws` (CSV/JSON exporter).  
- **Planned** — `euromillions/roi.py`: walk‑forward backtests, EV gating, bankroll/ROI metrics.ries — analysis & guessing toolkit

Small research playground for lottery data (EuroMillions, Totoloto, and EuroDreams/“Edreams”).  
Focus: clean datasets, quick feature engineering, sanity/randomness checks, and baseline models for ranking number combinations. **This is research code—use responsibly.**

---

## ✨ What’s new
- **2025‑08‑22** — Added docs & examples for `euromillions/get_draws` (CSV/JSON exporter).  
- **Planned** — `euromillions/roi.py`: walk‑forward backtests, EV gating, bankroll/ROI metrics.

---

## 📁 Project structure

```
lotteries/
├─ grok.py                          # Main neural network analysis script
├─ setup.py                         # Project setup and dependency installer
├─ requirements.txt                 # Python dependencies
├─ FIXES_SUMMARY.md                 # Documentation of recent fixes
├─ LICENSE                          # MIT
├─ README.md                        # This file
├─ eurodreams/                      # EuroDreams analysis
│  └─ Edreams.py                    # EuroDreams ticket generation algorithm
├─ euromillions/                    # EuroMillions analysis suite
│  ├─ euromillions_2016_2025.csv    # Historical data (2016-2025)
│  ├─ euromillions_values.txt       # Heuristic weights/prior values
│  ├─ euromillions.r                # R analysis script
│  ├─ get_draws.py                  # Fetch & normalize EuroMillions draw history
│  ├─ grok.py                       # Neural network analysis for EuroMillions
│  └─ roi.py                        # ROI analysis and backtesting lab
├─ euromillions_agent/              # EuroMillions scraping and prize tools
│  ├─ fetch_prizes_range.py         # Fetch prize ranges
│  ├─ fetch_prizes.py               # Fetch individual prize breakdowns
│  └─ lotto_lab.py                  # Complete analysis laboratory
└─ totoloto/                        # Portuguese Totoloto analysis
   ├─ grok.py                       # Neural network analysis for Totoloto
   ├─ lottery_values.txt            # Heuristic weights/prior values
   └─ totoloto.r                    # R analysis script
```

> The `*_values.txt` files are simple newline-separated lists you can tweak; scripts may read them as weights, seeds, or “value tables” depending on experiment.

---

## 🚀 Quickstart

### 1) Clone & create a virtual environment
```bash
git clone https://github.com/kugguk2022/lotteries
cd lotteries

# Python 3.10+ recommended
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install minimal dependencies
```bash
python -m pip install -U pip
pip install numpy pandas scipy scikit-learn statsmodels matplotlib requests python-dateutil
```

---

## 🇪🇺 EuroMillions: `get_draws` (now available)

Fetches historical EuroMillions draws, normalizes schema, and saves to **CSV** or **JSON**.  
Intended as a clean, reproducible source for downstream feature engineering and backtests.

### CLI
```bash
python euromillions/get_draws.py --help
```

### Examples
```bash
# 1) Full history → CSV
python euromillions/get_draws.py --out data/euromillions.csv

# 2) Date‑bounded export
python euromillions/get_draws.py --from 2016-01-01 --to 2025-08-22 --out data/eur_2016_2025.csv

# 3) JSON output
python euromillions/get_draws.py --format json --out data/euromillions.json

# 4) Append to existing (dedup by draw_date)
python euromillions/get_draws.py --out data/euromillions.csv --append
```

> **Notes**
> - Default output is CSV unless `--format json` is provided.  
> - Script performs light validation (range checks, dedup by `draw_date`, ascending sort).  
> - If a data source requires a key or alternate URL, update constants at the top of `get_draws.py`.

### Output schema (CSV)
| column        | type   | notes |
|---------------|--------|-------|
| `draw_date`   | date   | ISO `YYYY-MM-DD` |
| `n1..n5`      | int    | 5 main numbers |
| `star1,star2` | int    | 2 Lucky Stars |
| `jackpot`     | float? | if available (EUR) |
| `rollover`    | int?   | optional |
| `source_url`  | str    | provenance (if captured) |
| `scraped_at`  | datetime | UTC timestamp of fetch |

> Columns with `?` appear only if available from the upstream source.

---

## 📊 Feature engineering (shared ideas)

- Frequencies, gaps, hot/cold, digit sums, parity, primes, pair/cluster counts
- Residues (mod *k*), wheel residues (e.g., mod 210), repeating patterns
- Randomness checks: chi-square, runs test, serial correlation; optional OU / random‑matrix style visuals
- Baseline rankers: simple ML (logistic/linear, trees) with strict walk‑forward splits

---

## 💸 EuroMillions ROI module (planned)

`euromillions/roi.py` will provide **walk‑forward** evaluation of simple strategies, bankroll tracking, and EV filters.

**Design (subject to change):**
```bash
python euromillions/roi.py   --csv data/euromillions.csv   --strategy freq            \  # freq | gaps | wheel | hybrid | random
  --window 365               \  # days of history for features
  --tickets 10               \  # tickets per draw
  --budget 5                 \  # € per ticket (adjust to real rules)
  --ev-min 1.2               \  # only place bets if EV >= 1.2 * cost
  --kelly 0.25               \  # optional Kelly fraction on bankroll
  --out out/roi_euromillions.csv
```

**Planned metrics**
- Hit rate (any prize / tiered), mean payout, **ROI**, cumulative P&L
- Max drawdown, time‑to‑recover, volatility
- Baseline vs. random ticket sets
- Per‑strategy leaderboard CSV

**Risk controls**
- EV gate (skip low‑value draws), bankroll caps, Kelly fraction (optional), stop‑loss limits

> Nothing here implies predictability—this is for controlled experiments and to avoid self‑deception (compare vs random baselines).

---

## 🧪 Backtesting discipline (suggested)

1. Split by date (past → future only).  
2. Fit features on a rolling window; generate candidates for the **next** draw.  
3. Record tickets, hits, payouts, and diagnostics.  
4. Repeat forward; compare strategies vs random.  
5. Keep a leaderboard in `out/` (CSV/JSON).

---

## 📦 Data hygiene

- Avoid future leakage (no peeking past the target draw).  
- Keep draw dates in ISO (YYYY‑MM‑DD).  
- Normalize number ranges to the real game rules before comparing.  
- Deduplicate by `draw_date` + full combination when applicable.

---

## 🤝 Contributing

PRs/issues welcome:
- Add clean, reproducible scripts per game.  
- Keep dependencies minimal.  
- Prefer small, self‑contained experiments over monoliths.  
- Document input/output clearly (`--help` should explain flags).

---

## ⚠️ Disclaimer

This code is for **research and education** only. Lotteries are games of chance; nothing here is financial advice. Use at your own risk.

---

## 📜 License

[MIT](LICENSE)
