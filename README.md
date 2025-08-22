# lotteries
Lotteries guesser

# 🎲 Lotteries — analysis & guessing toolkit

Small research playground for lottery data (EuroMillions, Totoloto, and EuroDreams/“Edreams”).  
Focus: clean datasets, quick feature engineering, sanity/randomness checks, and a few baseline models for ranking number combinations. **This is research code—use responsibly.**

---

## 📁 Project structure

```
lotteries/
├─ Edreams/                   # EuroDreams / Euromilhões Dreams experiments (WIP)
├─ euromillions/              # EuroMillions-specific scripts/notebooks (WIP)
├─ totoloto/                  # Totoloto-specific scripts/notebooks (WIP)
├─ grok.py                    # Shared helpers/experiments
├─ euromillions_values.txt    # Helper values used by heuristics
├─ lottery_values.txt         # Helper values used by heuristics
├─ LICENSE                    # MIT
└─ README.md
```

> Tip: the `*_values.txt` files are simple newline‑separated lists you can tweak; scripts read them as weights, seeds, or “value tables” depending on experiment.

---

## 🚀 Quickstart

### 1) Clone & create a virtual environment
```bash
git clone https://github.com/kugguk2022/lotteries
cd lotteries

# Python 3.10+ recommended
python -m venv .venv
# Windows
.\.venv\Scriptsctivate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install minimal dependencies
> The repo keeps dependencies light so you can mix and match.
```bash
python -m pip install -U pip
pip install numpy pandas scipy scikit-learn statsmodels matplotlib
```

### 3) Bring your data
Place CSVs under each game folder (or anywhere you prefer) with simple schemas like:

- **EuroMillions**:  
  `draw_date, n1, n2, n3, n4, n5, star1, star2`

- **Totoloto** (adjust as needed):  
  `draw_date, n1, n2, n3, n4, n5, bonus`

- **EuroDreams** (adjust as needed):  
  `draw_date, n1, n2, n3, n4, n5, n6, dream`

> Column names aren’t strict—just keep them consistent and update scripts if you rename.

### 4) Run experiments
Most scripts accept `--help`. Typical flows:

```bash
# See what grok.py can do
python grok.py --help

# Example: run a simple ranking or feature dump on EuroMillions data
python euromillions/<script>.py --csv path/to/euromillions.csv --out out/euromillions_run

# Example: Totoloto
python totoloto/<script>.py --csv path/to/totoloto.csv --out out/totoloto_run
```

> If a script doesn’t exist yet, use the notebooks in each folder or create your own runner—this repo is intentionally flexible.

---

## 🧩 What’s inside (methods & ideas)

- **Feature engineering**: frequencies, gaps, hot/cold, residues (mod k), digit sums, parity, primes, pair/cluster counts.
- **Randomness checks**: chi-square, runs test, gap test, serial correlation; optional OU/random‑matrix diagnostics as visuals.
- **Baselines**: simple ML rankers (e.g., logistic/linear baselines, trees) comparing candidate sets; walk‑forward backtests.
- **EV gate (optional)**: only output picks when expected value clears a configurable threshold.
- **Heuristics**: value tables from `*_values.txt` as light priors/weights.

> None of the above implies predictability. These are *sanity tools* to compare heuristics and avoid self‑deception.

---

## 📊 Example: minimal “feature dump”
```bash
python euromillions/features.py   --csv data/euromillions.csv   --report out/eur_features.json   --plots out/eur_plots
```
(If `features.py` isn’t present yet, copy a template from `grok.py` or start a notebook—this repo encourages quick iteration.)

---

## 🧪 Backtesting (suggested)
1. Split by date (strictly past → future).  
2. Fit on history, rank candidates for the next draw.  
3. Record hits + diagnostics.  
4. Repeat rolling.  
5. Compare against random baselines.

A simple leaderboard JSON/CSV in `out/` helps track which ideas survive longer than chance.

---

## 📦 Data hygiene

- Avoid future leakage (don’t compute features using future draws).
- Keep draw dates in ISO (YYYY‑MM‑DD).
- Normalize number ranges to the real game rules before comparing.

---

## 🤝 Contributing

PRs/issues welcome:
- Add clean, reproducible scripts for each game.
- Keep dependencies minimal.
- Prefer small, self‑contained experiments over monoliths.
- Document input/output clearly (`--help` should explain flags).

---

## ⚠️ Disclaimer

This code is for **research and education** only. Lotteries are games of chance; nothing here is financial advice. Use at your own risk.

---

## 📜 License

[MIT](LICENSE)

