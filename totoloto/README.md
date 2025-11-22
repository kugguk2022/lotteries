# Totoloto


pip install requests

# Full history since 2000 → CSV
python3 totoloto_get_draws.py --out data/totoloto.csv

# A specific window
python3 totoloto_get_draws.py --out data/totoloto_2015_2025.csv --start-year 2015 --end-year 2025

# JSON output
python3 totoloto_get_draws.py --format json --out data/totoloto.json

# If rules differ (override ranges)
python3 totoloto_get_draws.py --out data/totoloto.csv --ball-range 1 49 --bonus-range 1 13


Legacy R scripts in this folder are deprecated and kept for reference only. Prefer the Python modules in this package for new work.
