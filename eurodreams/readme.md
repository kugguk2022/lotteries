pip install requests

# All history (2023→present) to CSV
python3 eurodreams_get_draws.py --out eurodreams_all.csv

# Limit by year range
python3 eurodreams_get_draws.py --out eurodreams_2023_2025.csv --start-year 2023 --end-year 2025

# JSON output instead of CSV
python3 eurodreams_get_draws.py --format json --out eurodreams_all.json

# Force a specific source
python3 eurodreams_get_draws.py --source irish      # primary full archive
python3 eurodreams_get_draws.py --source euro       # per-year pages with draw_code
python3 eurodreams_get_draws.py --source lottery_ie # last ~90 days
