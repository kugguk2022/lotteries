#!/usr/bin/env python3
"""
Fetch EuroMillions draws, filter 2016–2025 inclusive, and save as one CSV.

Primary source: MerseyWorld Euro -> Winning_index?display=CSV (full history)
Fallback:      euromillions.api.pedromealha.dev /v1/draws (JSON, 2004->present)

Output columns (CSV, chronological order):
draw_no,date,weekday,n1,n2,n3,n4,n5,star1,star2,jackpot,jackpot_wins
"""
import argparse
import csv
import datetime as dt
import json
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import requests

MERSEYWORLD_URL = (
    "https://lottery.merseyworld.com/Euro/Winning_index.html"
    "?display=CSV&order=1&show=1&year=0"
)
API_URL = "https://euromillions.api.pedromealha.dev/v1/draws"

MONTHS = {m: i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)
}

FIELDS = [
    "draw_no","date","weekday","n1","n2","n3","n4","n5","star1","star2","jackpot","jackpot_wins"
]

def _iso(y: int, mmm: str, d: int) -> str:
    return dt.date(y, MONTHS[mmm], d).isoformat()

def _int_or_none(x: str) -> Optional[int]:
    x = x.strip()
    if not x:
        return None
    # keep only digits (jackpot often has commas)
    digits = re.sub(r"[^\d]", "", x)
    return int(digits) if digits else None

def fetch_mersey() -> str:
    r = requests.get(MERSEYWORLD_URL, timeout=30)
    r.raise_for_status()
    return r.text

def mersey_lines(html: str) -> Iterable[str]:
    m = re.search(r"<pre[^>]*>(.*?)</pre>", html, flags=re.S|re.I)
    if not m:
        raise RuntimeError("CSV <pre> block not found on MerseyWorld page")
    pre = m.group(1)
    for ln in (ln.strip() for ln in pre.splitlines()):
        # Header or data rows only (skip explanatory text)
        if ln.startswith("No.,") or re.match(r"^\d+,\s", ln):
            yield ln

def parse_mersey(lines: Iterable[str]) -> List[Dict]:
    import csv as _csv
    rows = list(_csv.reader(lines))
    if not rows or not rows[0][0].startswith("No."):
        raise RuntimeError("Unexpected MerseyWorld header")
    out = []
    for r in rows[1:]:
        # Expected minimal layout:
        # 0 No. 1 Day 2 DD 3 MMM 4 YYYY 5..9 nums 10..11 stars 12 Jackpot 13 Wins
        if len(r) < 14:
            continue
        try:
            date_iso = _iso(int(r[4]), r[3], int(r[2]))
        except Exception:
            continue
        rec = {
            "draw_no": int(r[0]),
            "date": date_iso,
            "weekday": r[1].strip(),
            "n1": int(r[5]), "n2": int(r[6]), "n3": int(r[7]), "n4": int(r[8]), "n5": int(r[9]),
            "star1": int(r[10]), "star2": int(r[11]),
            "jackpot": _int_or_none(r[12]),
            "jackpot_wins": _int_or_none(r[13]),
        }
        out.append(rec)
    return out

def fetch_api() -> List[Dict]:
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    out = []
    for d in data:
        date_iso = d.get("draw_date") or d.get("date")  # API uses draw_date
        if not date_iso:
            continue
        date_iso = date_iso[:10]
        nums = d.get("numbers") or []
        stars = d.get("stars") or []
        if len(nums) != 5 or len(stars) != 2:
            continue
        out.append({
            "draw_no": int(d.get("id") or d.get("draw_id")),
            "date": date_iso,
            "weekday": d.get("day_of_week") or "",
            "n1": int(nums[0]), "n2": int(nums[1]), "n3": int(nums[2]),
            "n4": int(nums[3]), "n5": int(nums[4]),
            "star1": int(stars[0]), "star2": int(stars[1]),
            "jackpot": d.get("jackpot_eur"),
            "jackpot_wins": d.get("jackpot_winners"),
        })
    return out

def filter_year_range(recs: List[Dict], start_year: int, end_year: int) -> List[Dict]:
    out = []
    for r in recs:
        y = int(r["date"][:4])
        if start_year <= y <= end_year:
            out.append(r)
    return out

def dedupe_keep_latest(recs: List[Dict]) -> List[Dict]:
    """
    Rare protection: if both sources are combined, keep one per draw_no (prefer later date/complete fields).
    """
    best: Dict[int, Dict] = {}
    for r in recs:
        k = r["draw_no"]
        prev = best.get(k)
        if not prev:
            best[k] = r
            continue
        # Prefer the one with jackpot info if available, or later date
        def score(x: Dict) -> Tuple[int, str]:
            return (1 if x.get("jackpot") is not None else 0, x["date"])
        best[k] = max([prev, r], key=score)
    return list(best.values())

def save_csv(recs: List[Dict], path: str) -> None:
    recs_sorted = sorted(recs, key=lambda x: (x["date"], x["draw_no"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in recs_sorted:
            w.writerow({k: r.get(k) for k in FIELDS})

def main():
    ap = argparse.ArgumentParser(description="Compile EuroMillions draws 2016–2025 into one CSV (chronological).")
    ap.add_argument("--out", default="euromillions_2016_2025.csv",
                    help="Output CSV file (default: euromillions_2016_2025.csv)")
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--source", choices=["auto","mersey","api"], default="auto",
                    help="Data source preference: auto (default), mersey, or api")
    args = ap.parse_args()

    recs: List[Dict] = []

    try_mersey = args.source in ("auto","mersey")
    try_api = args.source in ("auto","api")

    err_msgs = []

    if try_mersey:
        try:
            html = fetch_mersey()
            lines = list(mersey_lines(html))
            recs_m = parse_mersey(lines)
            recs.extend(recs_m)
        except Exception as e:
            err_msgs.append(f"MerseyWorld failed: {e}")

    if (not recs) and try_api:
        try:
            recs_api = fetch_api()
            recs.extend(recs_api)
        except Exception as e:
            err_msgs.append(f"API failed: {e}")

    if not recs:
        raise SystemExit("No records fetched.\n" + "\n".join(err_msgs))

    # If both sources were used (auto mode) we might have duplicates -> dedupe
    recs = dedupe_keep_latest(recs)

    # Filter year range
    recs = filter_year_range(recs, args.start_year, args.end_year)

    if not recs:
        raise SystemExit("No records in requested year range.")

    save_csv(recs, args.out)

    print(f"OK: wrote {len(recs)} draws to {args.out} "
          f"({recs[0]['date']} → {recs[-1]['date']}).")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Aborted.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
