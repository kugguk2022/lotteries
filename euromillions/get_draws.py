from __future__ import annotations

import argparse
import hashlib
import json
from io import StringIO
from pathlib import Path
from typing import Dict

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

from .schema import validate_df

PRIMARY_URL = "https://www.merseyworld.com/euromillions/resultsArchive.php?format=csv"
CACHE_DIR = Path(".cache/euromillions")


def _session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "lotteries/0.1 (github.com/kugguk2022/lotteries)"})
    return session


def _cache_key(url: str, params: Dict[str, str]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = f"{url}|{json.dumps(params, sort_keys=True)}".encode()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.csv"


def fetch_raw_csv(date_from: str | None = None, date_to: str | None = None) -> str:
    params: Dict[str, str] = {}
    if date_from:
        params["from"] = date_from
    if date_to:
        params["to"] = date_to

    cache_file = _cache_key(PRIMARY_URL, params)
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    with _session().get(PRIMARY_URL, params=params, timeout=5) as response:
        response.raise_for_status()
        if "text" not in response.headers.get("Content-Type", ""):
            raise RuntimeError(f"Unexpected content type: {response.headers.get('Content-Type')}")
        text = response.text
        cache_file.write_text(text, encoding="utf-8")
        return text


def normalize(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(csv_text))

    rename = {}
    for column in df.columns:
        norm = column.lower().strip().replace(" ", "_")
        rename[column] = norm
    df = df.rename(columns=rename)

    mapping = {
        "date": "draw_date",
        "draw_date": "draw_date",
        "ball1": "ball_1",
        "ball_1": "ball_1",
        "ball2": "ball_2",
        "ball_2": "ball_2",
        "ball3": "ball_3",
        "ball_3": "ball_3",
        "ball4": "ball_4",
        "ball_4": "ball_4",
        "ball5": "ball_5",
        "ball_5": "ball_5",
        "lucky_star1": "star_1",
        "lucky_star_1": "star_1",
        "star_1": "star_1",
        "lucky_star2": "star_2",
        "lucky_star_2": "star_2",
        "star_2": "star_2",
    }
    df = df.rename(columns={key: value for key, value in mapping.items() if key in df.columns})

    keep = ["draw_date", "ball_1", "ball_2", "ball_3", "ball_4", "ball_5", "star_1", "star_2"]
    df = df[keep]
    df = validate_df(df)
    df = df.sort_values("draw_date").drop_duplicates(subset=["draw_date"]).reset_index(drop=True)
    return df


def write_csv(df: pd.DataFrame, out_path: Path, append: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if append and out_path.exists():
        previous = pd.read_csv(out_path, parse_dates=["draw_date"])
        previous = validate_df(previous)
        df = pd.concat([previous, df], axis=0, ignore_index=True)
        df = df.sort_values("draw_date").drop_duplicates(subset=["draw_date"]).reset_index(drop=True)

    df.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EuroMillions draws and write normalized CSV")
    parser.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--append", action="store_true", help="Append/dedup to existing CSV")
    args = parser.parse_args()

    raw_csv = fetch_raw_csv(args.date_from, args.date_to)
    dataframe = normalize(raw_csv)
    write_csv(dataframe, args.out, append=args.append)

    print(f"Wrote {len(dataframe):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
