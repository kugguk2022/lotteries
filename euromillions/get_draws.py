from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .schema import validate_df

PRIMARY_URL = "https://www.merseyworld.com/euromillions/resultsArchive.php?format=csv"
SECONDARY_URL = "https://www.national-lottery.co.uk/results/euromillions/draw-history/csv"
CACHE_DIR = Path(".cache/euromillions")
_MIN_ROWS_FULL_HISTORY = 200


class FetchError(RuntimeError):
    """Raised when fetching remote draws fails."""


class ContentTypeError(RuntimeError):
    """Raised when the remote endpoint does not return a text-like payload."""


class NormalizationError(ValueError):
    """Raised when raw CSV cannot be normalized into the expected schema."""


@dataclass(frozen=True)
class FetchResult:
    """Container returned by fetch_and_normalize."""

    raw_csv: str
    dataframe: pd.DataFrame
    cache_path: Path


def _session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "lotteries/0.1 (github.com/kugguk2022/lotteries)"})
    return session


def _cache_dir(custom: Path | None = None) -> Path:
    override = os.environ.get("EUROMILLIONS_CACHE_DIR")
    if custom:
        return custom
    if override:
        return Path(override)
    return CACHE_DIR


def _cache_key(url: str, params: Dict[str, str], cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = f"{url}|{json.dumps(params, sort_keys=True)}".encode()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return cache_dir / f"{digest}.csv"


def fetch_raw_csv(
    date_from: str | None = None,
    date_to: str | None = None,
    *,
    session: Optional[requests.Session] = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    urls: Optional[List[str]] = None,
    min_rows_full_history: int = _MIN_ROWS_FULL_HISTORY,
) -> str:
    """
    Fetch CSV text from EuroMillions sources with retries + on-disk caching.

    Set ``EUROMILLIONS_CACHE_DIR`` or pass ``cache_dir`` to control cache location.
    If ``use_cache`` is False, the network is always hit.
    """

    params: Dict[str, str] = {}
    if date_from:
        params["from"] = date_from
    if date_to:
        params["to"] = date_to

    candidates = urls or [PRIMARY_URL, SECONDARY_URL]
    errors: List[str] = []

    def _looks_like_draw_csv(text: str) -> bool:
        header = text.splitlines()[0].lower() if text else ""
        return "ball1" in header or "ball 1" in header or "lucky" in header or "star" in header

    def _looks_complete(text: str) -> bool:
        if date_from or date_to:
            return True
        return _looks_like_draw_csv(text) and (text.count("\n") + 1 >= min_rows_full_history)

    best_valid_text: str | None = None

    for url in candidates:
        cache_file = _cache_key(url, params, _cache_dir(cache_dir))
        if use_cache and cache_file.exists():
            cached = cache_file.read_text(encoding="utf-8")
            if _looks_complete(cached):
                return cached
            if _looks_like_draw_csv(cached):
                best_valid_text = best_valid_text or cached
                return cached

        try:
            with (session or _session()).get(url, params=params, timeout=5) as response:
                response.raise_for_status()
                if "text" not in response.headers.get("Content-Type", ""):
                    raise ContentTypeError(f"Unexpected content type: {response.headers.get('Content-Type')}")
                text = response.text
                if not _looks_like_draw_csv(text):
                    errors.append(f"{url}: unexpected payload (missing draw headers)")
                    continue
                cache_file.write_text(text, encoding="utf-8")
        except requests.RequestException as exc:  # network / HTTP errors
            errors.append(f"{url}: {exc}")
            if cache_file.exists() and use_cache:
                cached = cache_file.read_text(encoding="utf-8")
                if _looks_complete(cached):
                    return cached
                if _looks_like_draw_csv(cached):
                    best_valid_text = best_valid_text or cached
            continue

        if _looks_complete(text):
            return text
        if _looks_like_draw_csv(text):
            best_valid_text = best_valid_text or text
        errors.append(f"{url}: insufficient rows ({text.count(chr(10)) + 1})")

    if best_valid_text:
        return best_valid_text
    raise FetchError("Failed to fetch EuroMillions CSV; attempts: " + "; ".join(errors))


def normalize(csv_text: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception as exc:
        raise NormalizationError(f"Failed to parse CSV payload: {exc}") from exc

    rename = {}
    for column in df.columns:
        norm = column.lower().strip().replace(" ", "_")
        rename[column] = norm
    df = df.rename(columns=rename)

    mapping = {
        "date": "draw_date",
        "draw_date": "draw_date",
        "drawdate": "draw_date",
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
    missing = [col for col in keep if col not in df.columns]
    if missing:
        raise NormalizationError(f"Missing expected columns after rename: {missing}")
    df = df[keep]

    df = validate_df(df)
    df = df.sort_values("draw_date").drop_duplicates(subset=["draw_date"]).reset_index(drop=True)
    return df


def write_csv(df: pd.DataFrame, out_path: Path, append: bool) -> None:
    """Persist normalized draws to CSV, optionally merging with an existing file."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if append and out_path.exists():
        previous = pd.read_csv(out_path, parse_dates=["draw_date"])
        previous = validate_df(previous)
        df = pd.concat([previous, df], axis=0, ignore_index=True)
        df = df.sort_values("draw_date").drop_duplicates(subset=["draw_date"]).reset_index(drop=True)

    df.to_csv(out_path, index=False)


def fetch_and_normalize(
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    out_path: Path | None = None,
    append: bool = False,
    session: Optional[requests.Session] = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> FetchResult:
    """High-level helper used by the CLI and tests."""

    raw_csv = fetch_raw_csv(
        date_from=date_from,
        date_to=date_to,
        session=session,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    dataframe = normalize(raw_csv)
    cache_params = {k: v for k, v in {"from": date_from, "to": date_to}.items() if v}
    cache_path = _cache_key(PRIMARY_URL, cache_params, _cache_dir(cache_dir))

    if out_path:
        write_csv(dataframe, out_path, append=append)
    return FetchResult(raw_csv=raw_csv, dataframe=dataframe, cache_path=cache_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EuroMillions draws and write normalized CSV")
    parser.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--append", action="store_true", help="Append/dedup to existing CSV")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Override cache directory (.cache/euromillions)")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Bypass local cache on fetch")
    parser.add_argument("--quiet", action="store_true", help="Suppress summary print")
    args = parser.parse_args()

    result = fetch_and_normalize(
        date_from=args.date_from,
        date_to=args.date_to,
        out_path=args.out,
        append=args.append,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
    )

    if not args.quiet:
        print(
            f"Wrote {len(result.dataframe):,} rows -> {args.out} "
            f"(cache: {_cache_dir(args.cache_dir)})"
        )


__all__ = [
    "FetchError",
    "ContentTypeError",
    "NormalizationError",
    "FetchResult",
    "fetch_raw_csv",
    "normalize",
    "write_csv",
    "fetch_and_normalize",
]


if __name__ == "__main__":
    main()
