"""
Public EuroMillions API.

Stable surface:
- EuroMillionsGuess: immutable 5-number + 2-star ticket.
- evaluate_guess: scoring helper returning (ball_hits, star_hits).
- get_draws.normalize: CSV -> normalized DataFrame with schema validation.
- load_history: convenience reader for cached CSVs (wraps pandas).

Everything else in this package should be treated as experimental.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .get_draws import normalize  # re-export for users
from .guess import EuroMillionsGuess, evaluate_guess


def load_history(path: str | Path, *, expect_columns: bool = True) -> pd.DataFrame:
    """
    Load a EuroMillions history CSV into a validated DataFrame.

    Args:
        path: CSV path produced by ``euromillions.get_draws`` or equivalent.
        expect_columns: if True, raise if expected columns are missing.
    """

    df = pd.read_csv(path, parse_dates=["draw_date"])
    if expect_columns:
        expected = {
            "draw_date",
            "ball_1",
            "ball_2",
            "ball_3",
            "ball_4",
            "ball_5",
            "star_1",
            "star_2",
        }
        missing = expected.difference(df.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {sorted(missing)}")
    df["draw_date"] = pd.to_datetime(df["draw_date"], utc=True).dt.tz_convert(None)
    return df


__all__ = ["EuroMillionsGuess", "evaluate_guess", "normalize", "load_history"]
