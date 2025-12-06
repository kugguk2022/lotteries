from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .schema import EXPECTED_COLUMNS, validate_df

BALLS: Sequence[int] = tuple(range(1, 51))
STARS: Sequence[int] = tuple(range(1, 13))


def _counts(values: pd.Series, max_value: int) -> np.ndarray:
    counts = np.bincount(values.to_numpy(), minlength=max_value + 1)[1:]
    return counts.astype(float)


def probability_tables(history: pd.DataFrame, smoothing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed probability tables for balls and stars."""
    df = validate_df(history)

    ball_counts = np.zeros(len(BALLS), dtype=float)
    for column in ["ball_1", "ball_2", "ball_3", "ball_4", "ball_5"]:
        ball_counts += _counts(df[column], max(BALLS))
    star_counts = _counts(df["star_1"], max(STARS)) + _counts(df["star_2"], max(STARS))

    ball_probs = (ball_counts + smoothing) / (ball_counts + smoothing).sum()
    star_probs = (star_counts + smoothing) / (star_counts + smoothing).sum()
    return ball_probs, star_probs


def _sample_numbers(
    population: Sequence[int], probabilities: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    probabilities = probabilities / probabilities.sum()
    return np.sort(rng.choice(population, size=k, replace=False, p=probabilities))


def generate_candidates(
    history: pd.DataFrame,
    n: int,
    *,
    smoothing: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate `n` candidate tickets using frequency-weighted sampling."""
    ball_probs, star_probs = probability_tables(history, smoothing=smoothing)
    return generate_candidates_from_tables(ball_probs, star_probs, n=n, seed=seed)


def random_candidates(n: int, *, seed: int | None = None) -> pd.DataFrame:
    """Generate `n` uniformly random tickets (baseline)."""
    ball_probs = np.ones(len(BALLS), dtype=float)
    star_probs = np.ones(len(STARS), dtype=float)
    return generate_candidates_from_tables(ball_probs, star_probs, n=n, seed=seed)


def generate_candidates_from_tables(
    ball_probs: np.ndarray, star_probs: np.ndarray, n: int, *, seed: int | None = None
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        balls = _sample_numbers(BALLS, ball_probs, k=5, rng=rng)
        stars = _sample_numbers(STARS, star_probs, k=2, rng=rng)
        rows.append(
            {
                "ball_1": int(balls[0]),
                "ball_2": int(balls[1]),
                "ball_3": int(balls[2]),
                "ball_4": int(balls[3]),
                "ball_5": int(balls[4]),
                "star_1": int(stars[0]),
                "star_2": int(stars[1]),
            }
        )
    return pd.DataFrame(rows, columns=[c for c in EXPECTED_COLUMNS if c != "draw_date"])


def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["draw_date"])
    return validate_df(df)


def save_candidates(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate EuroMillions candidate tickets from historical draws."
    )
    parser.add_argument("--history", type=Path, required=True, help="Path to history CSV")
    parser.add_argument("--n", type=int, default=10, help="Number of candidate tickets")
    parser.add_argument("--out", type=Path, default=None, help="Optional output CSV path")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Additive smoothing weight")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    history = load_history(args.history)
    candidates = generate_candidates(history, n=args.n, smoothing=args.smoothing, seed=args.seed)

    if args.out:
        save_candidates(candidates, args.out)
        print(f"Wrote {len(candidates)} candidates -> {args.out}")
    else:
        print(candidates.to_csv(index=False))


if __name__ == "__main__":
    main()
