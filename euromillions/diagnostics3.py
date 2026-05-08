from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import combinations, islice
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from euromillions_agent.phase2_sobol import euler_phi_upto

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = REPO_ROOT / "data" / "euromillions.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "euromillions" / "diagnostics3"
DEFAULT_BATCH_SIZE = 200_000
DEFAULT_MODE = "full7"


@dataclass
class Diagnostics3Summary:
    mode: str
    rows: int
    start_date: str
    end_date: str
    main_n: int
    star_n: int
    pair_dimension: int
    phi_poi_corr: float
    poi_mean: float
    poi_std: float
    glm_family: str
    glm_link: str
    raw_glm_intercept: float
    raw_glm_phi_slope: float
    pruned_glm_intercept: float
    pruned_glm_phi_slope: float
    pruned_glm_aic: float
    pruned_glm_deviance: float
    pruned_glm_fitted_mean: float
    pruned_glm_shifted_prediction_mean: float
    pruning_z_threshold: float
    inlier_count: int
    outlier_count: int
    trailing_window_r_compatible: int
    trailing_mean_r_compatible: float
    trailing_mean_last_5: float
    target_score_r_compatible: int
    matching_guess_count: int
    total_combinations_scored: int


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostics 3: Euler-phi GLM and guess extraction for EuroMillions."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY,
        help="Normalized EuroMillions history CSV with draw_date and ball_1..ball_5.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for diagnostics 3 CSV, JSON, and plot outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of 5-ball combinations to score per batch.",
    )
    parser.add_argument(
        "--mode",
        choices=("full7", "main5"),
        default=DEFAULT_MODE,
        help="Use the full 5+2 ticket or the original 5-main-only R shape.",
    )
    parser.add_argument(
        "--outlier-z",
        type=float,
        default=3.5,
        help="Robust z-threshold for Euler-totient residual pruning.",
    )
    return parser.parse_args()


def load_history(history_path: Path) -> pd.DataFrame:
    history = pd.read_csv(history_path, parse_dates=["draw_date"])
    history = history.sort_values("draw_date").drop_duplicates(subset=["draw_date"]).reset_index(drop=True)

    required = [f"ball_{idx}" for idx in range(1, 6)] + [f"star_{idx}" for idx in range(1, 3)]
    missing = [col for col in required if col not in history.columns]
    if missing:
        raise ValueError(
            f"History file is missing expected columns: {missing}. "
            "Expected normalized columns ball_1..ball_5 and star_1..star_2."
        )
    return history


@dataclass
class PairFeatureOut:
    output_pairs: np.ndarray
    poi: np.ndarray
    pair_counts: np.ndarray


def build_pair_features_generic(
    draws: np.ndarray,
    universe_size: int,
    *,
    include_current: bool = True,
) -> PairFeatureOut:
    draws = np.asarray(draws, dtype=int)
    n_rows, width = draws.shape
    pair_idx = list(combinations(range(width), 2))
    pair_counts = np.zeros((universe_size + 1, universe_size + 1), dtype=np.int32)
    output_pairs = np.zeros((n_rows, len(pair_idx)), dtype=np.int32)

    for row_idx in range(n_rows):
        draw = np.sort(draws[row_idx])
        if include_current:
            for a, b in combinations(draw.tolist(), 2):
                pair_counts[a, b] += 1
                pair_counts[b, a] += 1

        for pair_pos, (left_idx, right_idx) in enumerate(pair_idx):
            left = int(draw[left_idx])
            right = int(draw[right_idx])
            output_pairs[row_idx, pair_pos] = pair_counts[left, right]

        if not include_current:
            for a, b in combinations(draw.tolist(), 2):
                pair_counts[a, b] += 1
                pair_counts[b, a] += 1

    poi = output_pairs.sum(axis=1).astype(float)
    return PairFeatureOut(output_pairs=output_pairs, poi=poi, pair_counts=pair_counts)


def encode_full7_draws(history: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    mains = history[[f"ball_{idx}" for idx in range(1, 6)]].to_numpy(dtype=int)
    stars = history[[f"star_{idx}" for idx in range(1, 3)]].to_numpy(dtype=int)
    main_n = int(mains.max())
    star_n = int(stars.max())
    encoded_stars = stars + main_n
    draws = np.concatenate([mains, encoded_stars], axis=1)
    return draws, main_n, star_n


def robust_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad < 1e-12:
        return np.zeros_like(arr)
    return 0.6744897501960817 * (arr - med) / mad


def fit_phi_glm(poi: np.ndarray, phi_current: np.ndarray, phi_shifted: np.ndarray):
    exog = pd.DataFrame({"const": 1.0, "g1": phi_current})
    glm = sm.GLM(
        poi,
        exog,
        family=sm.families.Gaussian(link=sm.families.links.Identity()),
    ).fit()
    fitted = np.asarray(glm.fittedvalues, dtype=float)
    shifted_exog = pd.DataFrame({"const": 1.0, "g1": phi_shifted})
    shifted_pred = np.asarray(glm.predict(shifted_exog), dtype=float)
    return glm, fitted, shifted_pred


def fit_pruned_phi_glm(
    poi: np.ndarray,
    phi_current: np.ndarray,
    phi_shifted: np.ndarray,
    *,
    outlier_z: float,
):
    raw_glm, raw_fitted, _ = fit_phi_glm(poi, phi_current, phi_shifted)
    raw_resid = poi - raw_fitted
    resid_z = robust_zscore(raw_resid)
    inlier_mask = np.abs(resid_z) <= outlier_z
    if int(inlier_mask.sum()) < max(20, len(poi) // 4):
        inlier_mask = np.ones(len(poi), dtype=bool)

    pruned_glm, _, shifted_pred = fit_phi_glm(
        poi[inlier_mask],
        phi_current[inlier_mask],
        phi_shifted,
    )
    full_exog = pd.DataFrame({"const": 1.0, "g1": phi_current})
    pruned_fitted = np.asarray(pruned_glm.predict(full_exog), dtype=float)
    return raw_glm, pruned_glm, pruned_fitted, shifted_pred, resid_z, inlier_mask


def score_batch(pair_counts: np.ndarray, combos: np.ndarray) -> np.ndarray:
    return (
        pair_counts[combos[:, 0], combos[:, 1]]
        + pair_counts[combos[:, 0], combos[:, 2]]
        + pair_counts[combos[:, 0], combos[:, 3]]
        + pair_counts[combos[:, 0], combos[:, 4]]
        + pair_counts[combos[:, 1], combos[:, 2]]
        + pair_counts[combos[:, 1], combos[:, 3]]
        + pair_counts[combos[:, 1], combos[:, 4]]
        + pair_counts[combos[:, 2], combos[:, 3]]
        + pair_counts[combos[:, 2], combos[:, 4]]
        + pair_counts[combos[:, 3], combos[:, 4]]
    ).astype(int)


def find_matching_guesses(
    pair_counts: np.ndarray,
    main_n: int,
    target_score: int,
    batch_size: int,
) -> tuple[pd.DataFrame, int]:
    combo_iter = combinations(range(1, main_n + 1), 5)
    matched_frames: list[pd.DataFrame] = []
    total = 0

    while True:
        batch = list(islice(combo_iter, batch_size))
        if not batch:
            break

        combos = np.asarray(batch, dtype=np.int16)
        scores = score_batch(pair_counts, combos)
        start_index = total + 1
        total += len(combos)

        hit_mask = scores == target_score
        if not np.any(hit_mask):
            continue

        matched = combos[hit_mask]
        matched_scores = scores[hit_mask]
        matched_positions = np.flatnonzero(hit_mask) + start_index
        matched_frames.append(
            pd.DataFrame(
                {
                    "combination_index": matched_positions.astype(np.int64),
                    "ball_1": matched[:, 0].astype(int),
                    "ball_2": matched[:, 1].astype(int),
                    "ball_3": matched[:, 2].astype(int),
                    "ball_4": matched[:, 3].astype(int),
                    "ball_5": matched[:, 4].astype(int),
                    "score": matched_scores.astype(int),
                }
            )
        )

    if matched_frames:
        return pd.concat(matched_frames, ignore_index=True), total
    return pd.DataFrame(
        columns=["combination_index", "ball_1", "ball_2", "ball_3", "ball_4", "ball_5", "score"]
    ), total


def find_matching_full7_guesses(
    pair_counts: np.ndarray,
    *,
    main_n: int,
    star_n: int,
    target_score: int,
    batch_size: int,
) -> tuple[pd.DataFrame, int]:
    main_iter = combinations(range(1, main_n + 1), 5)
    star_slots = np.arange(main_n + 1, main_n + star_n + 1, dtype=np.int16)
    star_pair_idx = np.asarray(list(combinations(range(star_n), 2)), dtype=np.int16)
    star_encoded = star_slots[star_pair_idx]
    star_display = star_pair_idx + 1
    star_scores = pair_counts[star_encoded[:, 0], star_encoded[:, 1]].astype(int)
    tickets_per_main = len(star_pair_idx)

    matched_frames: list[pd.DataFrame] = []
    processed_mains = 0

    while True:
        batch = list(islice(main_iter, batch_size))
        if not batch:
            break

        mains = np.asarray(batch, dtype=np.int16)
        main_scores = score_batch(pair_counts, mains)
        main_star_sum = pair_counts[mains[:, :, None], star_slots[None, None, :]].sum(axis=1).astype(int)

        for star_pos, (star_left_idx, star_right_idx) in enumerate(star_pair_idx):
            total_scores = (
                main_scores
                + star_scores[star_pos]
                + main_star_sum[:, star_left_idx]
                + main_star_sum[:, star_right_idx]
            )
            hit_mask = total_scores == target_score
            if not np.any(hit_mask):
                continue

            hit_positions = np.flatnonzero(hit_mask)
            combo_indices = (
                (processed_mains + hit_positions).astype(np.int64) * tickets_per_main
                + int(star_pos)
                + 1
            )
            matched_mains = mains[hit_mask]
            matched_frames.append(
                pd.DataFrame(
                    {
                        "combination_index": combo_indices,
                        "ball_1": matched_mains[:, 0].astype(int),
                        "ball_2": matched_mains[:, 1].astype(int),
                        "ball_3": matched_mains[:, 2].astype(int),
                        "ball_4": matched_mains[:, 3].astype(int),
                        "ball_5": matched_mains[:, 4].astype(int),
                        "star_1": np.full(hit_positions.shape, int(star_display[star_pos, 0]), dtype=int),
                        "star_2": np.full(hit_positions.shape, int(star_display[star_pos, 1]), dtype=int),
                        "score": total_scores[hit_mask].astype(int),
                    }
                )
            )

        processed_mains += len(mains)

    total = processed_mains * tickets_per_main
    if matched_frames:
        return pd.concat(matched_frames, ignore_index=True), total
    return pd.DataFrame(
        columns=[
            "combination_index",
            "ball_1",
            "ball_2",
            "ball_3",
            "ball_4",
            "ball_5",
            "star_1",
            "star_2",
            "score",
        ]
    ), total


def save_plot(
    frame: pd.DataFrame,
    *,
    phi_current: np.ndarray,
    fitted: np.ndarray,
    shifted_pred: np.ndarray,
    inlier_mask: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), constrained_layout=True)

    axes[0].plot(frame["draw_date"], frame["poi"], color="steelblue", lw=0.8, label="poi")
    axes[0].plot(frame["draw_date"], fitted, color="navy", lw=1.4, label="phi-pruned glm fitted")
    if np.any(~inlier_mask):
        axes[0].scatter(
            frame.loc[~inlier_mask, "draw_date"],
            frame.loc[~inlier_mask, "poi"],
            color="tomato",
            s=22,
            label="pruned outliers",
            zorder=3,
        )
    axes[0].set_title("Diagnostics 3: poi vs Euler-totient-pruned Gaussian GLM")
    axes[0].set_ylabel("poi")
    axes[0].legend()

    order = np.argsort(phi_current)
    axes[1].scatter(
        phi_current[inlier_mask],
        frame.loc[inlier_mask, "poi"],
        s=10,
        alpha=0.55,
        color="steelblue",
        label="inliers",
    )
    if np.any(~inlier_mask):
        axes[1].scatter(
            phi_current[~inlier_mask],
            frame.loc[~inlier_mask, "poi"],
            s=18,
            alpha=0.8,
            color="tomato",
            label="pruned outliers",
        )
    axes[1].plot(phi_current[order], fitted[order], color="tomato", lw=2, label="glm fit")
    axes[1].set_title("Euler phi vs poi")
    axes[1].set_xlabel("phi(t)")
    axes[1].set_ylabel("poi")
    axes[1].legend()

    axes[2].plot(
        frame["draw_date"].iloc[1:],
        shifted_pred[:-1],
        color="purple",
        lw=1.2,
        label="predict using phi(t+1)",
    )
    axes[2].scatter(
        frame["draw_date"].iloc[-1],
        shifted_pred[-1],
        color="tomato",
        s=45,
        label="out-of-sample phi(T+1) response",
        zorder=3,
    )
    axes[2].set_title("Shifted response predictions from g[2:length(g)]")
    axes[2].set_xlabel("draw date")
    axes[2].set_ylabel("predicted response")
    axes[2].legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    history_path = resolve_repo_path(args.history)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(history_path)
    main_n = int(history[[f"ball_{idx}" for idx in range(1, 6)]].to_numpy(dtype=int).max())
    star_n = int(history[[f"star_{idx}" for idx in range(1, 3)]].to_numpy(dtype=int).max())

    if args.mode == "full7":
        draws, _, _ = encode_full7_draws(history)
        pair_dimension = draws.shape[1]
        features = build_pair_features_generic(draws, universe_size=main_n + star_n, include_current=True)
    else:
        draws = history[[f"ball_{idx}" for idx in range(1, 6)]].to_numpy(dtype=int)
        pair_dimension = draws.shape[1]
        features = build_pair_features_generic(draws, universe_size=main_n, include_current=True)

    poi = features.poi.astype(float)
    phi_full = euler_phi_upto(len(poi) + 1).astype(float)
    phi_current = phi_full[:-1]
    phi_shifted = phi_full[1:]

    raw_glm, pruned_glm, fitted, shifted_pred, resid_z, inlier_mask = fit_pruned_phi_glm(
        poi,
        phi_current,
        phi_shifted,
        outlier_z=args.outlier_z,
    )
    corr = float(np.corrcoef(phi_current, poi)[0, 1])

    r_window = min(6, len(poi))
    trailing_mean_r = float(np.mean(poi[-r_window:]))
    trailing_mean_last_5 = float(np.mean(poi[-min(5, len(poi)) :]))
    target_score = int(round(trailing_mean_r))

    series = pd.DataFrame(
        {
            "draw_date": history["draw_date"],
            "poi": poi,
            "phi": phi_current,
            "phi_resid_robust_z": resid_z,
            "phi_inlier": inlier_mask.astype(int),
            "glm_fitted_pruned": fitted,
            "glm_resid": poi - fitted,
            "phi_next": phi_shifted,
            "glm_predicted_next_response": shifted_pred,
        }
    )
    series.to_csv(out_dir / "diagnostics3_series.csv", index=False)

    if args.mode == "full7":
        matches, total_scored = find_matching_full7_guesses(
            features.pair_counts,
            main_n=main_n,
            star_n=star_n,
            target_score=target_score,
            batch_size=args.batch_size,
        )
    else:
        matches, total_scored = find_matching_guesses(
            features.pair_counts,
            main_n=main_n,
            target_score=target_score,
            batch_size=args.batch_size,
        )
    matches.to_csv(out_dir / "diagnostics3_guesses.csv", index=False)

    save_plot(
        series,
        phi_current=phi_current,
        fitted=fitted,
        shifted_pred=shifted_pred,
        inlier_mask=inlier_mask,
        out_path=out_dir / "diagnostics3_glm.png",
    )

    summary = Diagnostics3Summary(
        mode=args.mode,
        rows=len(series),
        start_date=history["draw_date"].min().date().isoformat(),
        end_date=history["draw_date"].max().date().isoformat(),
        main_n=main_n,
        star_n=star_n,
        pair_dimension=pair_dimension,
        phi_poi_corr=corr,
        poi_mean=float(poi.mean()),
        poi_std=float(poi.std(ddof=0)),
        glm_family="gaussian",
        glm_link="identity",
        raw_glm_intercept=float(raw_glm.params["const"]),
        raw_glm_phi_slope=float(raw_glm.params["g1"]),
        pruned_glm_intercept=float(pruned_glm.params["const"]),
        pruned_glm_phi_slope=float(pruned_glm.params["g1"]),
        pruned_glm_aic=float(pruned_glm.aic),
        pruned_glm_deviance=float(pruned_glm.deviance),
        pruned_glm_fitted_mean=float(fitted.mean()),
        pruned_glm_shifted_prediction_mean=float(shifted_pred.mean()),
        pruning_z_threshold=float(args.outlier_z),
        inlier_count=int(inlier_mask.sum()),
        outlier_count=int((~inlier_mask).sum()),
        trailing_window_r_compatible=r_window,
        trailing_mean_r_compatible=trailing_mean_r,
        trailing_mean_last_5=trailing_mean_last_5,
        target_score_r_compatible=target_score,
        matching_guess_count=int(len(matches)),
        total_combinations_scored=total_scored,
    )
    (out_dir / "diagnostics3_summary.json").write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print("DIAGNOSTICS 3 — Euler Phi Gaussian GLM")
    print("=" * 70)
    print(
        f"Rows: {summary.rows}  Date range: {summary.start_date} -> {summary.end_date}  "
        f"mode={summary.mode}  pair_dim={summary.pair_dimension}"
    )
    print(f"Correlation(phi, poi): {summary.phi_poi_corr:.4f}")
    print(
        f"GLM: poi ~ phi(t)  family={summary.glm_family}  link={summary.glm_link}  "
        f"raw_slope={summary.raw_glm_phi_slope:.4f}  pruned_slope={summary.pruned_glm_phi_slope:.4f}"
    )
    print(
        f"Euler-totient pruning: z<={summary.pruning_z_threshold:.2f}  "
        f"inliers={summary.inlier_count}  outliers={summary.outlier_count}"
    )
    print(
        f"R-compatible trailing mean window={summary.trailing_window_r_compatible} draws  "
        f"mean={summary.trailing_mean_r_compatible:.3f}  "
        f"target score={summary.target_score_r_compatible}"
    )
    print(f"Matching guesses: {summary.matching_guess_count} out of {summary.total_combinations_scored}")
    print(f"Wrote: {out_dir / 'diagnostics3_series.csv'}")
    print(f"Wrote: {out_dir / 'diagnostics3_guesses.csv'}")
    print(f"Wrote: {out_dir / 'diagnostics3_summary.json'}")
    print(f"Wrote: {out_dir / 'diagnostics3_glm.png'}")


if __name__ == "__main__":
    main()
