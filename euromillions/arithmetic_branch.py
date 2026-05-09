from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import combinations, islice
from math import gcd
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, t as t_dist

from euromillions.diagnostics3 import (
    annotate_match_statistics,
    apply_start_date_cutoff,
    build_pair_features_generic,
    build_pair_z_diagnostics,
    encode_full7_draws,
    find_nearest_full7_guesses,
    load_history,
    write_excel_workbook,
)
from euromillions_agent.phase2_sobol import euler_phi_upto

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = REPO_ROOT / "data" / "euromillions.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "euromillions" / "arithmetic_branch"
DEFAULT_START_DATE = "2016-09-27"
DEFAULT_BATCH_SIZE = 200_000
DEFAULT_THRESHOLD = 0.5


def is_prime_scalar(value: int) -> bool:
    n = int(value)
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    factor = 3
    while factor * factor <= n:
        if n % factor == 0:
            return False
        factor += 2
    return True


@dataclass
class ResidualDistribution:
    family: str
    loc: float
    scale: float
    df: float | None
    aic: float


@dataclass
class BranchFitSummary:
    branch: str
    count: int
    regression_intercept: float
    regression_prev_poi: float
    regression_prev_ratio: float
    regression_prev_coprime: float
    rmse: float
    mae: float
    selected_distribution: ResidualDistribution


@dataclass
class ArithmeticBranchSummary:
    raw_rows: int
    rows: int
    cutoff_start_date: str
    branch_mode: str
    ratio_mode: str
    modulus: int | None
    branch_threshold: float
    current_branch: str
    next_branch: str
    gcd_last: int
    last_coprime: bool
    upper_count: int
    lower_count: int
    prime_ceiling_count: int
    gcd_coprime_share: float
    gcd_rule_match_rate: float
    predicted_next_poi: float
    predicted_score: int
    predicted_interval_80: list[float]
    predicted_interval_95: list[float]
    exact_match_count: int
    exact_matches_saved: bool
    nearest_score_gap: int
    upper_fit: BranchFitSummary
    lower_fit: BranchFitSummary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arithmetic branch selector using Euler totient ratios, GCD transitions, and branch-specific densities."
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument(
        "--ratio-mode",
        choices=("raw", "mod"),
        default="raw",
        help="Use raw phi(n)/n on POI scores or phi(score mod M)/(score mod M).",
    )
    parser.add_argument(
        "--modulus",
        type=int,
        default=None,
        help="Modulo base when --ratio-mode=mod. Defaults to 62 for full 5+2 tickets.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Branch threshold on the totient ratio. upper if ratio >= threshold.",
    )
    parser.add_argument(
        "--branch-mode",
        choices=("classic", "prime-pruned"),
        default="classic",
        help="Use the original upper/lower split or exclude prime-ceiling rows from the upper branch model.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument(
        "--max-save-matches",
        type=int,
        default=100_000,
        help="Save all exact matches only when the exact hit count stays below this bound.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def compute_totient_ratio(
    poi: np.ndarray,
    *,
    ratio_mode: str,
    modulus: int | None,
    default_modulus: int,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    poi_int = np.asarray(poi, dtype=int)
    if ratio_mode == "raw":
        base = poi_int
        phi_lookup = euler_phi_upto(int(base.max()) + 1).astype(float)
        ratio = phi_lookup[base - 1] / base
        return ratio.astype(float), base.astype(int), None

    eff_modulus = int(modulus or default_modulus)
    base = np.mod(poi_int, eff_modulus)
    base = np.where(base == 0, eff_modulus, base).astype(int)
    phi_lookup = euler_phi_upto(eff_modulus + 1).astype(float)
    ratio = phi_lookup[base - 1] / base
    return ratio.astype(float), base.astype(int), eff_modulus


def build_branch_frame(
    history: pd.DataFrame,
    *,
    ratio_mode: str,
    modulus: int | None,
    threshold: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int, int, int | None]:
    draws, main_n, star_n = encode_full7_draws(history)
    features = build_pair_features_generic(draws, universe_size=main_n + star_n, include_current=True)
    poi = features.poi.astype(int)
    ratio, ratio_base, eff_modulus = compute_totient_ratio(
        poi,
        ratio_mode=ratio_mode,
        modulus=modulus,
        default_modulus=main_n + star_n,
    )

    gcd_prev = np.full(len(poi), np.nan, dtype=float)
    coprime_prev = np.zeros(len(poi), dtype=int)
    actual_branch_change = np.zeros(len(poi), dtype=int)
    gcd_expected_change = np.zeros(len(poi), dtype=int)
    branch = np.where(ratio >= threshold, "upper", "lower")
    prime_ceiling = np.fromiter(
        (is_prime_scalar(value) for value in ratio_base),
        dtype=bool,
        count=len(ratio_base),
    )
    if ratio_mode != "raw":
        prime_ceiling[:] = False
    pruned_branch = np.where(prime_ceiling, "prime_ceiling", np.where(branch == "upper", "composite_upper", "lower"))

    for idx in range(1, len(poi)):
        gcd_prev[idx] = gcd(int(poi[idx]), int(poi[idx - 1]))
        coprime_prev[idx] = int(gcd_prev[idx] == 1)
        actual_branch_change[idx] = int(branch[idx] != branch[idx - 1])
        gcd_expected_change[idx] = coprime_prev[idx]

    frame = pd.DataFrame(
        {
            "draw_date": history["draw_date"].to_numpy(),
            "poi": poi.astype(int),
            "ratio_base": ratio_base.astype(int),
            "totient_ratio": ratio.astype(float),
            "branch": branch,
            "prime_ceiling": prime_ceiling.astype(int),
            "pruned_branch": pruned_branch.astype(object),
            "gcd_prev": gcd_prev,
            "coprime_prev": coprime_prev.astype(int),
            "actual_branch_change": actual_branch_change.astype(int),
            "gcd_expected_change": gcd_expected_change.astype(int),
        }
    )
    return frame, features.pair_counts, poi.astype(int), main_n, star_n, eff_modulus


def build_training_frame(branch_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    poi = branch_frame["poi"].to_numpy(dtype=float)
    ratio = branch_frame["totient_ratio"].to_numpy(dtype=float)
    branch = branch_frame["model_branch"].to_numpy(dtype=object)
    coprime_prev = branch_frame["coprime_prev"].to_numpy(dtype=int)

    for idx in range(2, len(branch_frame)):
        if str(branch[idx]) == "prime_ceiling":
            continue
        rows.append(
            {
                "target_idx": idx,
                "target_poi": poi[idx],
                "target_branch": str(branch[idx]),
                "prev_poi": poi[idx - 1],
                "prev_ratio": ratio[idx - 1],
                "prev_coprime": int(coprime_prev[idx - 1]),
            }
        )
    return pd.DataFrame(rows)


def apply_branch_mode(branch_frame: pd.DataFrame, *, branch_mode: str) -> pd.DataFrame:
    frame = branch_frame.copy()
    if branch_mode == "prime-pruned":
        frame["model_branch"] = np.where(frame["pruned_branch"] == "composite_upper", "upper", frame["pruned_branch"])
    else:
        frame["model_branch"] = frame["branch"]
    return frame


def resolve_current_branch(branch_frame: pd.DataFrame, *, branch_mode: str) -> str:
    series = branch_frame["model_branch"].astype(str)
    if branch_mode != "prime-pruned":
        return str(series.iloc[-1])

    usable = series[series != "prime_ceiling"]
    if usable.empty:
        raise ValueError("No composite rows available after prime pruning.")
    return str(usable.iloc[-1])


def fit_residual_distribution(resid: np.ndarray) -> ResidualDistribution:
    arr = np.asarray(resid, dtype=float)
    scale_norm = max(float(arr.std(ddof=0)), 1e-6)
    loc_norm = float(arr.mean())
    ll_norm = float(np.sum(norm.logpdf(arr, loc=loc_norm, scale=scale_norm)))
    aic_norm = 2.0 * 2.0 - 2.0 * ll_norm

    best = ResidualDistribution(
        family="normal",
        loc=loc_norm,
        scale=scale_norm,
        df=None,
        aic=aic_norm,
    )

    try:
        df_t, loc_t, scale_t = t_dist.fit(arr)
        scale_t = max(float(scale_t), 1e-6)
        ll_t = float(np.sum(t_dist.logpdf(arr, df=float(df_t), loc=float(loc_t), scale=scale_t)))
        aic_t = 2.0 * 3.0 - 2.0 * ll_t
        if aic_t < aic_norm:
            best = ResidualDistribution(
                family="student_t",
                loc=float(loc_t),
                scale=scale_t,
                df=float(df_t),
                aic=aic_t,
            )
    except Exception:
        pass

    return best


def fit_branch_model(
    training: pd.DataFrame, branch_name: str
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, BranchFitSummary, np.ndarray]:
    subset = training[training["target_branch"] == branch_name].reset_index(drop=True)
    if len(subset) < 20:
        raise ValueError(f"Not enough rows to fit branch model for {branch_name}.")

    exog = sm.add_constant(subset[["prev_poi", "prev_ratio", "prev_coprime"]], has_constant="add")
    model = sm.OLS(subset["target_poi"], exog).fit()
    fitted = np.asarray(model.fittedvalues, dtype=float)
    actual = subset["target_poi"].to_numpy(dtype=float)
    resid = actual - fitted
    dist = fit_residual_distribution(resid)

    summary = BranchFitSummary(
        branch=branch_name,
        count=int(len(subset)),
        regression_intercept=float(model.params["const"]),
        regression_prev_poi=float(model.params["prev_poi"]),
        regression_prev_ratio=float(model.params["prev_ratio"]),
        regression_prev_coprime=float(model.params["prev_coprime"]),
        rmse=float(np.sqrt(np.mean(resid**2))),
        mae=float(np.mean(np.abs(resid))),
        selected_distribution=dist,
    )
    return model, summary, resid


def predict_branch_value(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    summary: BranchFitSummary,
    *,
    prev_poi: float,
    prev_ratio: float,
    prev_coprime: int,
) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    exog = pd.DataFrame(
        {
            "const": [1.0],
            "prev_poi": [float(prev_poi)],
            "prev_ratio": [float(prev_ratio)],
            "prev_coprime": [int(prev_coprime)],
        }
    )
    conditional_mean = float(model.predict(exog).iloc[0])
    center = conditional_mean + summary.selected_distribution.loc
    dist = summary.selected_distribution

    if dist.family == "student_t" and dist.df is not None:
        q80 = float(t_dist.ppf(0.90, df=dist.df, loc=center, scale=dist.scale))
        q20 = float(t_dist.ppf(0.10, df=dist.df, loc=center, scale=dist.scale))
        q95 = float(t_dist.ppf(0.975, df=dist.df, loc=center, scale=dist.scale))
        q05 = float(t_dist.ppf(0.025, df=dist.df, loc=center, scale=dist.scale))
    else:
        q80 = float(norm.ppf(0.90, loc=center, scale=dist.scale))
        q20 = float(norm.ppf(0.10, loc=center, scale=dist.scale))
        q95 = float(norm.ppf(0.975, loc=center, scale=dist.scale))
        q05 = float(norm.ppf(0.025, loc=center, scale=dist.scale))

    return center, conditional_mean, (q20, q80), (q05, q95)


def likely_rank(matches: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if matches.empty:
        return matches.copy()
    ranked = matches.drop(columns=["rank"], errors="ignore").sort_values(
        ["ticket_pair_sum_z", "ticket_pair_mean_z", "min_ticket_pair_z"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))
    return ranked.head(top_n).reset_index(drop=True)


def search_branch_candidates(
    pair_counts: np.ndarray,
    *,
    main_n: int,
    star_n: int,
    target_score: int,
    batch_size: int,
    pair_diag,
    top_n: int,
    max_save_matches: int,
) -> tuple[pd.DataFrame | None, pd.DataFrame, int, int]:
    star_slots = np.arange(main_n + 1, main_n + star_n + 1, dtype=np.int16)
    star_pair_idx = np.asarray(list(combinations(range(star_n), 2)), dtype=np.int16)
    star_encoded = star_slots[star_pair_idx]
    star_display = star_pair_idx + 1
    star_scores = pair_counts[star_encoded[:, 0], star_encoded[:, 1]].astype(int)
    tickets_per_main = len(star_pair_idx)

    exact_count = 0
    stored_frames: list[pd.DataFrame] = []
    keep_all = True
    shortlist = pd.DataFrame()
    processed_mains = 0

    main_iter = combinations(range(1, main_n + 1), 5)
    from euromillions.diagnostics3 import score_batch  # local import to avoid circular style noise

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
            batch_df = pd.DataFrame(
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
            batch_df = annotate_match_statistics(batch_df, pair_diag)
            exact_count += len(batch_df)

            batch_top = likely_rank(batch_df, top_n)
            shortlist = likely_rank(pd.concat([shortlist, batch_top], ignore_index=True), top_n)

            if keep_all:
                stored_frames.append(batch_df)
                stored_rows = int(sum(len(frame) for frame in stored_frames))
                if stored_rows > max_save_matches:
                    keep_all = False
                    stored_frames = []

        processed_mains += len(mains)

    if exact_count > 0:
        exact_df = pd.concat(stored_frames, ignore_index=True) if keep_all and stored_frames else None
        return exact_df, shortlist, exact_count, 0

    nearest_df, nearest_gap = find_nearest_full7_guesses(
        pair_counts,
        main_n=main_n,
        star_n=star_n,
        target_score=target_score,
        batch_size=batch_size,
    )
    nearest_df = annotate_match_statistics(nearest_df, pair_diag)
    shortlist = likely_rank(nearest_df, top_n)
    return None, shortlist, 0, int(nearest_gap)


def save_branch_plot(
    branch_frame: pd.DataFrame,
    *,
    upper_fit: BranchFitSummary,
    lower_fit: BranchFitSummary,
    upper_resid: np.ndarray,
    lower_resid: np.ndarray,
    predicted_score: int,
    next_branch: str,
    threshold: float,
    branch_mode: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    upper_mask = branch_frame["model_branch"] == "upper"
    lower_mask = branch_frame["model_branch"] == "lower"
    prime_mask = branch_frame["model_branch"] == "prime_ceiling"
    delta = branch_frame["draw_date"].diff().dropna().median()
    if pd.isna(delta):
        delta = pd.Timedelta(days=3)
    next_date = branch_frame["draw_date"].iloc[-1] + delta

    ax1 = axes[0, 0]
    ax1.plot(branch_frame["draw_date"], branch_frame["poi"], color="lightgray", lw=0.7, alpha=0.8)
    ax1.scatter(
        branch_frame.loc[lower_mask, "draw_date"],
        branch_frame.loc[lower_mask, "poi"],
        s=12,
        color="steelblue",
        label="lower branch",
    )
    ax1.scatter(
        branch_frame.loc[upper_mask, "draw_date"],
        branch_frame.loc[upper_mask, "poi"],
        s=12,
        color="tomato",
        label="upper branch",
    )
    if bool(prime_mask.any()):
        ax1.scatter(
            branch_frame.loc[prime_mask, "draw_date"],
            branch_frame.loc[prime_mask, "poi"],
            s=18,
            color="dimgray",
            marker="x",
            label="prime ceiling",
        )
    ax1.scatter(
        [next_date],
        [predicted_score],
        s=70,
        marker="*",
        color="purple",
        label=f"next branch={next_branch}, score={predicted_score}",
        zorder=4,
    )
    ax1.set_title(f"POI branch path with next-score forecast ({branch_mode})")
    ax1.set_ylabel("poi")
    ax1.legend(fontsize=8)

    ax2 = axes[0, 1]
    ax2.plot(branch_frame["draw_date"], branch_frame["totient_ratio"], color="navy", lw=0.9)
    ax2.axhline(threshold, color="gray", ls="--", lw=1, label=f"threshold {threshold:.3f}")
    ax2.fill_between(
        branch_frame["draw_date"],
        branch_frame["totient_ratio"],
        threshold,
        where=branch_frame["totient_ratio"] >= threshold,
        color="tomato",
        alpha=0.25,
    )
    ax2.fill_between(
        branch_frame["draw_date"],
        branch_frame["totient_ratio"],
        threshold,
        where=branch_frame["totient_ratio"] < threshold,
        color="steelblue",
        alpha=0.25,
    )
    ax2.set_title("Normalized Euler-totient ratio")
    ax2.set_ylabel("phi(x) / x")
    if bool(prime_mask.any()):
        ax2.scatter(
            branch_frame.loc[prime_mask, "draw_date"],
            branch_frame.loc[prime_mask, "totient_ratio"],
            s=14,
            color="dimgray",
            marker="x",
            alpha=0.85,
            label="prime ceiling",
        )
    ax2.legend(fontsize=8)

    ax3 = axes[1, 0]
    bins = 30
    x_grid = np.linspace(
        min(float(lower_resid.min()), float(upper_resid.min())),
        max(float(lower_resid.max()), float(upper_resid.max())),
        400,
    )
    ax3.hist(lower_resid, bins=bins, density=True, alpha=0.35, color="steelblue", label="lower residuals")
    ax3.hist(upper_resid, bins=bins, density=True, alpha=0.35, color="tomato", label="upper residuals")

    for fit, color in [(lower_fit, "steelblue"), (upper_fit, "tomato")]:
        dist = fit.selected_distribution
        if dist.family == "student_t" and dist.df is not None:
            density = t_dist.pdf(x_grid, df=dist.df, loc=dist.loc, scale=dist.scale)
        else:
            density = norm.pdf(x_grid, loc=dist.loc, scale=dist.scale)
        ax3.plot(x_grid, density, color=color, lw=2, label=f"{fit.branch} {dist.family}")

    ax3.set_title("Branch regression residual densities")
    ax3.set_xlabel("regression residual")
    ax3.legend(fontsize=8)

    ax4 = axes[1, 1]
    stats = (
        branch_frame.iloc[1:]
        .groupby("coprime_prev", as_index=True)["actual_branch_change"]
        .mean()
        .reindex([0, 1], fill_value=0.0)
    )
    ax4.bar(["gcd>1", "gcd=1"], stats.to_numpy(dtype=float), color=["gray", "purple"], alpha=0.7)
    ax4.scatter(["gcd>1", "gcd=1"], [0.0, 1.0], color="black", s=60, label="hard rule target")
    ax4.set_ylim(0.0, 1.05)
    ax4.set_title("Observed branch-flip rate vs GCD rule")
    ax4.set_ylabel("actual flip share")
    ax4.legend(fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_pruned_branch_plot(
    branch_frame: pd.DataFrame,
    *,
    threshold: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    prime_mask = branch_frame["prime_ceiling"].astype(bool)
    composite_upper_mask = branch_frame["pruned_branch"] == "composite_upper"
    lower_mask = branch_frame["pruned_branch"] == "lower"

    ax1 = axes[0, 0]
    ax1.plot(branch_frame["draw_date"], branch_frame["poi"], color="lightgray", lw=0.7, alpha=0.8)
    ax1.scatter(
        branch_frame.loc[lower_mask, "draw_date"],
        branch_frame.loc[lower_mask, "poi"],
        s=12,
        color="steelblue",
        label="lower composite",
    )
    ax1.scatter(
        branch_frame.loc[composite_upper_mask, "draw_date"],
        branch_frame.loc[composite_upper_mask, "poi"],
        s=14,
        color="darkorange",
        label="upper composite",
    )
    ax1.scatter(
        branch_frame.loc[prime_mask, "draw_date"],
        branch_frame.loc[prime_mask, "poi"],
        s=18,
        color="dimgray",
        marker="x",
        label="prime ceiling",
    )
    ax1.set_title("POI path with prime-ceiling pruning")
    ax1.set_ylabel("poi")
    ax1.legend(fontsize=8)

    ax2 = axes[0, 1]
    ax2.plot(branch_frame["draw_date"], branch_frame["totient_ratio"], color="navy", lw=0.8, alpha=0.6)
    ax2.axhline(threshold, color="gray", ls="--", lw=1, label=f"threshold {threshold:.3f}")
    ax2.scatter(
        branch_frame.loc[lower_mask, "draw_date"],
        branch_frame.loc[lower_mask, "totient_ratio"],
        s=10,
        color="steelblue",
        alpha=0.7,
        label="lower composite",
    )
    ax2.scatter(
        branch_frame.loc[composite_upper_mask, "draw_date"],
        branch_frame.loc[composite_upper_mask, "totient_ratio"],
        s=12,
        color="darkorange",
        alpha=0.8,
        label="upper composite",
    )
    ax2.scatter(
        branch_frame.loc[prime_mask, "draw_date"],
        branch_frame.loc[prime_mask, "totient_ratio"],
        s=16,
        color="dimgray",
        marker="x",
        alpha=0.9,
        label="prime ceiling",
    )
    ax2.set_title("Totient ratio with primes removed from the upper branch")
    ax2.set_ylabel("phi(x) / x")
    ax2.legend(fontsize=8)

    ax3 = axes[1, 0]
    composite_mask = ~prime_mask
    ax3.scatter(
        branch_frame.loc[composite_mask, "ratio_base"],
        branch_frame.loc[composite_mask, "totient_ratio"],
        s=12,
        color="steelblue",
        alpha=0.5,
        label="composite bases",
    )
    ax3.scatter(
        branch_frame.loc[prime_mask, "ratio_base"],
        branch_frame.loc[prime_mask, "totient_ratio"],
        s=18,
        color="dimgray",
        marker="x",
        alpha=0.85,
        label="prime bases",
    )
    x_max = int(branch_frame["ratio_base"].max())
    x_grid = np.arange(2, x_max + 1)
    ax3.plot(x_grid, 1.0 - 1.0 / x_grid, color="darkred", lw=1.5, label="prime ceiling 1 - 1/x")
    ax3.axhline(threshold, color="gray", ls="--", lw=1)
    ax3.set_title("Normalized prime ceiling vs composite scatter")
    ax3.set_xlabel("ratio base x")
    ax3.set_ylabel("phi(x) / x")
    ax3.legend(fontsize=8)

    ax4 = axes[1, 1]
    counts = pd.Series(
        {
            "lower": int(lower_mask.sum()),
            "upper composite": int(composite_upper_mask.sum()),
            "prime ceiling": int(prime_mask.sum()),
        }
    )
    shares = counts / max(int(len(branch_frame)), 1)
    ax4.bar(
        counts.index.tolist(),
        shares.to_numpy(dtype=float),
        color=["steelblue", "darkorange", "dimgray"],
        alpha=0.8,
    )
    for xpos, share, count in zip(range(len(counts)), shares.to_numpy(dtype=float), counts.to_numpy(dtype=int)):
        ax4.text(xpos, share + 0.01, f"{count}", ha="center", va="bottom", fontsize=9)
    ax4.set_ylim(0.0, min(1.0, float(shares.max()) + 0.12))
    ax4.set_title("Pruned class share")
    ax4.set_ylabel("share of rows")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.replace([np.inf, -np.inf], np.nan)
    return cleaned.where(pd.notna(cleaned), "")


def main() -> None:
    args = parse_args()
    history_path = resolve_repo_path(args.history)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_history = load_history(history_path)
    history, cutoff_start_date = apply_start_date_cutoff(
        raw_history,
        mode="full7",
        start_date_arg=args.start_date,
    )

    branch_frame, pair_counts, poi, main_n, star_n, eff_modulus = build_branch_frame(
        history,
        ratio_mode=args.ratio_mode,
        modulus=args.modulus,
        threshold=args.threshold,
    )
    branch_frame = apply_branch_mode(branch_frame, branch_mode=args.branch_mode)
    pair_diag = build_pair_z_diagnostics(history)
    training = build_training_frame(branch_frame)

    upper_model, upper_fit, upper_resid = fit_branch_model(training, "upper")
    lower_model, lower_fit, lower_resid = fit_branch_model(training, "lower")

    current_branch = resolve_current_branch(branch_frame, branch_mode=args.branch_mode)
    last_gcd = int(branch_frame["gcd_prev"].iloc[-1])
    last_coprime = bool(branch_frame["coprime_prev"].iloc[-1])
    next_branch = "lower" if current_branch == "upper" and last_coprime else current_branch
    if current_branch == "lower" and last_coprime:
        next_branch = "upper"

    prev_poi = float(branch_frame["poi"].iloc[-1])
    prev_ratio = float(branch_frame["totient_ratio"].iloc[-1])
    prev_coprime = int(branch_frame["coprime_prev"].iloc[-1])
    chosen_model = upper_model if next_branch == "upper" else lower_model
    chosen_fit = upper_fit if next_branch == "upper" else lower_fit
    predicted_center, conditional_mean, interval_80, interval_95 = predict_branch_value(
        chosen_model,
        chosen_fit,
        prev_poi=prev_poi,
        prev_ratio=prev_ratio,
        prev_coprime=prev_coprime,
    )
    predicted_score = max(1, int(round(predicted_center)))

    exact_df, shortlist_df, exact_count, nearest_gap = search_branch_candidates(
        pair_counts,
        main_n=main_n,
        star_n=star_n,
        target_score=predicted_score,
        batch_size=args.batch_size,
        pair_diag=pair_diag,
        top_n=args.top_n,
        max_save_matches=args.max_save_matches,
    )

    branch_frame["forecast_next_branch"] = ""
    branch_frame.loc[branch_frame.index[-1], "forecast_next_branch"] = next_branch
    branch_frame["forecast_next_score"] = np.nan
    branch_frame.loc[branch_frame.index[-1], "forecast_next_score"] = predicted_score
    branch_frame.to_csv(out_dir / "branch_series.csv", index=False)

    shortlist_df.to_csv(out_dir / "branch_superlikely_shortlist.csv", index=False)
    exact_saved = exact_df is not None
    if exact_saved:
        exact_df.to_csv(out_dir / "branch_exact_matches.csv", index=False)

    save_branch_plot(
        branch_frame,
        upper_fit=upper_fit,
        lower_fit=lower_fit,
        upper_resid=upper_resid,
        lower_resid=lower_resid,
        predicted_score=predicted_score,
        next_branch=next_branch,
        threshold=float(args.threshold),
        branch_mode=str(args.branch_mode),
        out_path=out_dir / "branch_selector.png",
    )
    save_pruned_branch_plot(
        branch_frame,
        threshold=float(args.threshold),
        out_path=out_dir / "branch_selector_pruned.png",
    )

    summary = ArithmeticBranchSummary(
        raw_rows=len(raw_history),
        rows=len(branch_frame),
        cutoff_start_date=cutoff_start_date,
        branch_mode=str(args.branch_mode),
        ratio_mode=args.ratio_mode,
        modulus=eff_modulus,
        branch_threshold=float(args.threshold),
        current_branch=current_branch,
        next_branch=next_branch,
        gcd_last=last_gcd,
        last_coprime=last_coprime,
        upper_count=int((branch_frame["model_branch"] == "upper").sum()),
        lower_count=int((branch_frame["model_branch"] == "lower").sum()),
        prime_ceiling_count=int((branch_frame["model_branch"] == "prime_ceiling").sum()),
        gcd_coprime_share=float(branch_frame["coprime_prev"].iloc[1:].mean()),
        gcd_rule_match_rate=float(
            np.mean(
                branch_frame["actual_branch_change"].iloc[1:].to_numpy(dtype=int)
                == branch_frame["gcd_expected_change"].iloc[1:].to_numpy(dtype=int)
            )
        ),
        predicted_next_poi=float(predicted_center),
        predicted_score=predicted_score,
        predicted_interval_80=[float(interval_80[0]), float(interval_80[1])],
        predicted_interval_95=[float(interval_95[0]), float(interval_95[1])],
        exact_match_count=int(exact_count),
        exact_matches_saved=bool(exact_saved),
        nearest_score_gap=int(nearest_gap),
        upper_fit=upper_fit,
        lower_fit=lower_fit,
    )
    (out_dir / "branch_summary.json").write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )

    summary_frame = pd.DataFrame(
        [
            {
                "raw_rows": summary.raw_rows,
                "rows": summary.rows,
                "cutoff_start_date": summary.cutoff_start_date,
                "branch_mode": summary.branch_mode,
                "ratio_mode": summary.ratio_mode,
                "modulus": summary.modulus,
                "branch_threshold": summary.branch_threshold,
                "current_branch": summary.current_branch,
                "next_branch": summary.next_branch,
                "gcd_last": summary.gcd_last,
                "last_coprime": int(summary.last_coprime),
                "upper_count": summary.upper_count,
                "lower_count": summary.lower_count,
                "prime_ceiling_count": summary.prime_ceiling_count,
                "gcd_coprime_share": summary.gcd_coprime_share,
                "gcd_rule_match_rate": summary.gcd_rule_match_rate,
                "predicted_next_poi": summary.predicted_next_poi,
                "predicted_score": summary.predicted_score,
                "predicted_interval_80_lo": summary.predicted_interval_80[0],
                "predicted_interval_80_hi": summary.predicted_interval_80[1],
                "predicted_interval_95_lo": summary.predicted_interval_95[0],
                "predicted_interval_95_hi": summary.predicted_interval_95[1],
                "exact_match_count": summary.exact_match_count,
                "exact_matches_saved": int(summary.exact_matches_saved),
                "nearest_score_gap": summary.nearest_score_gap,
            }
        ]
    )
    branch_fit_frame = pd.DataFrame([asdict(upper_fit), asdict(lower_fit)])
    write_excel_workbook(
        out_dir / "branch_selector.xlsx",
        [
            ("summary", excel_safe(summary_frame)),
            ("branch_fits", excel_safe(branch_fit_frame)),
            ("shortlist", excel_safe(shortlist_df)),
            ("series_tail", excel_safe(branch_frame.tail(200))),
        ],
    )

    print("=" * 72)
    print("ARITHMETIC BRANCH SELECTOR")
    print("=" * 72)
    print(
        f"Rows: {summary.rows} of {summary.raw_rows}  "
        f"Cutoff: {summary.cutoff_start_date}  "
        f"ratio={summary.ratio_mode}  modulus={summary.modulus}  branch_mode={summary.branch_mode}"
    )
    print(
        f"Branch split: upper={summary.upper_count}  lower={summary.lower_count}  "
        f"prime_ceiling={summary.prime_ceiling_count}  threshold={summary.branch_threshold:.3f}"
    )
    print(
        f"GCD rule: coprime share={summary.gcd_coprime_share:.3f}  "
        f"historical match rate={summary.gcd_rule_match_rate:.3f}  "
        f"current={summary.current_branch}  next={summary.next_branch}"
    )
    print(
        f"Next-score forecast: poi={summary.predicted_next_poi:.3f}  "
        f"score={summary.predicted_score}  "
        f"80%=[{summary.predicted_interval_80[0]:.2f}, {summary.predicted_interval_80[1]:.2f}]"
    )
    print(
        f"Upper fit: {upper_fit.selected_distribution.family}  "
        f"rmse={upper_fit.rmse:.3f}  mae={upper_fit.mae:.3f}"
    )
    print(
        f"Lower fit: {lower_fit.selected_distribution.family}  "
        f"rmse={lower_fit.rmse:.3f}  mae={lower_fit.mae:.3f}"
    )
    print(
        f"Candidate bar: exact_matches={summary.exact_match_count}  "
        f"saved_all={summary.exact_matches_saved}  nearest_gap={summary.nearest_score_gap}"
    )
    print(f"Wrote: {out_dir / 'branch_selector.png'}")
    print(f"Wrote: {out_dir / 'branch_selector_pruned.png'}")
    print(f"Wrote: {out_dir / 'branch_superlikely_shortlist.csv'}")
    print(f"Wrote: {out_dir / 'branch_summary.json'}")


if __name__ == "__main__":
    main()
