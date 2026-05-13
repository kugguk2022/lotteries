from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from euromillions.arithmetic_branch import build_branch_frame
from euromillions.diagnostics3 import (
    annotate_match_statistics,
    build_pair_z_diagnostics,
)
from euromillions_agent.phase2_sobol import generate_sobol_tickets

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = REPO_ROOT / "data" / "euromillions.csv"
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs" / "euromillions"
DEFAULT_PRIZES_RANGE = REPO_ROOT / "data" / "prizes_range.json"
DEFAULT_OUT_DIR = DEFAULT_OUTPUTS_ROOT / "garch_return_benchmark"
DEFAULT_TOP_N = 25
DEFAULT_HOLDOUT = 12
DEFAULT_POOL_SIZE = 10000
DEFAULT_TICKET_COST = 2.5


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fitted_csv: str


TOP_GARCH_MODELS = (
    ModelSpec("garchx", "garchx/garchx_fitted.csv"),
    ModelSpec("garchx_alternative_volatility", "garchx_alternative_volatility/garchx_fitted.csv"),
    ModelSpec("garchx_alternative_volatility_v2", "garchx_alternative_volatility_v2/garchx_fitted.csv"),
)


@dataclass
class MethodSummary:
    steps: int
    total_ticket_cost: float
    total_return: float
    net_profit: float
    roi: float
    mean_return_per_draw: float
    mean_profit_per_draw: float
    mean_best_ball_hits: float
    mean_best_star_hits: float
    any_payout_draw_rate: float
    break_even_draw_rate: float
    exact_main5_accuracy: float
    exact_5plus2_accuracy: float
    mean_target_gap: float
    mean_abs_poi_error: float
    mean_shortlist_size: float


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark top GARCH-family EuroMillions approaches on realized shortlist returns."
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--prizes-range", type=Path, default=DEFAULT_PRIZES_RANGE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--holdout", type=int, default=DEFAULT_HOLDOUT)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--ticket-cost", type=float, default=DEFAULT_TICKET_COST)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument(
        "--pool-path",
        type=Path,
        default=None,
        help="Optional CSV of candidate tickets. If missing, a Sobol pool is generated.",
    )
    return parser.parse_args()


def load_prize_steps(path: Path) -> list[dict[tuple[int, int], float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    steps = raw.get("steps", [])
    out: list[dict[tuple[int, int], float]] = []
    for entry in steps:
        prizes = entry.get("prizes", {})
        table: dict[tuple[int, int], float] = {}
        for key, value in prizes.items():
            main_hits, star_hits = map(int, str(key).split("_"))
            table[(main_hits, star_hits)] = float(value)
        out.append(table)
    return out


def load_prediction_frame(outputs_root: Path, spec: ModelSpec) -> pd.DataFrame:
    frame = pd.read_csv(outputs_root / spec.fitted_csv, parse_dates=["date"])
    return frame[["date", "poi", "mu_hat"]].copy().sort_values("date").reset_index(drop=True)


def load_all_prediction_frames(outputs_root: Path) -> dict[str, pd.DataFrame]:
    return {spec.name: load_prediction_frame(outputs_root, spec) for spec in TOP_GARCH_MODELS}


def generate_or_load_candidate_pool(
    *,
    pool_path: Path,
    pool_size: int,
) -> pd.DataFrame:
    if pool_path.exists():
        df = pd.read_csv(pool_path, header=None)
    else:
        tickets = generate_sobol_tickets(
            n_tickets=int(pool_size),
            main_n=50,
            main_k=5,
            star_n=12,
            star_k=2,
            oversample=2,
            seed=42,
            max_shared_main=5,
            max_shared_star=2,
        )
        rows: list[list[int]] = []
        for main, stars in tickets:
            if stars is None:
                raise ValueError("Expected full 5+2 tickets from the Sobol generator.")
            rows.append([*main, *stars])
        df = pd.DataFrame(rows)
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(pool_path, index=False, header=False)

    if df.shape[1] != 7:
        raise ValueError(f"Candidate pool must have 7 integer columns, got shape={df.shape}.")

    df = df.astype(int)
    df.columns = [f"ball_{idx}" for idx in range(1, 6)] + ["star_1", "star_2"]
    df.insert(0, "combination_index", np.arange(1, len(df) + 1, dtype=int))
    return df


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


def score_ticket_pool(
    candidate_pool: pd.DataFrame,
    *,
    pair_counts: np.ndarray,
    main_n: int,
) -> np.ndarray:
    mains = candidate_pool[[f"ball_{idx}" for idx in range(1, 6)]].to_numpy(dtype=int)
    stars = candidate_pool[["star_1", "star_2"]].to_numpy(dtype=int)
    main_scores = score_batch(pair_counts, mains)
    star1_enc = main_n + stars[:, 0]
    star2_enc = main_n + stars[:, 1]
    star_scores = pair_counts[star1_enc, star2_enc].astype(int)
    main_star_sum_1 = pair_counts[mains, star1_enc[:, None]].sum(axis=1).astype(int)
    main_star_sum_2 = pair_counts[mains, star2_enc[:, None]].sum(axis=1).astype(int)
    return (main_scores + star_scores + main_star_sum_1 + main_star_sum_2).astype(int)


def build_garch_shortlist(
    candidate_pool: pd.DataFrame,
    *,
    pair_counts: np.ndarray,
    main_n: int,
    pair_diag,
    target_score: int,
    top_n: int,
) -> tuple[pd.DataFrame, int]:
    scored = candidate_pool.copy()
    scored["score"] = score_ticket_pool(scored, pair_counts=pair_counts, main_n=main_n)
    scored["score_gap"] = (scored["score"] - int(target_score)).abs().astype(int)
    best_gap = int(scored["score_gap"].min())
    scored = annotate_match_statistics(scored, pair_diag)
    shortlist = (
        scored.sort_values(
            ["score_gap", "ticket_pair_sum_zscore", "ticket_pair_mean_z", "min_ticket_pair_z"],
            ascending=[True, False, False, False],
        )
        .head(int(top_n))
        .reset_index(drop=True)
    )
    shortlist.insert(0, "rank", np.arange(1, len(shortlist) + 1, dtype=int))
    return shortlist, best_gap


def ticket_hit_tuple(ticket_row: pd.Series, actual_draw: pd.Series) -> tuple[int, int]:
    actual_balls = {int(actual_draw[f"ball_{idx}"]) for idx in range(1, 6)}
    actual_stars = {int(actual_draw[f"star_{idx}"]) for idx in range(1, 3)}
    balls = {int(ticket_row[f"ball_{idx}"]) for idx in range(1, 6)}
    stars = {int(ticket_row["star_1"]), int(ticket_row["star_2"])}
    return len(actual_balls.intersection(balls)), len(actual_stars.intersection(stars))


def score_shortlist_returns(
    shortlist: pd.DataFrame,
    *,
    actual_draw: pd.Series,
    prize_table: dict[tuple[int, int], float],
    ticket_cost: float,
) -> dict[str, float | int]:
    if shortlist.empty:
        return {
            "shortlist_size": 0,
            "best_ball_hits": 0,
            "best_star_hits": 0,
            "exact_main5": 0,
            "exact_5plus2": 0,
            "ticket_cost": 0.0,
            "draw_return": 0.0,
            "draw_profit": 0.0,
            "draw_roi": float("nan"),
            "winning_ticket_count": 0,
            "best_ticket_payout": 0.0,
            "any_payout_draw": 0,
            "break_even_draw": 0,
        }

    best_ball_hits = -1
    best_star_hits = -1
    best_ticket_payout = 0.0
    draw_return = 0.0
    winning_ticket_count = 0

    for row in shortlist.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        ball_hits, star_hits = ticket_hit_tuple(row_series, actual_draw)
        payout = float(prize_table.get((ball_hits, star_hits), 0.0))
        draw_return += payout
        if payout > 0.0:
            winning_ticket_count += 1
        best_ticket_payout = max(best_ticket_payout, payout)
        if (ball_hits, star_hits, payout) > (best_ball_hits, best_star_hits, best_ticket_payout):
            best_ball_hits = int(ball_hits)
            best_star_hits = int(star_hits)

    cost = float(len(shortlist) * ticket_cost)
    profit = float(draw_return - cost)
    roi = float(profit / cost) if cost > 0 else float("nan")
    return {
        "shortlist_size": int(len(shortlist)),
        "best_ball_hits": int(best_ball_hits),
        "best_star_hits": int(best_star_hits),
        "exact_main5": int(best_ball_hits == 5),
        "exact_5plus2": int(best_ball_hits == 5 and best_star_hits == 2),
        "ticket_cost": cost,
        "draw_return": float(draw_return),
        "draw_profit": profit,
        "draw_roi": roi,
        "winning_ticket_count": int(winning_ticket_count),
        "best_ticket_payout": float(best_ticket_payout),
        "any_payout_draw": int(draw_return > 0.0),
        "break_even_draw": int(profit >= 0.0),
    }


def summarize_method(step_frame: pd.DataFrame) -> MethodSummary:
    total_cost = float(step_frame["ticket_cost"].sum())
    total_return = float(step_frame["draw_return"].sum())
    net_profit = float(total_return - total_cost)
    roi = float(net_profit / total_cost) if total_cost > 0 else float("nan")
    return MethodSummary(
        steps=int(len(step_frame)),
        total_ticket_cost=total_cost,
        total_return=total_return,
        net_profit=net_profit,
        roi=roi,
        mean_return_per_draw=float(step_frame["draw_return"].mean()),
        mean_profit_per_draw=float(step_frame["draw_profit"].mean()),
        mean_best_ball_hits=float(step_frame["best_ball_hits"].mean()),
        mean_best_star_hits=float(step_frame["best_star_hits"].mean()),
        any_payout_draw_rate=float(step_frame["any_payout_draw"].mean()),
        break_even_draw_rate=float(step_frame["break_even_draw"].mean()),
        exact_main5_accuracy=float(step_frame["exact_main5"].mean()),
        exact_5plus2_accuracy=float(step_frame["exact_5plus2"].mean()),
        mean_target_gap=float(step_frame["target_gap"].mean()),
        mean_abs_poi_error=float(step_frame["abs_poi_error"].mean()),
        mean_shortlist_size=float(step_frame["shortlist_size"].mean()),
    )


def build_report(manifest: dict) -> str:
    lines = [
        "# GARCH Return Benchmark",
        "",
        f"- Benchmark mode: {manifest['benchmark_mode']}",
        f"- Holdout draws: {manifest['holdout_steps']}",
        f"- Ticket budget per draw: common top-{manifest['requested_top_n']} clipped to each step's shared shortlist size",
        f"- Candidate pool size: {manifest['candidate_pool_size']}",
        f"- Ticket cost: {manifest['ticket_cost']}",
        f"- Prize alignment: {manifest['prize_alignment_note']}",
        "",
        "## Ranking by ROI",
    ]
    for index, item in enumerate(manifest["ranking_by_roi"], start=1):
        summary = manifest["methods"][item]
        lines.append(
            f"- {index}. `{item}` | roi={summary['roi']:.4f} | net_profit={summary['net_profit']:.2f} | "
            f"total_return={summary['total_return']:.2f} | total_cost={summary['total_ticket_cost']:.2f} | "
            f"mean_best_ball_hits={summary['mean_best_ball_hits']:.4f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    history_path = resolve_repo_path(args.history)
    outputs_root = resolve_repo_path(args.outputs_root)
    prizes_range_path = resolve_repo_path(args.prizes_range)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(history_path, parse_dates=["draw_date"])
    history = history.sort_values("draw_date").drop_duplicates(subset=["draw_date"]).reset_index(drop=True)
    prediction_frames = load_all_prediction_frames(outputs_root)
    prize_steps = load_prize_steps(prizes_range_path)

    effective_holdout = min(
        int(args.holdout),
        len(prize_steps),
        min(len(frame) for frame in prediction_frames.values()),
        max(len(history) - 25, 1),
    )
    prize_tail = prize_steps[-effective_holdout:]
    history_tail = history.tail(effective_holdout).reset_index(drop=True)
    history_tail_dates = history_tail["draw_date"].dt.strftime("%Y-%m-%d").tolist()

    pool_path = (
        resolve_repo_path(args.pool_path)
        if args.pool_path is not None
        else out_dir / f"sobol_ticket_pool_{int(args.pool_size)}.csv"
    )
    candidate_pool = generate_or_load_candidate_pool(pool_path=pool_path, pool_size=int(args.pool_size))

    prediction_lookup: dict[str, pd.DataFrame] = {}
    for name, frame in prediction_frames.items():
        tail_frame = frame[frame["date"].dt.strftime("%Y-%m-%d").isin(history_tail_dates)].copy()
        tail_frame["draw_date"] = tail_frame["date"].dt.strftime("%Y-%m-%d")
        prediction_lookup[name] = tail_frame.set_index("draw_date").sort_index()

    rows: list[dict[str, object]] = []
    start_idx = len(history) - effective_holdout
    for offset, draw_idx in enumerate(range(start_idx, len(history))):
        train_history = history.iloc[:draw_idx].reset_index(drop=True)
        actual_draw = history.iloc[draw_idx]
        actual_date = pd.Timestamp(actual_draw["draw_date"]).strftime("%Y-%m-%d")
        pair_diag = build_pair_z_diagnostics(train_history)
        _, pair_counts, _, main_n, star_n, _ = build_branch_frame(
            train_history,
            ratio_mode="raw",
            modulus=None,
            threshold=0.5,
        )

        per_method: dict[str, tuple[pd.DataFrame, int, float, int]] = {}
        for spec in TOP_GARCH_MODELS:
            pred_row = prediction_lookup[spec.name].loc[actual_date]
            predicted_poi = float(pred_row["mu_hat"])
            target_score = int(round(predicted_poi))
            shortlist, target_gap = build_garch_shortlist(
                candidate_pool,
                pair_counts=pair_counts,
                main_n=main_n,
                pair_diag=pair_diag,
                target_score=target_score,
                top_n=int(args.top_n),
            )
            per_method[spec.name] = (shortlist, target_gap, predicted_poi, target_score)

        common_n = min(len(value[0]) for value in per_method.values())
        prize_table = prize_tail[offset]

        for spec in TOP_GARCH_MODELS:
            shortlist, target_gap, predicted_poi, target_score = per_method[spec.name]
            shortlist = shortlist.head(common_n).reset_index(drop=True)
            scored = score_shortlist_returns(
                shortlist,
                actual_draw=actual_draw,
                prize_table=prize_table,
                ticket_cost=float(args.ticket_cost),
            )
            rows.append(
                {
                    "draw_date": actual_date,
                    "model": spec.name,
                    "predicted_poi": predicted_poi,
                    "actual_poi": float(prediction_lookup[spec.name].loc[actual_date]["poi"]),
                    "abs_poi_error": float(abs(predicted_poi - float(prediction_lookup[spec.name].loc[actual_date]["poi"]))),
                    "target_score": int(target_score),
                    "target_gap": int(target_gap),
                    "common_top_n": int(common_n),
                    "prize_step_offset": int(offset),
                    **scored,
                }
            )

    step_frame = pd.DataFrame(rows).sort_values(["draw_date", "model"]).reset_index(drop=True)
    methods = {
        model: asdict(summarize_method(step_frame[step_frame["model"] == model].reset_index(drop=True)))
        for model in [spec.name for spec in TOP_GARCH_MODELS]
    }
    ranking_by_roi = sorted(
        methods,
        key=lambda name: (
            float(methods[name]["roi"]),
            float(methods[name]["net_profit"]),
            float(methods[name]["mean_best_ball_hits"]),
        ),
        reverse=True,
    )

    manifest = {
        "benchmark_mode": "saved_tail_predictions_with_forward_ticket_scoring",
        "holdout_steps": int(effective_holdout),
        "requested_top_n": int(args.top_n),
        "candidate_pool_size": int(len(candidate_pool)),
        "ticket_cost": float(args.ticket_cost),
        "prize_alignment_note": "Per-draw prize tables were aligned to the last available history draws by tail count because prizes_range.json has draw ids but no draw dates.",
        "models": [spec.name for spec in TOP_GARCH_MODELS],
        "ranking_by_roi": ranking_by_roi,
        "methods": methods,
    }

    step_frame.to_csv(out_dir / "garch_return_benchmark_steps.csv", index=False)
    (out_dir / "garch_return_benchmark.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (out_dir / "garch_return_benchmark.md").write_text(
        build_report(manifest),
        encoding="utf-8",
    )

    print("GARCH RETURN BENCHMARK")
    print(
        f"mode={manifest['benchmark_mode']} holdout={effective_holdout} "
        f"top_n={args.top_n} pool={len(candidate_pool)}"
    )
    for name in ranking_by_roi:
        summary = methods[name]
        print(
            f"{name}: roi={summary['roi']:.4f} "
            f"net_profit={summary['net_profit']:.2f} "
            f"return={summary['total_return']:.2f} "
            f"cost={summary['total_ticket_cost']:.2f} "
            f"mean_best_ball_hits={summary['mean_best_ball_hits']:.4f}"
        )
    print(out_dir / "garch_return_benchmark_steps.csv")
    print(out_dir / "garch_return_benchmark.json")
    print(out_dir / "garch_return_benchmark.md")


if __name__ == "__main__":
    main()
