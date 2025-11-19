"""
Detect potential LiDAR load-sensor anomalies for each forklift CSV.

Heuristics used per machine:
  * Fraction of load segments shorter than ``short_threshold`` seconds
  * Load/unload toggles per hour
  * Number of load-state flips that happen while the truck is moving faster than
    ``speed_threshold`` km/h (per hour and absolute count)

If any metric exceeds the configurable limits, the machine is tagged with the
corresponding reason(s).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from evaluate_load_sensor import (
    load_forklift_files,
    prepare_dataframe,
    summarize_directory,
)

DEFAULT_DATA_DIR = Path(__file__).parent / "real_data_with_headers"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flag forklifts whose load sensors behave anomalously."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing *_forklift.csv files (default: %(default)s)",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        default=10.0,
        help="Seconds that define a 'short' load segment (default: %(default)s)",
    )
    parser.add_argument(
        "--short-ratio-limit",
        type=float,
        default=0.35,
        help="Ratio of short load segments that triggers an anomaly (default: %(default)s)",
    )
    parser.add_argument(
        "--toggle-limit",
        type=float,
        default=12.0,
        help="Load/unload toggles per hour that trigger an anomaly (default: %(default)s)",
    )
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=0.2,
        help="Speed in km/h to treat a load flip as 'in motion' (default: %(default)s)",
    )
    parser.add_argument(
        "--fast-flip-limit",
        type=float,
        default=2.0,
        help="Fast load flips per hour that trigger an anomaly (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the anomaly table as CSV.",
    )
    return parser.parse_args(argv)


def detect_fast_flips(
    df: pd.DataFrame,
    speed_threshold: float,
) -> tuple[int, float]:
    speed = pd.to_numeric(df.get("Speed"), errors="coerce").fillna(0)
    state_change = df["LoadState"].ne(df["LoadState"].shift())
    fast_mask = state_change & (
        (speed > speed_threshold) | (speed.shift(fill_value=0) > speed_threshold)
    )
    count = int(fast_mask.sum())

    total_time = df["Duration"].sum()
    total_hours = total_time / 3600 if total_time else 0.0
    per_hour = count / total_hours if total_hours else 0.0
    return count, per_hour


def compute_metrics(
    data_dir: Path,
    short_threshold: float,
    speed_threshold: float,
) -> pd.DataFrame:
    base_summary = summarize_directory(data_dir, short_threshold)
    csv_files = load_forklift_files(data_dir)

    fast_flip_rows = []
    for csv_file in csv_files:
        df = prepare_dataframe(csv_file)
        count, per_hour = detect_fast_flips(df, speed_threshold)
        fast_flip_rows.append(
            {
                "machine": csv_file.stem,
                "fast_flip_count": count,
                "fast_flips_per_hour": per_hour,
            }
        )

    fast_flip_df = pd.DataFrame(fast_flip_rows)
    return base_summary.merge(fast_flip_df, on="machine")


def classify_machine(row: pd.Series, limits: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if row["short_load_ratio"] >= limits["short_ratio_limit"]:
        reasons.append(
            f"short load ratio {row['short_load_ratio']:.0%} >= "
            f"{limits['short_ratio_limit']:.0%}"
        )
    if row["toggles_per_hour"] >= limits["toggle_limit"]:
        reasons.append(
            f"toggles/hour {row['toggles_per_hour']:.2f} >= {limits['toggle_limit']:.2f}"
        )
    if row["fast_flips_per_hour"] >= limits["fast_flip_limit"]:
        reasons.append(
            f"fast flips/hour {row['fast_flips_per_hour']:.2f} >= "
            f"{limits['fast_flip_limit']:.2f}"
        )
    return reasons


def format_anomaly_table(summary_df: pd.DataFrame) -> str:
    display_df = summary_df.copy()
    display_df["short_load_ratio"] = display_df["short_load_ratio"].map(
        lambda v: f"{v:.1%}"
    )
    percent_cols = ["duty_cycle"]
    for col in percent_cols:
        display_df[col] = display_df[col].map(lambda v: f"{v:.1%}")

    float_cols = [
        "toggles_per_hour",
        "fast_flips_per_hour",
        "fast_flip_count",
    ]
    for col in float_cols:
        display_df[col] = display_df[col].map(lambda v: f"{v:.2f}")

    display_df["anomaly_reasons"] = display_df["anomaly_reasons"].map(
        lambda reasons: "; ".join(reasons) if reasons else ""
    )
    return display_df[
        [
            "machine",
            "duty_cycle",
            "short_load_ratio",
            "toggles_per_hour",
            "fast_flips_per_hour",
            "fast_flip_count",
            "anomaly_reasons",
        ]
    ].to_string(index=False)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary_df = compute_metrics(
        args.data_dir,
        args.short_threshold,
        args.speed_threshold,
    )

    limits = {
        "short_ratio_limit": args.short_ratio_limit,
        "toggle_limit": args.toggle_limit,
        "fast_flip_limit": args.fast_flip_limit,
    }
    summary_df["anomaly_reasons"] = summary_df.apply(
        classify_machine,
        axis=1,
        limits=limits,
    )
    anomalies = summary_df[summary_df["anomaly_reasons"].map(bool)].copy()

    if anomalies.empty:
        print("No machines crossed the anomaly thresholds.")
    else:
        print("Potential load-sensor anomalies:")
        print(format_anomaly_table(anomalies))

    if args.output:
        summary_df.to_csv(args.output, index=False)
        print(f"\nFull metric table written to {args.output}")


if __name__ == "__main__":
    main()
