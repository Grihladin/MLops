"""
Assess reliability indicators for the load sensor signal in each forklift CSV.

For every ``*_forklift.csv`` file, this script computes:
  * Share of time the machine reports carrying a load (duty cycle)
  * Number of load/unload toggles per hour
  * How many load segments are shorter than a configurable threshold
  * Median duration spent carrying vs. not carrying

Use it alongside the plots to quickly spot machines whose LiDAR sensor flickers
between load states.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).parent / "real_data_with_headers"
DEFAULT_OUTPUT = Path(__file__).parent / "plots" / "load_sensor_summary.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize load-sensor reliability metrics per forklift CSV."
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
        help=(
            "Seconds below which a load segment is considered suspiciously short "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path to store the summary table "
        f"(default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def load_forklift_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*_forklift.csv"))
    if not files:
        raise FileNotFoundError(
            f"No *_forklift.csv files found in {data_dir}. "
            "Run assign_columns.py first or point to a directory that contains forklift data."
        )
    return files


def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing_cols = {"Timestamp", "Load"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path} is missing required columns: {missing_cols}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", errors="coerce")
    df["Load"] = pd.to_numeric(df["Load"], errors="coerce").fillna(0)
    df["LoadState"] = df["Load"].gt(0).astype(int)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    df["NextTimestamp"] = df["Timestamp"].shift(-1)
    df["Duration"] = (df["NextTimestamp"] - df["Timestamp"]).dt.total_seconds()
    median_duration = df["Duration"].median()
    if pd.isna(median_duration) or median_duration <= 0:
        median_duration = 1.0
    df["Duration"] = df["Duration"].fillna(median_duration)
    df = df.drop(columns=["NextTimestamp"])
    return df


def compute_segments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["LoadState", "Duration"],
            dtype=float,
        )

    change_groups = df["LoadState"].ne(df["LoadState"].shift()).cumsum()
    segments = (
        df.groupby(change_groups)
        .agg(
            LoadState=("LoadState", "first"),
            Duration=("Duration", "sum"),
        )
        .reset_index(drop=True)
    )
    return segments


def summarize_machine(
    csv_path: Path,
    short_threshold: float,
) -> dict[str, float | str]:
    df = prepare_dataframe(csv_path)
    segments = compute_segments(df)

    total_time = df["Duration"].sum()
    loaded_time = (df["Duration"] * df["LoadState"]).sum()
    duty_cycle = loaded_time / total_time if total_time else 0.0

    state_changes = df["LoadState"].ne(df["LoadState"].shift()).sum() - 1
    state_changes = max(state_changes, 0)
    total_hours = total_time / 3600 if total_time else 0.0
    toggles_per_hour = state_changes / total_hours if total_hours else 0.0

    load_segments = segments[segments["LoadState"] == 1]
    unload_segments = segments[segments["LoadState"] == 0]

    short_load_segments = load_segments[
        load_segments["Duration"] < short_threshold
    ].shape[0]
    load_segment_count = load_segments.shape[0]
    short_ratio = (
        short_load_segments / load_segment_count if load_segment_count else 0.0
    )

    return {
        "machine": csv_path.stem,
        "total_hours": total_hours,
        "duty_cycle": duty_cycle,
        "toggles_per_hour": toggles_per_hour,
        "load_segments": load_segment_count,
        "short_load_segments": short_load_segments,
        "short_load_ratio": short_ratio,
        "median_load_duration_s": load_segments["Duration"].median()
        if not load_segments.empty
        else 0.0,
        "median_unload_duration_s": unload_segments["Duration"].median()
        if not unload_segments.empty
        else 0.0,
    }


def summarize_directory(
    data_dir: Path,
    short_threshold: float,
) -> pd.DataFrame:
    csv_files = load_forklift_files(data_dir)
    records = [
        summarize_machine(csv_file, short_threshold) for csv_file in csv_files
    ]
    return pd.DataFrame(records)


def format_summary_table(summary_df: pd.DataFrame) -> str:
    table_df = summary_df.copy()
    table_df["duty_cycle"] = table_df["duty_cycle"].map(lambda v: f"{v:.1%}")
    table_df["short_load_ratio"] = table_df["short_load_ratio"].map(
        lambda v: f"{v:.1%}"
    )
    float_cols = [
        "total_hours",
        "toggles_per_hour",
        "median_load_duration_s",
        "median_unload_duration_s",
    ]
    for col in float_cols:
        table_df[col] = table_df[col].map(lambda v: f"{v:.2f}")
    return table_df.to_string(index=False)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary_df = summarize_directory(args.data_dir, args.short_threshold).sort_values(
        ["short_load_ratio", "toggles_per_hour"], ascending=[False, False]
    )

    print("Load sensor reliability summary:")
    print(
        "Higher short_load_ratio or toggle rates may indicate a noisy LiDAR sensor."
    )
    print(format_summary_table(summary_df))

    output_path = args.output or DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"\nWrote raw metrics to {output_path}")


if __name__ == "__main__":
    main()
