"""
Monitor forklift safety/wear indicators using height and speed alone.

The script flags three behaviors per cleaned forklift CSV:
  * Overspeed with raised forks: height above `--raised-threshold` while
    speed exceeds `--overspeed-limit`.
  * Tall lift while moving: height above `--tall-threshold` while speed
    exceeds `--tall-move-speed`.
  * Height oscillations: frequent transitions across the raised threshold,
    which often indicate handling issues or dropped pallets.

Results are summarized per machine and written to stdout (optional CSV).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_DATA_DIR = (
    Path(__file__).parent / "real_data_with_headers_height_cleaned"
)
DEFAULT_OUTPUT = Path(__file__).parent / "plots" / "safety_monitor.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flag forklift safety/wear behaviors using height + speed."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing cleaned *_forklift.csv files (default: %(default)s)",
    )
    parser.add_argument(
        "--raised-threshold",
        type=float,
        default=0.5,
        help="Height (m) that counts as forks raised (default: %(default)s)",
    )
    parser.add_argument(
        "--tall-threshold",
        type=float,
        default=3.5,
        help="Height (m) considered a tall lift (default: %(default)s)",
    )
    parser.add_argument(
        "--overspeed-limit",
        type=float,
        default=8.0,
        help="Speed (km/h) limit while forks are raised (default: %(default)s)",
    )
    parser.add_argument(
        "--tall-move-speed",
        type=float,
        default=1.0,
        help="Speed (km/h) that counts as moving during tall lifts (default: %(default)s)",
    )
    parser.add_argument(
        "--toggle-limit",
        type=float,
        default=30.0,
        help="Raised/lowered transitions per hour that trigger an oscillation alert (default: %(default)s)",
    )
    parser.add_argument(
        "--overspeed-limit-rate",
        type=float,
        default=4.0,
        help="Overspeed events per hour that trigger an alert (default: %(default)s)",
    )
    parser.add_argument(
        "--tall-move-limit-rate",
        type=float,
        default=2.0,
        help="Tall-lift-while-moving events per hour that trigger an alert (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path to store the summary table (default: %(default)s)",
    )
    return parser.parse_args(argv)


def load_forklift_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*_forklift.csv"))
    if not files:
        raise FileNotFoundError(
            f"No *_forklift.csv files found in {data_dir}. "
            "Run clean_forklift_height.py first or point to the cleaned directory."
        )
    return files


def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {"Timestamp", "Height", "Speed"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], unit="ms", errors="coerce"
    )
    df["Height"] = pd.to_numeric(df["Height"], errors="coerce")
    df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Height", "Speed"]).sort_values(
        "Timestamp"
    )
    if df.empty:
        raise ValueError(f"{csv_path} has no valid rows after preprocessing.")

    df["NextTimestamp"] = df["Timestamp"].shift(-1)
    df["Duration"] = (df["NextTimestamp"] - df["Timestamp"]).dt.total_seconds()
    median_duration = df["Duration"].median()
    if pd.isna(median_duration) or median_duration <= 0:
        median_duration = 1.0
    df["Duration"] = df["Duration"].fillna(median_duration)
    df = df.drop(columns=["NextTimestamp"])
    return df.reset_index(drop=True)


def compute_event_stats(mask: pd.Series, durations: pd.Series) -> tuple[float, int]:
    seconds = float((mask.astype(float) * durations).sum())
    entries = int((mask & ~mask.shift(fill_value=False)).sum())
    return seconds, entries


def summarize_machine(
    csv_path: Path,
    raised_threshold: float,
    tall_threshold: float,
    overspeed_limit: float,
    tall_move_speed: float,
    toggle_limit: float,
    overspeed_limit_rate: float,
    tall_move_limit_rate: float,
) -> dict[str, float | str | list[str]]:
    df = prepare_dataframe(csv_path)
    durations = df["Duration"]
    total_seconds = durations.sum()
    total_hours = total_seconds / 3600 if total_seconds else 0.0

    raised = df["Height"].gt(raised_threshold)
    tall = df["Height"].gt(tall_threshold)
    fast = df["Speed"].gt(overspeed_limit)
    moving = df["Speed"].gt(tall_move_speed)

    overspeed_mask = raised & fast
    overspeed_seconds, overspeed_entries = compute_event_stats(
        overspeed_mask, durations
    )

    tall_move_mask = tall & moving
    tall_move_seconds, tall_move_entries = compute_event_stats(
        tall_move_mask, durations
    )

    toggle_count = int(raised.ne(raised.shift()).sum() - 1)
    toggle_count = max(toggle_count, 0)
    toggle_rate = toggle_count / total_hours if total_hours else 0.0

    overspeed_rate = (
        overspeed_entries / total_hours if total_hours else 0.0
    )
    tall_move_rate = (
        tall_move_entries / total_hours if total_hours else 0.0
    )

    alerts: list[str] = []
    if overspeed_rate >= overspeed_limit_rate:
        alerts.append(
            f"Overspeed w/raised forks {overspeed_rate:.1f}/hr ≥ {overspeed_limit_rate:.1f}/hr"
        )
    if tall_move_rate >= tall_move_limit_rate:
        alerts.append(
            f"Tall lift while moving {tall_move_rate:.1f}/hr ≥ {tall_move_limit_rate:.1f}/hr"
        )
    if toggle_rate >= toggle_limit:
        alerts.append(
            f"Height toggles {toggle_rate:.1f}/hr ≥ {toggle_limit:.1f}/hr"
        )

    return {
        "machine": csv_path.stem,
        "hours_observed": total_hours,
        "alert_count": len(alerts),
        "overspeed_events": overspeed_entries,
        "overspeed_time_pct": overspeed_seconds / total_seconds
        if total_seconds
        else 0.0,
        "overspeed_rate_per_hr": overspeed_rate,
        "tall_moving_events": tall_move_entries,
        "tall_moving_time_pct": tall_move_seconds / total_seconds
        if total_seconds
        else 0.0,
        "tall_moving_rate_per_hr": tall_move_rate,
        "height_toggle_rate_per_hr": toggle_rate,
        "alerts": alerts,
    }


def format_table(df: pd.DataFrame) -> str:
    table = df.copy()
    percent_cols = ["overspeed_time_pct", "tall_moving_time_pct"]
    for col in percent_cols:
        table[col] = table[col].map(lambda v: f"{v:.1%}")
    float_cols = [
        "hours_observed",
        "overspeed_rate_per_hr",
        "tall_moving_rate_per_hr",
        "height_toggle_rate_per_hr",
    ]
    for col in float_cols:
        table[col] = table[col].map(lambda v: f"{v:.2f}")
    table["alerts"] = table["alerts"].map(
        lambda items: "; ".join(items) if items else ""
    )
    columns = [
        "machine",
        "hours_observed",
        "overspeed_events",
        "overspeed_time_pct",
        "overspeed_rate_per_hr",
        "tall_moving_events",
        "tall_moving_time_pct",
        "tall_moving_rate_per_hr",
        "height_toggle_rate_per_hr",
        "alerts",
    ]
    return table[columns].to_string(index=False)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    csv_files = load_forklift_files(args.data_dir)

    records = [
        summarize_machine(
            csv_file,
            args.raised_threshold,
            args.tall_threshold,
            args.overspeed_limit,
            args.tall_move_speed,
            args.toggle_limit,
            args.overspeed_limit_rate,
            args.tall_move_limit_rate,
        )
        for csv_file in csv_files
    ]
    summary_df = pd.DataFrame(records).sort_values(
        ["alert_count", "overspeed_rate_per_hr", "tall_moving_rate_per_hr"],
        ascending=[False, False, False],
    )

    print("Forklift safety/wear summary (height + speed derived):")
    print(format_table(summary_df))

    output_path = args.output or DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"\nWrote raw metrics to {output_path}")


if __name__ == "__main__":
    main()
