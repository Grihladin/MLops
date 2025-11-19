"""
Add a debounced boolean ``Carrying`` feature to every processed forklift CSV.

The script reads each ``*_forklift.csv`` file, combines the noisy ``Load`` flag
with fork-height / speed heuristics, collapses short pulses, optionally fills
tiny gaps, and ignores samples collected while the machine is off-duty (unless
``--allow-off-duty`` is provided). The cleaned Carrying signal is written to a
separate directory so downstream analyses can rely on a more stable flag.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).parent / "real_data_with_headers"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "real_data_with_carrying"
DEFAULT_MIN_DURATION = 45.0  # seconds required for a segment to count as carrying
DEFAULT_FILL_GAP = 1.0  # seconds of false gap we merge between true carrying pulses
DEFAULT_HEIGHT_THRESHOLD = 0.35  # meters the forks must be raised before trusting the load sensor
DEFAULT_LOAD_THRESHOLD = 0.5  # raw load value that indicates weight on the forks
DEFAULT_MAX_SPEED = 7.2  # km/h speed cap while carrying; faster samples drop to False
HEIGHT_SENSOR_MIN_OVERLAP_RATIO = 0.05  # min fraction of load samples that must have forks up
HEIGHT_SENSOR_MIN_OVERLAP_COUNT = 10  # absolute overlap samples required before trusting height
HEIGHT_SENSOR_MIN_LOAD_SAMPLES = 100  # only enforce the ratio once we have this many load samples


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a debounced boolean Carrying feature for each forklift CSV."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing *_forklift.csv files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store the annotated CSVs (default: %(default)s)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION,
        help="Seconds a load segment must last to be considered true (default: %(default)s)",
    )
    parser.add_argument(
        "--fill-gap",
        type=float,
        default=DEFAULT_FILL_GAP,
        help=(
            "Seconds of not-carrying allowed between two carrying segments "
            "before they get merged (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--height-threshold",
        type=float,
        default=DEFAULT_HEIGHT_THRESHOLD,
        help=(
            "Minimum fork height (meters) required to trust the Carrying flag "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--load-threshold",
        type=float,
        default=DEFAULT_LOAD_THRESHOLD,
        help="Load value above which the raw sensor is considered high (default: %(default)s)",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=DEFAULT_MAX_SPEED,
        help=(
            "Optional maximum speed allowed while carrying; samples faster than "
            "this are forced to not carrying"
        ),
    )
    parser.add_argument(
        "--allow-off-duty",
        action="store_true",
        help="Process samples even if OnDuty == 0 (default: ignore them).",
    )
    return parser.parse_args(argv)


def find_forklift_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*_forklift.csv"))
    if not files:
        raise FileNotFoundError(
            f"No *_forklift.csv files found in {data_dir}. "
            "Run assign_columns.py first or point to a directory that contains forklift data."
        )
    return files


def compute_carrying_mask(
    df: pd.DataFrame,
    min_duration: float,
    fill_gap: float,
    height_threshold: float,
    load_threshold: float,
    max_speed: float | None,
    allow_off_duty: bool,
) -> pd.Series:
    ts = pd.to_datetime(df["Timestamp"], unit="ms", errors="coerce")
    load = pd.to_numeric(df["Load"], errors="coerce").fillna(0)

    height_raw = df.get("Height")
    height_sensor_available = False
    if height_raw is not None:
        height = pd.to_numeric(height_raw, errors="coerce")
        height_sensor_available = height.notna().any()
        height = height.fillna(0)
    else:
        height = pd.Series(0.0, index=df.index)

    speed_raw = df.get("Speed")
    if speed_raw is not None:
        speed = pd.to_numeric(speed_raw, errors="coerce")
    else:
        speed = pd.Series(index=df.index, dtype=float)

    onduty_raw = df.get("OnDuty")
    if onduty_raw is not None:
        on_duty = pd.to_numeric(onduty_raw, errors="coerce").fillna(0).gt(0)
    else:
        on_duty = pd.Series(False, index=df.index)

    load_state = load.gt(load_threshold)
    forks_up = height.gt(height_threshold)

    load_samples = int(load_state.sum())
    overlap_samples = int((load_state & forks_up).sum())
    overlap_ratio = overlap_samples / load_samples if load_samples else 0.0
    ratio_requirement_met = (
        overlap_ratio >= HEIGHT_SENSOR_MIN_OVERLAP_RATIO
        if load_samples >= HEIGHT_SENSOR_MIN_LOAD_SAMPLES
        else True
    )
    height_sensor_trustworthy = (
        height_sensor_available
        and overlap_samples >= HEIGHT_SENSOR_MIN_OVERLAP_COUNT
        and ratio_requirement_met
    )

    raw_state = load_state & forks_up if height_sensor_trustworthy else load_state

    if max_speed is not None and not pd.isna(max_speed):
        moving_slow = speed.le(max_speed) | speed.isna()
        raw_state &= moving_slow

    raw_state = raw_state.fillna(False)

    valid = ts.notna()
    if (not allow_off_duty) and "OnDuty" in df:
        valid &= on_duty

    if not valid.any():
        return pd.Series(False, index=df.index)

    working = (
        pd.DataFrame(
            {
                "Timestamp": ts[valid],
                "State": raw_state[valid].fillna(False),
            }
        )
        .sort_values("Timestamp")
        .copy()
    )

    working["NextTimestamp"] = working["Timestamp"].shift(-1)
    working["Duration"] = (
        working["NextTimestamp"] - working["Timestamp"]
    ).dt.total_seconds()
    median_duration = working["Duration"].median()
    if pd.isna(median_duration) or median_duration <= 0:
        median_duration = 1.0
    working["Duration"] = working["Duration"].fillna(median_duration)

    change_groups = working["State"].ne(working["State"].shift()).cumsum()
    change_groups.name = "group"
    segments = (
        working.groupby(change_groups)
        .agg(State=("State", "first"), Duration=("Duration", "sum"))
        .reset_index()
    )
    segments["final_state"] = False

    true_segments = segments["State"]
    segments.loc[true_segments, "final_state"] = (
        segments.loc[true_segments, "Duration"] >= min_duration
    )

    if fill_gap > 0:
        false_segments = segments.index[
            (~segments["State"]) & (segments["Duration"] <= fill_gap)
        ]
        for idx in false_segments:
            prev_idx = idx - 1
            next_idx = idx + 1
            if prev_idx < 0 or next_idx >= len(segments):
                continue
            if segments.at[prev_idx, "final_state"] and segments.at[next_idx, "final_state"]:
                segments.at[idx, "final_state"] = True

    final_map = segments.set_index("group")["final_state"]
    carrying_working = change_groups.map(final_map)

    carrying_full = pd.Series(False, index=df.index)
    carrying_full.loc[working.index] = carrying_working.astype(bool).values
    return carrying_full


def annotate_file(
    csv_path: Path,
    output_dir: Path,
    *,
    min_duration: float,
    fill_gap: float,
    height_threshold: float,
    load_threshold: float,
    max_speed: float | None,
    allow_off_duty: bool,
) -> None:
    df = pd.read_csv(csv_path)
    if "Timestamp" not in df or "Load" not in df:
        raise ValueError(f"{csv_path} is missing Timestamp and/or Load columns.")

    df["Carrying"] = compute_carrying_mask(
        df,
        min_duration=min_duration,
        fill_gap=fill_gap,
        height_threshold=height_threshold,
        load_threshold=load_threshold,
        max_speed=max_speed,
        allow_off_duty=allow_off_duty,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / csv_path.name
    df.to_csv(destination, index=False)
    print(f"Wrote Carrying feature to {destination}")  # noqa: T201


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    csv_files = find_forklift_files(args.data_dir)

    for csv_path in csv_files:
        annotate_file(
            csv_path,
            args.output_dir,
            min_duration=args.min_duration,
            fill_gap=args.fill_gap,
            height_threshold=args.height_threshold,
            load_threshold=args.load_threshold,
            max_speed=args.max_speed,
            allow_off_duty=args.allow_off_duty,
        )


if __name__ == "__main__":
    main()
