"""
Visualize the debounced boolean Carrying feature for each annotated forklift CSV.

The script expects the files produced by ``add_carrying_feature.py``—they must
include a ``Carrying`` column alongside the original telemetry. Each plot shows
the carrying timeline and the share of time spent carrying per day.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).parent / "real_data_with_carrying"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "plots" / "carrying"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the Carrying feature timeline for each forklift CSV."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing *_forklift.csv files with Carrying column (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save the carrying plots (default: %(default)s)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for the saved plots (default: %(default)s)",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="1D",
        help="Pandas offset alias for carrying share bars (default: %(default)s)",
    )
    return parser.parse_args(argv)


def find_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*_forklift.csv"))
    if not files:
        raise FileNotFoundError(
            f"No *_forklift.csv files found in {data_dir}. "
            "Run add_carrying_feature.py first."
        )
    return files


def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Timestamp", "Carrying"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", errors="coerce")
    df["Carrying"] = df["Carrying"].astype(bool)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    df["NextTimestamp"] = df["Timestamp"].shift(-1)
    df["Duration"] = (df["NextTimestamp"] - df["Timestamp"]).dt.total_seconds()
    median_duration = df["Duration"].median()
    if pd.isna(median_duration) or median_duration <= 0:
        median_duration = 1.0
    df["Duration"] = df["Duration"].fillna(median_duration)
    df = df.drop(columns=["NextTimestamp"])
    return df


def compute_daily_share(df: pd.DataFrame, resample: str) -> pd.Series:
    indexed = df.set_index("Timestamp")
    carrying_seconds = (indexed["Carrying"] * indexed["Duration"]).resample(resample).sum()
    total_seconds = indexed["Duration"].resample(resample).sum()
    return (carrying_seconds / total_seconds).fillna(0)


def plot_carrying(
    df: pd.DataFrame,
    csv_path: Path,
    output_dir: Path,
    dpi: int,
    resample: str,
) -> None:
    daily_share = compute_daily_share(df, resample)
    change_groups = df["Carrying"].ne(df["Carrying"].shift()).cumsum()
    segments = (
        df.groupby(change_groups)
        .agg(
            start=("Timestamp", "first"),
            end=("Timestamp", "last"),
            carrying=("Carrying", "first"),
        )
        .reset_index(drop=True)
    )

    fig, (ax_timeline, ax_daily) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        dpi=dpi,
        sharex=False,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    if not segments.empty:
        for _, segment in segments.iterrows():
            color = "#2ca02c" if segment["carrying"] else "#bbbbbb"
            alpha = 0.8 if segment["carrying"] else 0.4
            ax_timeline.plot(
                [segment["start"], segment["end"]],
                [int(segment["carrying"])] * 2,
                color=color,
                linewidth=6,
                alpha=alpha,
                solid_capstyle="butt",
            )
    ax_timeline.set_ylim(-0.3, 1.3)
    ax_timeline.set_yticks([0, 1], labels=["Not carrying", "Carrying"])
    ax_timeline.set_ylabel("Carrying state")
    ax_timeline.set_title(f"Carrying timeline — {csv_path.stem}")
    ax_timeline.grid(True, axis="x", alpha=0.3)
    locator = mdates.AutoDateLocator()
    ax_timeline.xaxis.set_major_locator(locator)
    ax_timeline.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    if not daily_share.empty:
        ax_daily.bar(
            daily_share.index,
            daily_share.values,
            width=0.8,
            color="#1f77b4",
        )
        ax_daily.set_ylim(0, 1)
        ax_daily.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax_daily.set_ylabel(f"Carrying share ({resample})")
        ax_daily.grid(True, axis="y", alpha=0.3)
        daily_locator = mdates.AutoDateLocator()
        ax_daily.xaxis.set_major_locator(daily_locator)
        ax_daily.xaxis.set_major_formatter(mdates.ConciseDateFormatter(daily_locator))
    else:
        ax_daily.text(
            0.5,
            0.5,
            "Not enough data for carrying stats",
            transform=ax_daily.transAxes,
            ha="center",
            va="center",
            color="#555555",
        )
        ax_daily.set_axis_off()

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{csv_path.stem}_carrying.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved carrying plot to {output_path}")  # noqa: T201


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    csv_files = find_files(args.data_dir)

    for csv_path in csv_files:
        df = prepare_dataframe(csv_path)
        plot_carrying(df, csv_path, args.output_dir, args.dpi, args.resample)


if __name__ == "__main__":
    main()
