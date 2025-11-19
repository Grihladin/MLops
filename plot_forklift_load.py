"""
Generate informative load-state plots for every processed forklift CSV.

Each figure pairs a timeline of carrying intervals with a daily utilization
summary, making the binary load signal easier to interpret than a dense
step chart. Run this after ``assign_columns.py`` so the input files already
have headers and forklift/truck labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).parent / "real_data_with_headers"
DEFAULT_PLOTS_DIR = Path(__file__).parent / "plots" / "forklift_loads"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot load-state timelines plus daily utilization for each forklift."
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
        default=DEFAULT_PLOTS_DIR,
        help="Directory to store the generated PNGs (default: %(default)s)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for the saved plots (default: %(default)s)",
    )
    parser.add_argument(
        "--daily-frequency",
        default="1D",
        help="Pandas offset alias for the utilization bars (default: %(default)s)",
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


def compute_load_segments(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if df.empty:
        return []

    change_groups = df["LoadState"].ne(df["LoadState"].shift()).cumsum()
    median_delta = df["Timestamp"].diff().median()
    if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
        median_delta = pd.Timedelta(seconds=1)

    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, group in df.groupby(change_groups):
        if group["LoadState"].iloc[0] != 1:
            continue
        start = group["Timestamp"].iloc[0]
        end = group["Timestamp"].iloc[-1] + median_delta
        segments.append((start, end))
    return segments


def compute_utilisation(
    df: pd.DataFrame,
    frequency: str,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    indexed = df.set_index("Timestamp")
    load_time = (indexed["LoadState"] * indexed["Duration"]).resample(frequency).sum()
    total_time = indexed["Duration"].resample(frequency).sum()
    utilisation = (load_time / total_time).fillna(0)
    return utilisation


def plot_load(
    df: pd.DataFrame,
    csv_path: Path,
    output_dir: Path,
    dpi: int,
    frequency: str,
) -> None:
    segments = compute_load_segments(df)
    utilisation = compute_utilisation(df, frequency)

    fig, (ax_timeline, ax_daily) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        dpi=dpi,
        sharex=False,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # Timeline axis
    if segments:
        for start, end in segments:
            ax_timeline.plot(
                [start, end],
                [1, 1],
                color="#d62728",
                linewidth=6,
                solid_capstyle="butt",
            )
    ax_timeline.hlines(
        0,
        df["Timestamp"].min(),
        df["Timestamp"].max(),
        colors="#bbbbbb",
        linewidth=2,
        alpha=0.6,
    )
    ax_timeline.set_ylim(-0.3, 1.3)
    ax_timeline.set_yticks([0, 1], labels=["Not carrying load", "Carrying load"])
    ax_timeline.set_ylabel("Load state")
    ax_timeline.set_title(f"Load State & Utilisation — {csv_path.stem}")
    ax_timeline.grid(True, axis="x", alpha=0.3)
    timeline_locator = mdates.AutoDateLocator()
    ax_timeline.xaxis.set_major_locator(timeline_locator)
    ax_timeline.xaxis.set_major_formatter(mdates.ConciseDateFormatter(timeline_locator))

    if not segments:
        ax_timeline.text(
            0.5,
            0.5,
            "No carrying events detected",
            transform=ax_timeline.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#555555",
        )

    # Utilisation axis
    if not utilisation.empty:
        ax_daily.bar(
            utilisation.index,
            utilisation.values,
            width=0.8,
            color="#1f77b4",
        )
        ax_daily.set_ylim(0, 1)
        ax_daily.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax_daily.set_ylabel(f"Share of time loaded ({frequency})")
        ax_daily.grid(True, axis="y", alpha=0.3)
        daily_locator = mdates.AutoDateLocator()
        ax_daily.xaxis.set_major_locator(daily_locator)
        ax_daily.xaxis.set_major_formatter(mdates.ConciseDateFormatter(daily_locator))
    else:
        ax_daily.text(
            0.5,
            0.5,
            "Not enough data to compute utilisation",
            transform=ax_daily.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#555555",
        )
        ax_daily.set_axis_off()

    fig.autofmt_xdate()
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{csv_path.stem}_load.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")  # noqa: T201


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    csv_files = load_forklift_files(args.data_dir)

    for csv_file in csv_files:
        df = prepare_dataframe(csv_file)
        plot_load(df, csv_file, args.output_dir, args.dpi, args.daily_frequency)


if __name__ == "__main__":
    main()
