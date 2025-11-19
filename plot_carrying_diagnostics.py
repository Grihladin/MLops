"""
Generate diagnostic plots that compare the raw load signal to the debounced
Carrying feature for every annotated forklift CSV.

Each figure contains:
  1. Timeline overlay: raw load state vs. Carrying flag
  2. Histogram of Carrying segment durations
  3. Daily carrying share (bars) with number of carrying segments per day (line)
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
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "plots" / "carrying_diagnostics"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot diagnostics for the Carrying feature of each forklift."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with *_forklift.csv files containing Carrying (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store the diagnostic figures (default: %(default)s)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for saved PNGs (default: %(default)s)",
    )
    parser.add_argument(
        "--duration-bins",
        type=int,
        default=30,
        help="Number of bins in the carrying duration histogram (default: %(default)s)",
    )
    return parser.parse_args(argv)


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Timestamp", "Load", "Carrying"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    df["RawLoadState"] = pd.to_numeric(df["Load"], errors="coerce").fillna(0).gt(0)
    df["Carrying"] = df["Carrying"].astype(bool)

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
        return pd.DataFrame(columns=["start", "end", "carrying", "duration"])

    groups = df["Carrying"].ne(df["Carrying"].shift()).cumsum()
    segments = (
        df.groupby(groups)
        .agg(
            start=("Timestamp", "first"),
            end=("Timestamp", "last"),
            carrying=("Carrying", "first"),
            duration=("Duration", "sum"),
        )
        .reset_index(drop=True)
    )
    return segments


def compute_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "share", "segment_count"])

    indexed = df.set_index("Timestamp")
    share = (
        (indexed["Carrying"] * indexed["Duration"]) / indexed["Duration"]
    ).resample("1D").mean()

    segments = compute_segments(df)
    if segments.empty:
        counts = pd.Series(dtype=float)
    else:
        segment_start = segments.loc[segments["carrying"], "start"]
        counts = segment_start.dt.floor("1D").value_counts().sort_index()

    metrics = pd.DataFrame(
        {
            "share": share.fillna(0),
            "segment_count": counts.reindex(share.index, fill_value=0),
        }
    )
    metrics.index.name = "date"
    return metrics.reset_index()


def plot_diagnostics(
    df: pd.DataFrame,
    csv_path: Path,
    output_dir: Path,
    dpi: int,
    duration_bins: int,
) -> None:
    segments = compute_segments(df)
    carrying_segments = segments[segments["carrying"]]
    daily_metrics = compute_daily_metrics(df)

    fig, (ax_timeline, ax_hist, ax_daily) = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        dpi=dpi,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )

    # Timeline
    ax_timeline.plot(
        df["Timestamp"],
        df["RawLoadState"].astype(int),
        color="#bbbbbb",
        linewidth=1,
        alpha=0.5,
        label="Raw load state",
        drawstyle="steps-post",
    )
    ax_timeline.plot(
        df["Timestamp"],
        df["Carrying"].astype(int),
        color="#2ca02c",
        linewidth=2,
        label="Carrying",
        drawstyle="steps-post",
    )
    ax_timeline.set_ylim(-0.3, 1.3)
    ax_timeline.set_yticks([0, 1], labels=["Not carrying", "Carrying"])
    ax_timeline.set_ylabel("State")
    ax_timeline.set_title(f"Carrying diagnostics — {csv_path.stem}")
    ax_timeline.grid(True, axis="x", alpha=0.3)
    locator = mdates.AutoDateLocator()
    ax_timeline.xaxis.set_major_locator(locator)
    ax_timeline.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax_timeline.legend(loc="upper right")

    # Histogram of carrying durations
    if not carrying_segments.empty:
        durations_min = carrying_segments["duration"] / 60.0
        ax_hist.hist(
            durations_min,
            bins=duration_bins,
            color="#1f77b4",
            alpha=0.8,
            edgecolor="black",
        )
        ax_hist.set_xlabel("Carrying segment duration (minutes)")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Distribution of carrying durations")
    else:
        ax_hist.text(
            0.5,
            0.5,
            "No carrying segments",
            transform=ax_hist.transAxes,
            ha="center",
            va="center",
            color="#555555",
        )
        ax_hist.set_axis_off()

    # Daily share + counts
    if not daily_metrics.empty:
        ax_daily.bar(
            daily_metrics["date"],
            daily_metrics["share"],
            width=0.8,
            color="#9467bd",
            alpha=0.7,
            label="Carrying share",
        )
        ax_daily.set_ylim(0, 1)
        ax_daily.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax_daily.set_ylabel("Share of time carrying")
        ax_daily.grid(True, axis="y", alpha=0.3)

        ax2 = ax_daily.twinx()
        ax2.plot(
            daily_metrics["date"],
            daily_metrics["segment_count"],
            color="#ff7f0e",
            linewidth=2,
            marker="o",
            label="Carrying segments per day",
        )
        ax2.set_ylabel("Carrying segments per day")

        # combine legends
        lines = ax_daily.get_legend_handles_labels()
        lines2 = ax2.get_legend_handles_labels()
        ax_daily.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper left")

        daily_locator = mdates.AutoDateLocator()
        ax_daily.xaxis.set_major_locator(daily_locator)
        ax_daily.xaxis.set_major_formatter(mdates.ConciseDateFormatter(daily_locator))
    else:
        ax_daily.text(
            0.5,
            0.5,
            "Insufficient data for daily metrics",
            transform=ax_daily.transAxes,
            ha="center",
            va="center",
            color="#555555",
        )
        ax_daily.set_axis_off()

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{csv_path.stem}_carrying_diagnostics.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostics plot to {output_path}")  # noqa: T201


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    csv_files = sorted(args.data_dir.glob("*_forklift.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No *_forklift.csv files found in {args.data_dir}. "
            "Run add_carrying_feature.py first."
        )

    for csv_path in csv_files:
        df = load_dataframe(csv_path)
        plot_diagnostics(df, csv_path, args.output_dir, args.dpi, args.duration_bins)


if __name__ == "__main__":
    main()
