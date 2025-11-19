"""
Highlight load-sensor anomalies using scatter and bar plots.

This visualization overlays the anomaly thresholds on top of the metrics used in
``detect_load_anomalies.py`` so you can quickly see which forklifts are noisy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from detect_load_anomalies import classify_machine, compute_metrics

DEFAULT_DATA_DIR = Path(__file__).parent / "real_data_with_headers"
DEFAULT_OUTPUT = Path(__file__).parent / "plots" / "load_anomaly_visualization.png"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize load-sensor anomalies with threshold overlays."
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
        help="Short load ratio that marks an anomaly (default: %(default)s)",
    )
    parser.add_argument(
        "--toggle-limit",
        type=float,
        default=12.0,
        help="Toggles per hour that mark an anomaly (default: %(default)s)",
    )
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=0.2,
        help="Speed threshold (km/h) used to count fast load flips (default: %(default)s)",
    )
    parser.add_argument(
        "--fast-flip-limit",
        type=float,
        default=2.0,
        help="Fast load flips per hour that mark an anomaly (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the visualization (default: %(default)s)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for the saved figure (default: %(default)s)",
    )
    return parser.parse_args(argv)


def build_plot_data(summary_df: pd.DataFrame, limits: dict[str, float]) -> pd.DataFrame:
    df = summary_df.copy()
    df["short_load_ratio_pct"] = df["short_load_ratio"] * 100
    df["duty_cycle_pct"] = df["duty_cycle"] * 100
    df["anomaly_reasons"] = df.apply(
        classify_machine,
        axis=1,
        limits=limits,
    )
    df["is_anomaly"] = df["anomaly_reasons"].map(bool)
    return df


def plot_anomalies(df: pd.DataFrame, limits: dict[str, float], output: Path, dpi: int) -> None:
    fig, (ax_scatter, ax_bar) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        dpi=dpi,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # Scatter: short ratio vs fast flips/hour
    scatter = ax_scatter.scatter(
        df["short_load_ratio_pct"],
        df["fast_flips_per_hour"],
        c=df["toggles_per_hour"],
        cmap="plasma",
        s=120,
        edgecolor="black",
        linewidths=1,
        alpha=0.85,
    )
    for _, row in df.iterrows():
        ax_scatter.text(
            row["short_load_ratio_pct"] + 0.5,
            row["fast_flips_per_hour"] + 0.2,
            row["machine"].replace("_forklift", ""),
            fontsize=8,
        )
    ax_scatter.axvline(
        limits["short_ratio_limit"] * 100,
        color="red",
        linestyle="--",
        linewidth=1,
        label="Short ratio limit",
    )
    ax_scatter.axhline(
        limits["fast_flip_limit"],
        color="orange",
        linestyle="--",
        linewidth=1,
        label="Fast flip limit",
    )
    ax_scatter.set_xlabel("Short load ratio (%)")
    ax_scatter.set_ylabel("Fast flips per hour")
    ax_scatter.set_title("Load-sensor anomaly thresholds (color = toggles per hour)")
    ax_scatter.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax_scatter)
    cbar.set_label("Toggles per hour")
    ax_scatter.legend(loc="upper left")

    # Bar chart for toggles per hour, highlighting anomalies
    ordered = df.sort_values("toggles_per_hour", ascending=False)
    labels = ordered["machine"].str.replace("_forklift", "", regex=False)
    positions = range(len(labels))
    colors = ["#d62728" if flag else "#1f77b4" for flag in ordered["is_anomaly"]]
    ax_bar.bar(
        positions,
        ordered["toggles_per_hour"],
        color=colors,
    )
    ax_bar.axhline(limits["toggle_limit"], color="orange", linestyle="--", linewidth=1)
    ax_bar.set_ylabel("Toggles per hour")
    ax_bar.set_xlabel("Machine")
    ax_bar.set_title("Toggle intensity (red = flagged)")
    ax_bar.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax_bar.set_xticks(positions, labels, rotation=45, ha="right", fontsize=9)
    ax_bar.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved anomaly visualization to {output}")  # noqa: T201


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = compute_metrics(
        args.data_dir,
        args.short_threshold,
        args.speed_threshold,
    )
    limits = {
        "short_ratio_limit": args.short_ratio_limit,
        "toggle_limit": args.toggle_limit,
        "fast_flip_limit": args.fast_flip_limit,
    }
    plot_df = build_plot_data(metrics, limits)
    plot_anomalies(plot_df, limits, args.output, args.dpi)


if __name__ == "__main__":
    main()
