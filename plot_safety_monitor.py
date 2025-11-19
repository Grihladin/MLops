"""
Visualize forklift safety/wear metrics derived from height + speed.

The script recomputes the same statistics as monitor_safety.py and renders:
  * Scatter plot: overspeed events per hour (x) vs. tall-lift-while-moving per
    hour (y), color-coded by height toggle rate. Alert thresholds appear as
    dashed lines, and point size increases with alert count.
  * Bar chart: height toggle rate per hour for each machine, highlighting
    forklifts that triggered alerts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from monitor_safety import (
    DEFAULT_DATA_DIR as CLEANED_DATA_DIR,
    load_forklift_files,
    summarize_machine,
)

DEFAULT_OUTPUT = Path(__file__).parent / "plots" / "safety_monitor.png"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot safety/wear metrics derived from monitor_safety."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=CLEANED_DATA_DIR,
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


def compute_metrics(args: argparse.Namespace) -> pd.DataFrame:
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
    df = pd.DataFrame(records)
    df["alerts_text"] = df["alerts"].map(lambda items: "; ".join(items))
    df["has_alert"] = df["alert_count"].gt(0)
    return df


def plot_safety(df: pd.DataFrame, args: argparse.Namespace) -> None:
    fig, (ax_scatter, ax_bar) = plt.subplots(
        2,
        1,
        figsize=(13, 9),
        dpi=args.dpi,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    sizes = 80 + df["alert_count"] * 60
    scatter = ax_scatter.scatter(
        df["overspeed_rate_per_hr"],
        df["tall_moving_rate_per_hr"],
        c=df["height_toggle_rate_per_hr"],
        cmap="plasma",
        s=sizes,
        edgecolor="black",
        linewidths=0.8,
        alpha=0.9,
    )
    for _, row in df.iterrows():
        label = row["machine"].replace("_forklift", "")
        ax_scatter.text(
            row["overspeed_rate_per_hr"] + 0.05,
            row["tall_moving_rate_per_hr"] + 0.05,
            label,
            fontsize=8,
        )

    ax_scatter.axvline(
        args.overspeed_limit_rate,
        color="red",
        linestyle="--",
        linewidth=1,
        label="Overspeed limit rate",
    )
    ax_scatter.axhline(
        args.tall_move_limit_rate,
        color="orange",
        linestyle="--",
        linewidth=1,
        label="Tall-lift limit rate",
    )
    ax_scatter.set_xlabel("Overspeed events per hour")
    ax_scatter.set_ylabel("Tall-lift-while-moving events per hour")
    ax_scatter.set_title("Overspeed vs. tall-lift activity (color = toggle rate)")
    ax_scatter.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax_scatter)
    cbar.set_label("Height toggles per hour")
    ax_scatter.legend(loc="upper left")

    ordered = df.sort_values("height_toggle_rate_per_hr", ascending=False)
    labels = ordered["machine"].str.replace("_forklift", "", regex=False)
    colors = ["#d62728" if flag else "#1f77b4" for flag in ordered["has_alert"]]
    ax_bar.bar(
        range(len(ordered)),
        ordered["height_toggle_rate_per_hr"],
        color=colors,
    )
    ax_bar.axhline(args.toggle_limit, color="red", linestyle="--", linewidth=1)
    ax_bar.set_ylabel("Height toggles per hour")
    ax_bar.set_xlabel("Machine")
    ax_bar.set_title("Height oscillations (red = triggered alert)")
    ax_bar.set_xticks(range(len(ordered)), labels, rotation=45, ha="right")
    ax_bar.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax_bar.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved safety visualization to {args.output}")  # noqa: T201


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = compute_metrics(args)
    if metrics.empty:
        raise ValueError("No safety metrics computed.")
    plot_safety(metrics, args)


if __name__ == "__main__":
    main()
