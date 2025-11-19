"""
Visualize OnDuty deletion statistics across all processed files.

This script reads the raw data, identifies redundant OnDuty=0 rows that would be
deleted, and creates visualizations showing the deletion statistics per file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

COLUMNS = [
    "FFFID",
    "Height",
    "Load",
    "OnDuty",
    "Timestamp",
    "Latitude",
    "Longitude",
    "Speed",
]


def analyze_onduty_deletions(csv_path: Path) -> dict:
    """Analyze how many OnDuty=0 rows would be deleted from a file."""
    df = pd.read_csv(
        csv_path,
        sep=";",
        header=None,
        names=COLUMNS,
        engine="python",
    )
    df = df.sort_values("Timestamp").reset_index(drop=True)
    
    total_rows = len(df)
    
    # Identify consecutive duplicate OnDuty zeros
    zero_mask = df["OnDuty"].eq(0)
    redundant_zeros = zero_mask & zero_mask.shift(fill_value=False)
    num_deleted = redundant_zeros.sum()
    percentage_deleted = (num_deleted / total_rows * 100) if total_rows > 0 else 0
    
    return {
        "file": csv_path.stem,
        "total_rows": total_rows,
        "deleted_rows": num_deleted,
        "kept_rows": total_rows - num_deleted,
        "percentage_deleted": percentage_deleted,
    }


def create_visualizations(data_dir: Path, output_dir: Path) -> None:
    """Create visualizations of OnDuty deletion statistics."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    # Analyze all files
    results = [analyze_onduty_deletions(f) for f in csv_files]
    df_stats = pd.DataFrame(results)
    
    # Save statistics to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = output_dir / "onduty_deletion_stats.csv"
    df_stats.to_csv(stats_file, index=False)
    print(f"Saved statistics to {stats_file}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("OnDuty Deletion Statistics", fontsize=16, fontweight="bold")
    
    # 1. Bar chart: Deleted vs Kept rows per file
    ax1 = axes[0, 0]
    x_pos = range(len(df_stats))
    width = 0.35
    ax1.bar([p - width/2 for p in x_pos], df_stats["kept_rows"], 
            width, label="Kept", color="green", alpha=0.7)
    ax1.bar([p + width/2 for p in x_pos], df_stats["deleted_rows"], 
            width, label="Deleted", color="red", alpha=0.7)
    ax1.set_xlabel("File")
    ax1.set_ylabel("Number of Rows")
    ax1.set_title("Rows Kept vs Deleted per File")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df_stats["file"], rotation=45, ha="right", fontsize=8)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    
    # 2. Percentage deleted per file
    ax2 = axes[0, 1]
    colors = plt.cm.RdYlGn_r(df_stats["percentage_deleted"] / df_stats["percentage_deleted"].max())
    bars = ax2.bar(x_pos, df_stats["percentage_deleted"], color=colors)
    ax2.set_xlabel("File")
    ax2.set_ylabel("Percentage Deleted (%)")
    ax2.set_title("Percentage of Rows Deleted per File")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df_stats["file"], rotation=45, ha="right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, pct) in enumerate(zip(bars, df_stats["percentage_deleted"])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=7)
    
    # 3. Pie chart: Overall deletion summary
    ax3 = axes[1, 0]
    total_kept = df_stats["kept_rows"].sum()
    total_deleted = df_stats["deleted_rows"].sum()
    total_overall = total_kept + total_deleted
    
    wedges, texts, autotexts = ax3.pie([total_kept, total_deleted], 
            labels=[f"Kept\n{total_kept:,} rows", f"Deleted\n{total_deleted:,} rows"],
            autopct="%1.2f%%",
            colors=["green", "red"],
            startangle=90)
    for w in wedges:
        w.set_alpha(0.7)
    ax3.set_title(f"Overall Deletion Summary\n(Total: {total_overall:,} rows)")
    
    # 4. Statistics table
    ax4 = axes[1, 1]
    ax4.axis("off")
    
    summary_stats = [
        ["Total Files", len(df_stats)],
        ["Total Original Rows", f"{df_stats['total_rows'].sum():,}"],
        ["Total Deleted Rows", f"{df_stats['deleted_rows'].sum():,}"],
        ["Total Kept Rows", f"{df_stats['kept_rows'].sum():,}"],
        ["Average % Deleted", f"{df_stats['percentage_deleted'].mean():.2f}%"],
        ["Max % Deleted", f"{df_stats['percentage_deleted'].max():.2f}%"],
        ["Min % Deleted", f"{df_stats['percentage_deleted'].min():.2f}%"],
        ["Files with 0% Deleted", f"{(df_stats['percentage_deleted'] == 0).sum()}"],
    ]
    
    table = ax4.table(cellText=summary_stats, 
                     colLabels=["Metric", "Value"],
                     cellLoc="left",
                     loc="center",
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header
    for i in range(2):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    ax4.set_title("Summary Statistics", fontweight="bold", pad=20)
    
    plt.tight_layout()
    
    # Save figure
    plot_file = output_dir / "onduty_deletion_visualization.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {plot_file}")
    
    plt.show()
    
    # Print summary to console
    print("\n" + "="*60)
    print("OnDuty Deletion Summary")
    print("="*60)
    for metric, value in summary_stats:
        print(f"{metric:.<40} {value}")
    print("="*60)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize OnDuty deletion statistics across forklift CSV files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "real_data",
        help="Directory with raw CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "plots",
        help="Where to save the visualization (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    create_visualizations(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
