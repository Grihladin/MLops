"""
Generate Height vs. Timestamp plots for every cleaned forklift CSV.

This is identical to ``plot_forklift_height.py`` but defaults to the
``real_data_with_headers_height_cleaned`` directory that
``clean_forklift_height.py`` produces. Point it at another directory with
``*_forklift.csv`` files if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).parent / "real_data_with_headers_height_cleaned"
DEFAULT_PLOTS_DIR = Path(__file__).parent / "plots" / "forklift_heights_cleaned"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot height changes over time for each cleaned forklift CSV."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing cleaned *_forklift.csv files (default: %(default)s)",
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
    return parser.parse_args(argv)


def load_forklift_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*_forklift.csv"))
    if not files:
        raise FileNotFoundError(
            f"No *_forklift.csv files found in {data_dir}. "
            "Run clean_forklift_height.py first or point to a directory that contains forklift data."
        )
    return files


def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Timestamp" not in df or "Height" not in df:
        raise ValueError(f"{csv_path} is missing Timestamp and/or Height columns.")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp")

    if df.empty:
        raise ValueError(f"{csv_path} has no valid Timestamp/Height rows after preprocessing.")
    return df


def plot_height(
    df: pd.DataFrame,
    csv_path: Path,
    output_dir: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.vlines(
        df["Timestamp"],
        ymin=0,
        ymax=df["Height"],
        colors="#1f77b4",
        linewidth=1,
    )
    ax.set_title(f"Height Over Time — {csv_path.stem}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Height")
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{csv_path.stem}_height.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")  # noqa: T201


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    csv_files = load_forklift_files(args.data_dir)

    for csv_file in csv_files:
        df = prepare_dataframe(csv_file)
        plot_height(
            df,
            csv_file,
            args.output_dir,
            args.dpi,
        )


if __name__ == "__main__":
    main()
