"""
Assign human-readable column names to the forklift telemetry files.

Each CSV in the input directory currently lacks headers. This script reads every
file, applies the agreed column order, and writes the result to an output
directory (creating it if needed). By default the cleaned files land in a new
``real_data_with_headers`` folder so the raw data remains untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

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


def assign_headers(csv_path: Path) -> pd.DataFrame:
    """Load a raw forklift CSV and attach the canonical headers."""
    return pd.read_csv(
        csv_path,
        sep=";",
        header=None,
        names=COLUMNS,
        engine="python",
    )


def process_directory(data_dir: Path, output_dir: Path) -> None:
    """Apply headers to every CSV in ``data_dir`` and write them to ``output_dir``."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_files:
        df = assign_headers(csv_file)
        df = df.sort_values("Timestamp").reset_index(drop=True)
        
        total_rows = len(df)

        # Remove consecutive duplicate OnDuty zeros
        zero_mask = df["OnDuty"].eq(0)
        redundant_zeros = zero_mask & zero_mask.shift(fill_value=False)
        num_deleted = redundant_zeros.sum()
        percentage_deleted = (num_deleted / total_rows * 100) if total_rows > 0 else 0
        df = df[~redundant_zeros].reset_index(drop=True)

        # Label the file based on whether any Height readings are non-zero
        height_values = pd.to_numeric(df["Height"], errors="coerce")
        is_forklift = height_values.fillna(0).ne(0).any()
        label = "forklift" if is_forklift else "truck"

        destination = output_dir / f"{csv_file.stem}_{label}{csv_file.suffix}"

        legacy_path = output_dir / csv_file.name
        if legacy_path.exists() and legacy_path != destination:
            legacy_path.unlink()

        df.to_csv(destination, index=False)
        print(f"Wrote headers to {destination} (deleted {num_deleted}/{total_rows} redundant OnDuty=0 rows, {percentage_deleted:.2f}%)")  # noqa: T201


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add the agreed telemetry column names to each forklift CSV."
            " The cleaned files are written to a separate directory by default."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "real_data",
        help="Directory that holds the raw forklift CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "real_data_with_headers",
        help="Where to write the header-enriched CSV files (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    process_directory(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
