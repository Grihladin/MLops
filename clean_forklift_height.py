"""Filter forklift height samples and rewrite cleaned CSVs.

The script scans ``real_data_with_headers`` (or a user-provided directory),
keeps only forklift files (``*_forklift.csv``), removes rows whose height
outside the believable range, optionally rescales heights, and writes the cleaned CSVs to a destination
directory. Use it before plotting or training so downstream steps operate on
consistent height data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).parent / "real_data_with_headers"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "real_data_with_headers_height_cleaned"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove forklift height samples outside the expected range.",
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
        help="Where to store the filtered CSVs (default: %(default)s)",
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=0.0,
        help="Drop rows below this height in meters (default: %(default)s)",
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=6.7,
        help="Drop rows above this height in meters (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=("drop", "clip", "scale"),
        default="scale",
        help=(
            "How to handle out-of-range heights: "
            "'drop' removes the rows, 'clip' caps values to the bounds, "
            "and 'scale' linearly maps heights into the target range "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--source-min",
        type=float,
        default=None,
        help="Known minimum of the raw sensor range (used when --mode scale).",
    )
    parser.add_argument(
        "--source-max",
        type=float,
        default=None,
        help="Known maximum of the raw sensor range (used when --mode scale).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing output CSVs into the same directory as the inputs.",
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


def clean_dataframe(
    csv_path: Path,
    min_height: float,
    max_height: float,
    mode: str,
    source_min: float | None,
    source_max: float | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(csv_path)
    if "Timestamp" not in df or "Height" not in df:
        raise ValueError(f"{csv_path} is missing Timestamp and/or Height columns.")

    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df["Height"] = pd.to_numeric(df["Height"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Height"])

    stats = {"removed": 0, "adjusted": 0}

    if mode == "drop":
        mask = (df["Height"] >= min_height) & (df["Height"] <= max_height)
        stats["removed"] = int((~mask).sum())
        df = df[mask]

        if df.empty:
            raise ValueError(
                f"{csv_path} has no rows remaining after height filtering."
            )
    elif mode == "clip":
        clipped = df["Height"].clip(lower=min_height, upper=max_height)
        stats["adjusted"] = int((df["Height"] != clipped).sum())
        df = df.assign(Height=clipped)
    elif mode == "scale":
        data_min = df["Height"].min() if source_min is None else source_min
        data_max = df["Height"].max() if source_max is None else source_max
        if data_max <= data_min:
            raise ValueError(
                f"Invalid source range ({data_min}, {data_max}). "
                "Provide --source-min/--source-max or ensure data contains variation."
            )
        scaled = (df["Height"] - data_min) / (data_max - data_min)
        scaled = scaled.clip(0, 1)
        df["Height"] = scaled * (max_height - min_height) + min_height
        stats["adjusted"] = len(df)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    df = df.sort_values("Timestamp")
    df["Timestamp"] = df["Timestamp"].astype("int64")
    return df, stats


def ensure_output_dir(output_dir: Path, data_dir: Path, allow_same_dir: bool) -> None:
    if output_dir.resolve() == data_dir.resolve() and not allow_same_dir:
        raise ValueError(
            "Refusing to overwrite the source directory. "
            "Pass --overwrite if you really want to replace the originals."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def clean_files(
    csv_files: list[Path],
    output_dir: Path,
    min_height: float,
    max_height: float,
    mode: str,
    source_min: float | None,
    source_max: float | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_files:
        df, stats = clean_dataframe(
            csv_path,
            min_height,
            max_height,
            mode,
            source_min,
            source_max,
        )
        output_path = output_dir / csv_path.name
        df.to_csv(output_path, index=False)

        parts = [f"{csv_path.name}: wrote {len(df)} rows"]
        if stats["removed"]:
            parts.append(f"removed {stats['removed']}")
        if stats["adjusted"]:
            parts.append(f"adjusted {stats['adjusted']}")

        print(f"{'; '.join(parts)} → {output_path}")  # noqa: T201


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    csv_files = load_forklift_files(args.data_dir)
    ensure_output_dir(args.output_dir, args.data_dir, args.overwrite)
    clean_files(
        csv_files,
        args.output_dir,
        args.min_height,
        args.max_height,
        args.mode,
        args.source_min,
        args.source_max,
    )


if __name__ == "__main__":
    main()
