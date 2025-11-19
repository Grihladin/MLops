"""Dagster assets for ingesting and preprocessing forklift telemetry."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Type, TypeVar

import pandas as pd
from dagster import AssetExecutionContext, Config, MetadataValue, asset

from add_carrying_feature import (
    DEFAULT_FILL_GAP,
    DEFAULT_HEIGHT_THRESHOLD,
    DEFAULT_LOAD_THRESHOLD,
    DEFAULT_MAX_SPEED,
    DEFAULT_MIN_DURATION,
    annotate_file,
    find_forklift_files,
)
from assign_columns import process_directory
from clean_forklift_height import clean_files, load_forklift_files

from ..resources import LakeFSResource
from ..utils import paths
from ..utils.dataset_refs import DatasetArtifact


class RawIngestConfig(Config):
    prefix: str = "real_data"


class HeightCleaningConfig(Config):
    min_height: float = 0.0
    max_height: float = 6.7
    mode: Literal["drop", "clip", "scale"] = "scale"
    source_min: float | None = None
    source_max: float | None = None


class CarryingConfig(Config):
    min_duration: float = DEFAULT_MIN_DURATION
    fill_gap: float = DEFAULT_FILL_GAP
    height_threshold: float = DEFAULT_HEIGHT_THRESHOLD
    load_threshold: float = DEFAULT_LOAD_THRESHOLD
    max_speed: float | None = DEFAULT_MAX_SPEED
    allow_off_duty: bool = False


class TrainingDatasetConfig(Config):
    max_rows: int | None = None


TConfig = TypeVar("TConfig", bound=Config)


def _load_config(
    context: AssetExecutionContext,
    config_type: Type[TConfig],
) -> TConfig:
    raw_value = getattr(context, "op_config", None)
    if isinstance(raw_value, config_type):
        return raw_value
    if raw_value is None:
        return config_type()
    if isinstance(raw_value, dict):
        return config_type(**raw_value)
    if hasattr(raw_value, "model_dump"):
        return config_type(**raw_value.model_dump())
    raise ValueError(
        f"Unsupported config payload for {config_type.__name__}: {raw_value!r}"
    )


@asset(
    group_name="ingest",
    io_manager_key="artifact_io_manager",
    config_schema=RawIngestConfig.to_config_schema(),
)
def raw_forklift_snapshot(
    context,
    lakefs: LakeFSResource,
) -> DatasetArtifact:
    config = _load_config(context, RawIngestConfig)
    destination = paths.artifact_run_dir("raw", context.run_id)
    local_path, commit_id = lakefs.download_prefix(config.prefix, destination)
    csv_files = sorted(local_path.glob("*.csv"))
    dataset = DatasetArtifact(
        name="raw_csv",
        path=str(local_path),
        file_count=len(csv_files),
        version=commit_id or lakefs.branch,
    )
    context.add_output_metadata(
        {
            "local_path": MetadataValue.path(str(local_path)),
            "file_count": len(csv_files),
            "lakefs_branch": lakefs.branch,
            "commit_id": commit_id or "unknown",
        }
    )
    return dataset


@asset(
    group_name="processing",
    io_manager_key="artifact_io_manager",
)
def header_assigned_dataset(
    context,
    raw_forklift_snapshot: DatasetArtifact,
) -> DatasetArtifact:
    output_dir = paths.artifact_run_dir("with_headers", context.run_id)
    process_directory(Path(raw_forklift_snapshot.path), output_dir)
    csv_files = sorted(output_dir.glob("*.csv"))
    dataset = DatasetArtifact(
        name="with_headers",
        path=str(output_dir),
        file_count=len(csv_files),
        version=raw_forklift_snapshot.version,
    )
    context.add_output_metadata(
        {
            "file_count": len(csv_files),
            "output_dir": MetadataValue.path(str(output_dir)),
        }
    )
    return dataset


@asset(
    group_name="processing",
    io_manager_key="artifact_io_manager",
    config_schema=HeightCleaningConfig.to_config_schema(),
)
def cleaned_height_dataset(
    context,
    header_assigned_dataset: DatasetArtifact,
) -> DatasetArtifact:
    config = _load_config(context, HeightCleaningConfig)
    source_dir = Path(header_assigned_dataset.path)
    csv_files = load_forklift_files(source_dir)
    output_dir = paths.artifact_run_dir("height_cleaned", context.run_id)
    clean_files(
        csv_files,
        output_dir,
        min_height=config.min_height,
        max_height=config.max_height,
        mode=config.mode,
        source_min=config.source_min,
        source_max=config.source_max,
    )
    dataset = DatasetArtifact(
        name="height_cleaned",
        path=str(output_dir),
        file_count=len(list(output_dir.glob("*_forklift.csv"))),
        version=header_assigned_dataset.version,
    )
    context.add_output_metadata(
        {
            "output_dir": MetadataValue.path(str(output_dir)),
            "file_count": dataset.file_count,
        }
    )
    return dataset


@asset(
    group_name="processing",
    io_manager_key="artifact_io_manager",
    config_schema=CarryingConfig.to_config_schema(),
)
def carrying_dataset(
    context,
    cleaned_height_dataset: DatasetArtifact,
) -> DatasetArtifact:
    config = _load_config(context, CarryingConfig)
    source_dir = Path(cleaned_height_dataset.path)
    csv_files = find_forklift_files(source_dir)
    output_dir = paths.artifact_run_dir("carrying", context.run_id)
    for csv_path in csv_files:
        annotate_file(
            csv_path,
            output_dir,
            min_duration=config.min_duration,
            fill_gap=config.fill_gap,
            height_threshold=config.height_threshold,
            load_threshold=config.load_threshold,
            max_speed=config.max_speed,
            allow_off_duty=config.allow_off_duty,
        )
    dataset = DatasetArtifact(
        name="carrying",
        path=str(output_dir),
        file_count=len(csv_files),
        version=cleaned_height_dataset.version,
    )
    context.add_output_metadata(
        {
            "output_dir": MetadataValue.path(str(output_dir)),
            "file_count": dataset.file_count,
        }
    )
    return dataset


@asset(
    group_name="ml",
    io_manager_key="dataframe_io_manager",
    config_schema=TrainingDatasetConfig.to_config_schema(),
)
def training_dataset(
    context,
    carrying_dataset: DatasetArtifact,
) -> pd.DataFrame:
    config = _load_config(context, TrainingDatasetConfig)
    dataset_dir = Path(carrying_dataset.path)
    csv_files = sorted(dataset_dir.glob("*_forklift*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No forklift CSVs with Carrying feature found in {dataset_dir}"
        )

    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if "Carrying" not in df:
            continue
        df["source_file"] = csv_path.name
        frames.append(df)

    if not frames:
        raise ValueError("None of the annotated CSVs contained a Carrying column")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Carrying"])
    numeric_cols = ["Height", "Load", "Speed", "OnDuty"]
    for col in numeric_cols:
        if col not in combined:
            combined[col] = 0.0
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)

    combined["target"] = combined["Carrying"].astype(int)
    feature_cols = numeric_cols
    training_df = combined[feature_cols + ["target"]]

    if config.max_rows and len(training_df) > config.max_rows:
        training_df = training_df.sample(config.max_rows, random_state=42)

    context.add_output_metadata(
        {
            "rows": len(training_df),
            "features": MetadataValue.json(feature_cols),
        }
    )
    return training_df
