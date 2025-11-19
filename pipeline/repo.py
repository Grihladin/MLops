"""Dagster repository definition for the forklift pipeline."""

from __future__ import annotations

from dagster import AssetSelection, Definitions, define_asset_job

from .assets.data_assets import (
    carrying_dataset,
    cleaned_height_dataset,
    header_assigned_dataset,
    raw_forklift_snapshot,
    training_dataset,
)
from .assets.model_assets import model_performance_report, trained_model
from .io_managers.dataframe_io_manager import DataFrameParquetIOManager
from .io_managers.local_pickle_io_manager import LocalArtifactIOManager
from .resources.lakefs import LakeFSResource
from .resources.model_registry import LocalModelRegistry
from .sensors.lakefs_sensor import build_lakefs_sensor

ALL_ASSETS = [
    raw_forklift_snapshot,
    header_assigned_dataset,
    cleaned_height_dataset,
    carrying_dataset,
    training_dataset,
    trained_model,
    model_performance_report,
]

forklift_job = define_asset_job(
    "forklift_training_job",
    selection=AssetSelection.all(),
)

lakefs_sensor = build_lakefs_sensor(forklift_job)


defs = Definitions(
    assets=ALL_ASSETS,
    jobs=[forklift_job],
    resources={
        "lakefs": LakeFSResource(),
        "model_registry": LocalModelRegistry(),
        "artifact_io_manager": LocalArtifactIOManager(),
        "dataframe_io_manager": DataFrameParquetIOManager(),
    },
    sensors=[lakefs_sensor],
)
