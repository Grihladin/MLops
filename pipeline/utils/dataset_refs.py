"""Typed value objects shared between Dagster assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class DatasetArtifact:
    """Describes a directory of materialized files."""

    name: str
    path: str
    file_count: int
    version: str | None = None

    def as_path(self) -> Path:
        return Path(self.path)

    def metadata(self) -> Mapping[str, str | int | None]:
        return {
            "name": self.name,
            "path": self.path,
            "file_count": self.file_count,
            "version": self.version,
        }


@dataclass(frozen=True)
class ModelArtifact:
    """Represents a trained model persisted to disk."""

    model_path: str
    metrics: Mapping[str, float]
    feature_names: list[str]
    version: str

    def as_path(self) -> Path:
        return Path(self.model_path)

    def metadata(self) -> Mapping[str, str | float | list[str]]:
        return {
            "model_path": self.model_path,
            "version": self.version,
            "metric_keys": list(self.metrics.keys()),
            "feature_count": len(self.feature_names),
        }
