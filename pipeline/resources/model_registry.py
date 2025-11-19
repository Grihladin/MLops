"""Simple model registry resource that persists pickled estimators."""

from __future__ import annotations

import pickle
from pathlib import Path

from dagster import ConfigurableResource, get_dagster_logger

from ..utils import paths


class LocalModelRegistry(ConfigurableResource):
    base_dir: str | None = None

    @property
    def root(self) -> Path:
        if self.base_dir:
            return paths.ensure_dir(Path(self.base_dir).expanduser())
        return paths.models_root()

    def save_model(self, model, version: str) -> Path:
        destination = self.root / f"model_{version}.pkl"
        with open(destination, "wb") as buffer:
            pickle.dump(model, buffer)
        get_dagster_logger().info("Stored model at %s", destination)
        return destination
