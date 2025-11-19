"""Persist pandas DataFrames as Parquet files between asset runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from dagster import ConfigurableIOManager, InputContext, OutputContext

from ..utils import paths


class DataFrameParquetIOManager(ConfigurableIOManager):
    base_dir: str | None = None

    def _root(self) -> Path:
        if self.base_dir:
            return paths.ensure_dir(Path(self.base_dir).expanduser())
        return paths.ensure_dir(paths.artifacts_root() / "tables")

    def _path_for_context(self, context: OutputContext | InputContext) -> Path:
        parts = context.asset_key.path if context.has_asset_key else [context.step_key]
        asset_name = "__".join(parts)
        return self._root() / f"{asset_name}.parquet"

    def handle_output(self, context: OutputContext, obj) -> None:  # noqa: ANN001 - Dagster API
        if obj is None:
            return
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(
                f"DataFrameParquetIOManager expected a pandas DataFrame, got {type(obj)}"
            )
        target = self._path_for_context(context)
        target.parent.mkdir(parents=True, exist_ok=True)
        obj.to_parquet(target, index=False)
        context.add_output_metadata({"path": target.as_posix(), "rows": len(obj)})

    def load_input(self, context: InputContext):  # noqa: ANN001 - Dagster API
        source = self._path_for_context(context)
        return pd.read_parquet(source)
