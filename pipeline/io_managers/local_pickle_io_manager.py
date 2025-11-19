"""Pickle-based IO manager for structured Python objects."""

from __future__ import annotations

import pickle
from pathlib import Path

from dagster import ConfigurableIOManager, InputContext, OutputContext

from ..utils import paths


class LocalArtifactIOManager(ConfigurableIOManager):
    base_dir: str | None = None

    def _root(self) -> Path:
        if self.base_dir:
            return paths.ensure_dir(Path(self.base_dir).expanduser())
        return paths.ensure_dir(paths.artifacts_root() / "io_manager")

    def _path_for_context(self, context: OutputContext | InputContext) -> Path:
        parts = context.asset_key.path if context.has_asset_key else [context.step_key]
        asset_name = "__".join(parts)
        return self._root() / f"{asset_name}.pickle"

    def handle_output(self, context: OutputContext, obj) -> None:  # noqa: ANN001 - Dagster API
        target = self._path_for_context(context)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as buffer:
            pickle.dump(obj, buffer)
        context.log.debug("Serialized %s to %s", context.asset_key, target)

    def load_input(self, context: InputContext):  # noqa: ANN001 - Dagster API
        source = self._path_for_context(context)
        with open(source, "rb") as buffer:
            return pickle.load(buffer)
