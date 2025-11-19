"""ML-centric Dagster assets."""

from __future__ import annotations

import json
from typing import Literal, Mapping, Type, TypeVar

import pandas as pd
from dagster import AssetExecutionContext, Config, MetadataValue, asset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SKPipeline

from ..resources import LocalModelRegistry
from ..utils import paths
from ..utils.dataset_refs import ModelArtifact


class ModelTrainingConfig(Config):
    test_size: float = 0.2
    random_state: int = 7
    penalty: Literal["l2"] = "l2"
    C: float = 1.0
    max_iter: int = 500


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
    group_name="ml",
    io_manager_key="artifact_io_manager",
    config_schema=ModelTrainingConfig.to_config_schema(),
)
def trained_model(
    context,
    training_dataset: pd.DataFrame,
    model_registry: LocalModelRegistry,
) -> ModelArtifact:
    config = _load_config(context, ModelTrainingConfig)
    if training_dataset.empty:
        raise ValueError("Training dataset is empty; nothing to model")

    features = training_dataset.drop(columns=["target"])
    target = training_dataset["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )

    pipeline = SKPipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty=config.penalty,
                    C=config.C,
                    max_iter=config.max_iter,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    metrics: Mapping[str, float] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "test_size": len(y_test),
    }

    version = context.run_id
    model_path = model_registry.save_model(pipeline, version)
    artifact = ModelArtifact(
        model_path=str(model_path),
        metrics=metrics,
        feature_names=list(features.columns),
        version=version,
    )
    context.add_output_metadata(
        {
            "model_path": MetadataValue.path(str(model_path)),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
    )
    return artifact


@asset(
    group_name="ml",
    io_manager_key="artifact_io_manager",
)
def model_performance_report(
    context,
    trained_model: ModelArtifact,
) -> str:
    metrics_dir = paths.metrics_root()
    destination = metrics_dir / f"model_metrics_{trained_model.version}.json"
    payload = {
        "model_path": trained_model.model_path,
        "metrics": trained_model.metrics,
        "features": trained_model.feature_names,
        "version": trained_model.version,
    }
    destination.write_text(json.dumps(payload, indent=2))
    context.add_output_metadata({"report_path": MetadataValue.path(str(destination))})
    return str(destination)
