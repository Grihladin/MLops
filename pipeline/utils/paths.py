"""Centralized helpers for filesystem locations used by the pipeline."""

from __future__ import annotations

from pathlib import Path

# ``pipeline/utils`` → ``pipeline`` → repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACTS_ROOT = _PIPELINE_ROOT / "artifacts"
_MODELS_ROOT = _PIPELINE_ROOT / "models"
_METRICS_ROOT = _PIPELINE_ROOT / "metrics"


def repo_root() -> Path:
    return _REPO_ROOT


def pipeline_root() -> Path:
    return _PIPELINE_ROOT


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifacts_root() -> Path:
    return ensure_dir(_ARTIFACTS_ROOT)


def artifact_run_dir(stage: str, run_id: str) -> Path:
    return ensure_dir(_ARTIFACTS_ROOT / stage / run_id)


def models_root() -> Path:
    return ensure_dir(_MODELS_ROOT)


def metrics_root() -> Path:
    return ensure_dir(_METRICS_ROOT)
