"""Resource exports for convenience."""

from .lakefs import LakeFSResource
from .model_registry import LocalModelRegistry

__all__ = ["LakeFSResource", "LocalModelRegistry"]
