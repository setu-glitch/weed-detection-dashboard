"""
Model discovery and loading.

Models are loaded lazily: a checkpoint only reaches memory when a user actually
runs it, and Streamlit's resource cache keeps at most one instance per file so
repeated inferences on a free-tier host do not re-read the weights.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Ultralytics writes a settings file on import. Managed hosts often give the
# process a read-only or absent HOME, so point it somewhere writable first.
os.environ.setdefault("YOLO_CONFIG_DIR", os.environ.get("TMPDIR", "/tmp"))

from utils.config import MODEL_REGISTRY, MODELS_BY_KEY, ModelSpec

try:  # Streamlit is optional so the module stays testable outside the app.
    import streamlit as st

    _cache_resource = st.cache_resource
except Exception:  # pragma: no cover - exercised only outside Streamlit
    def _cache_resource(func=None, **_kwargs):
        def decorator(fn):
            return fn

        return decorator(func) if func else decorator


class ModelLoadError(RuntimeError):
    """Raised when a checkpoint cannot be turned into a usable model."""


@dataclass
class ModelStatus:
    spec: ModelSpec
    available: bool
    size_mb: Optional[float] = None
    reason: str = ""

    @property
    def label(self) -> str:
        return self.spec.label


def inspect_models() -> List[ModelStatus]:
    """Report which registered checkpoints are present on disk."""
    statuses: List[ModelStatus] = []
    for spec in MODEL_REGISTRY:
        path = spec.path
        if path.is_file():
            try:
                size_mb = round(path.stat().st_size / (1024 * 1024), 1)
            except OSError:
                size_mb = None
            statuses.append(ModelStatus(spec=spec, available=True, size_mb=size_mb))
        else:
            statuses.append(
                ModelStatus(
                    spec=spec,
                    available=False,
                    reason=f"Place {spec.filename} in the models directory to enable it.",
                )
            )
    return statuses


def available_model_keys() -> List[str]:
    return [s.spec.key for s in inspect_models() if s.available]


def get_spec(model_key: str) -> ModelSpec:
    if model_key not in MODELS_BY_KEY:
        raise ModelLoadError(f"'{model_key}' is not a registered model.")
    return MODELS_BY_KEY[model_key]


def ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return True


@_cache_resource(show_spinner=False)
def _load_from_path(path_str: str, fingerprint: str):
    """
    Load one checkpoint. Cached on (path, fingerprint) so replacing a file on
    disk transparently invalidates the cached instance.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ModelLoadError(
            "The 'ultralytics' package is not installed. Run "
            "`pip install -r requirements.txt` and restart the app."
        ) from exc

    try:
        model = YOLO(path_str)
    except Exception as exc:
        raise ModelLoadError(f"{Path(path_str).name} could not be loaded: {exc}") from exc

    # Force CPU-friendly evaluation. Ultralytics picks a device per call, but
    # fusing here keeps the first real inference from paying the cost.
    try:
        model.fuse()
    except Exception:
        pass
    return model


def load_model(model_key: str):
    """Return a ready-to-use model for the given registry key."""
    spec = get_spec(model_key)
    path = spec.path
    if not path.is_file():
        raise ModelLoadError(
            f"{spec.label} is not installed. Expected a checkpoint at {path}."
        )
    try:
        stat = path.stat()
        fingerprint = f"{stat.st_size}-{int(stat.st_mtime)}"
    except OSError as exc:
        raise ModelLoadError(f"{spec.filename} is unreadable: {exc}") from exc
    return _load_from_path(str(path), fingerprint)


def model_class_names(model) -> Dict[int, str]:
    """Extract the class-index → name mapping stored in a checkpoint."""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}
