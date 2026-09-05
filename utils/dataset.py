"""
Dataset configuration.

Class names are read from the project's ``data.yaml`` (the same file used for
training) rather than assumed, so the dashboard always speaks the dataset's own
vocabulary. If the file is missing or malformed the app still runs and falls
back to the class names embedded in the model checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from utils.config import DATA_YAML_PATH, categorise


@dataclass
class DatasetConfig:
    """Class names and metadata parsed from data.yaml."""

    names: Dict[int, str] = field(default_factory=dict)
    path: Optional[Path] = None
    loaded: bool = False
    error: str = ""
    raw: Dict = field(default_factory=dict)

    @property
    def class_list(self) -> List[str]:
        return [self.names[i] for i in sorted(self.names)]

    @property
    def num_classes(self) -> int:
        return len(self.names)

    @property
    def weed_classes(self) -> List[str]:
        return [n for n in self.class_list if categorise(n) == "weed"]

    @property
    def crop_classes(self) -> List[str]:
        return [n for n in self.class_list if categorise(n) == "crop"]


def _normalise_names(names) -> Dict[int, str]:
    """data.yaml stores names either as a list or as an index-keyed mapping."""
    if isinstance(names, dict):
        out = {}
        for key, value in names.items():
            try:
                out[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def load_dataset_config(path: Path = DATA_YAML_PATH) -> DatasetConfig:
    """Read data.yaml. Never raises — errors are reported on the returned object."""
    config = DatasetConfig(path=path)

    if not path.is_file():
        config.error = f"data.yaml not found at {path}."
        return config

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError) as exc:
        config.error = f"data.yaml could not be parsed: {exc}"
        return config

    if not isinstance(raw, dict):
        config.error = "data.yaml does not contain a mapping at the top level."
        return config

    names = _normalise_names(raw.get("names"))
    if not names:
        config.error = "data.yaml contains no usable 'names' entry."
        config.raw = raw
        return config

    config.names = names
    config.raw = raw
    config.loaded = True
    return config


def resolve_class_names(
    dataset: DatasetConfig,
    model_names: Optional[Dict[int, str]] = None,
) -> Dict[int, str]:
    """
    Decide which class names to display.

    The dataset file wins when it agrees with the checkpoint on class count;
    otherwise the checkpoint wins, because a mismatch means the model was
    trained on a different configuration than the one on disk.
    """
    model_names = {int(k): str(v) for k, v in (model_names or {}).items()}
    if dataset.loaded and (not model_names or len(model_names) == len(dataset.names)):
        return dict(dataset.names)
    return model_names or dict(dataset.names)


def class_mismatch_warning(
    dataset: DatasetConfig,
    model_names: Optional[Dict[int, str]],
) -> str:
    """Return a human-readable warning if data.yaml and the checkpoint disagree."""
    if not dataset.loaded or not model_names:
        return ""
    if len(model_names) != len(dataset.names):
        return (
            f"data.yaml defines {len(dataset.names)} classes but the selected "
            f"checkpoint defines {len(model_names)}. The checkpoint's names are "
            "being used for this run."
        )
    differing = [
        f"{i}: '{dataset.names.get(i)}' vs '{model_names.get(i)}'"
        for i in sorted(dataset.names)
        if str(dataset.names.get(i)).lower() != str(model_names.get(i, "")).lower()
    ]
    if differing:
        return "Class names differ between data.yaml and the checkpoint — " + "; ".join(differing)
    return ""
