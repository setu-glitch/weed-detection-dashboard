"""
Central configuration for the weed-detection dashboard.

Everything that a researcher may want to change without touching UI code lives
here: filesystem paths, the model registry, class-to-category mapping rules and
the visual design tokens.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories can be overridden with environment variables so the same code base
# runs locally, in Docker and on a free-tier host without edits.
MODELS_DIR = Path(os.environ.get("WEED_MODELS_DIR", PROJECT_ROOT / "models"))
ASSETS_DIR = Path(os.environ.get("WEED_ASSETS_DIR", PROJECT_ROOT / "assets"))
DATA_YAML_PATH = Path(os.environ.get("WEED_DATA_YAML", PROJECT_ROOT / "data.yaml"))
BENCHMARKS_PATH = Path(os.environ.get("WEED_BENCHMARKS", PROJECT_ROOT / "benchmarks.json"))

MONITORING_FRAME = ASSETS_DIR / "dummy_monitoring.jpg"

# --------------------------------------------------------------------------
# Inference defaults (tuned for CPU-only free-tier hosts)
# --------------------------------------------------------------------------

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
# Uploads larger than this on the long edge are downscaled before inference.
MAX_INPUT_EDGE = 1600
MAX_UPLOAD_MB = 20


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """A single selectable detection model."""

    key: str
    label: str
    filename: str
    generation: str
    released: str
    architecture: str
    note: str = ""

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.filename

    @property
    def is_available(self) -> bool:
        return self.path.is_file()


# Adding a sixth model means adding one entry here. Nothing else changes.
MODEL_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        key="yolov8n",
        label="YOLOv8n",
        filename="yolov8n.pt",
        generation="v8",
        released="2023",
        architecture="CSPDarknet backbone, anchor-free decoupled head",
        note="Baseline of the lightweight generations compared in the study.",
    ),
    ModelSpec(
        key="yolov9n",
        label="YOLOv9n",
        filename="yolov9n.pt",
        generation="v9",
        released="2024",
        architecture="GELAN backbone with programmable gradient information",
        note="Adds gradient-path reversibility to reduce information loss.",
    ),
    ModelSpec(
        key="yolov10n",
        label="YOLOv10n",
        filename="yolov10n.pt",
        generation="v10",
        released="2024",
        architecture="NMS-free training with consistent dual assignments",
        note="Removes non-maximum suppression from the inference path.",
    ),
    ModelSpec(
        key="yolov11n",
        label="YOLOv11n",
        filename="yolov11n.pt",
        generation="v11",
        released="2024",
        architecture="C3k2 blocks with C2PSA spatial attention",
        note="Strongest overall detection accuracy in this study.",
    ),
    ModelSpec(
        key="yolov12n",
        label="YOLOv12n",
        filename="yolov12n.pt",
        generation="v12",
        released="2025",
        architecture="Attention-centric design with area attention and R-ELAN",
        note="Accuracy comparable to YOLOv11n on this dataset.",
    ),
]

MODELS_BY_KEY: Dict[str, ModelSpec] = {spec.key: spec for spec in MODEL_REGISTRY}

DEFAULT_MODEL_KEY = "yolov11n"


# --------------------------------------------------------------------------
# Class → category mapping
# --------------------------------------------------------------------------
#
# The dataset's own class names come from data.yaml / the model checkpoint. The
# dashboard only needs to know which of those names denote weeds; everything
# else is treated as crop. Keep the keyword lists lowercase.

WEED_KEYWORDS = ("weed", "unkraut", "grass", "thistle", "amaranth")
CROP_KEYWORDS = ("soy", "soja", "soybean", "plant", "crop", "seedling")

CATEGORY_WEED = "weed"
CATEGORY_CROP = "crop"

CATEGORY_LABELS = {
    CATEGORY_WEED: "Weed",
    CATEGORY_CROP: "Soybean plant",
}


def categorise(class_name: str) -> str:
    """Map a dataset class name onto either the weed or the crop category."""
    name = (class_name or "").strip().lower()
    for keyword in WEED_KEYWORDS:
        if keyword in name:
            return CATEGORY_WEED
    for keyword in CROP_KEYWORDS:
        if keyword in name:
            return CATEGORY_CROP
    # Unknown names are treated as crop so that an unrecognised label never
    # silently inflates the weed count that drives intervention decisions.
    return CATEGORY_CROP


# --------------------------------------------------------------------------
# Research benchmarks (read from disk, never invented at runtime)
# --------------------------------------------------------------------------

@dataclass
class Benchmark:
    model_key: str
    precision: Optional[float] = None
    recall: Optional[float] = None
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    source: str = ""

    @property
    def has_values(self) -> bool:
        return any(
            v is not None
            for v in (self.precision, self.recall, self.map50, self.map50_95)
        )


@dataclass
class BenchmarkTable:
    """Published results of the underlying study, loaded from benchmarks.json."""

    per_model: Dict[str, Benchmark] = field(default_factory=dict)
    study_notes: List[str] = field(default_factory=list)
    citation: str = ""
    loaded: bool = False
    error: str = ""

    def get(self, model_key: str) -> Benchmark:
        return self.per_model.get(model_key, Benchmark(model_key=model_key))


def load_benchmarks(path: Path = BENCHMARKS_PATH) -> BenchmarkTable:
    """Load published benchmark values. Missing values stay missing."""
    table = BenchmarkTable()
    if not path.is_file():
        table.error = f"No benchmark file found at {path.name}."
        return table
    try:
        raw: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        table.error = f"Could not read {path.name}: {exc}"
        return table

    for key, values in (raw.get("models") or {}).items():
        table.per_model[key] = Benchmark(
            model_key=key,
            precision=values.get("precision"),
            recall=values.get("recall"),
            map50=values.get("map50"),
            map50_95=values.get("map50_95"),
            source=values.get("source", ""),
        )
    table.study_notes = list(raw.get("study_notes") or [])
    table.citation = raw.get("citation", "")
    table.loaded = True
    return table


# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
#
# One palette, used by both the CSS in the app shell and the bounding-box
# renderer, so an annotated image always matches the interface around it.

PALETTE = {
    # Surfaces
    "canvas": "#EEF1EA",        # pale field green-grey behind everything
    "surface": "#FFFFFF",
    "line": "#D8DED4",
    "paper": "#EEF1EA",         # alias used by the image renderer
    "border": "#D8DED4",
    "border_strong": "#B9C9BC",

    # Text
    "ink": "#0B1F14",           # deep canopy near-black
    "muted": "#4C5C51",
    "faint": "#7C8B80",

    # Chrome
    "field": "#14432A",         # sidebar, masthead, primary buttons
    "field_deep": "#0C2C1B",

    # Classes: green grows, orange gets pulled
    "crop": "#2E8B4E",
    "crop_bright": "#46B96A",
    "crop_soft": "#E7F2EA",
    "weed": "#E2571F",
    "weed_bright": "#FF8A4C",   # on dark backgrounds only
    "weed_soft": "#FDEDE4",

    "institution": "#3070B3",   # TUM blue, reserved for published figures
    "signal": "#46B96A",
    "warning": "#C9741A",
}

CATEGORY_COLORS = {
    CATEGORY_WEED: PALETTE["weed"],
    CATEGORY_CROP: PALETTE["crop"],
}

APP_TITLE = "AI-Powered Precision Weed Management"
APP_SUBTITLE = "Computer vision for autonomous soybean weed detection"
INSTITUTION = "Technical University of Munich"
