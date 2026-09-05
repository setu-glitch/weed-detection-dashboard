"""
Inference pipeline.

Kept free of Streamlit imports so it can be reused by a batch script, a REST
endpoint, or the future FarmBot control loop without dragging in the UI.

Every number produced here comes from an actual model run. Nothing in this
module carries published benchmark values — those live in benchmarks.json and
are only ever displayed alongside a clear "published study" label.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageOps

from utils.config import (
    CATEGORY_CROP,
    CATEGORY_WEED,
    DEFAULT_CONFIDENCE,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    MAX_INPUT_EDGE,
    categorise,
)


class InferenceError(RuntimeError):
    """Raised when an image cannot be prepared or a model run fails."""


@dataclass(frozen=True)
class Detection:
    """One detected object in image pixel coordinates."""

    class_id: int
    class_name: str
    category: str
    confidence: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def is_weed(self) -> bool:
        return self.category == CATEGORY_WEED


@dataclass
class DetectionResult:
    """Everything the UI needs about a single completed inference run."""

    detections: List[Detection]
    model_key: str
    model_label: str
    image_size: Tuple[int, int]
    original_size: Tuple[int, int]
    inference_ms: float
    total_ms: float
    confidence_threshold: float
    iou_threshold: float
    imgsz: int
    source_name: str = ""
    was_resized: bool = False
    timestamp: float = field(default_factory=time.time)

    # -- derived statistics -------------------------------------------------

    @property
    def weeds(self) -> List[Detection]:
        return [d for d in self.detections if d.category == CATEGORY_WEED]

    @property
    def crops(self) -> List[Detection]:
        return [d for d in self.detections if d.category == CATEGORY_CROP]

    @property
    def total(self) -> int:
        return len(self.detections)

    @property
    def weed_count(self) -> int:
        return len(self.weeds)

    @property
    def crop_count(self) -> int:
        return len(self.crops)

    @property
    def weed_share(self) -> float:
        """Share of detected objects classified as weeds, in percent."""
        return (self.weed_count / self.total * 100.0) if self.total else 0.0

    @property
    def mean_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)

    @property
    def mean_weed_confidence(self) -> float:
        if not self.weeds:
            return 0.0
        return sum(d.confidence for d in self.weeds) / len(self.weeds)

    @property
    def min_confidence(self) -> float:
        return min((d.confidence for d in self.detections), default=0.0)

    @property
    def weed_area_share(self) -> float:
        """Share of detected object area occupied by weeds, in percent."""
        total_area = sum(d.area for d in self.detections)
        if total_area <= 0:
            return 0.0
        return sum(d.area for d in self.weeds) / total_area * 100.0

    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def as_summary_rows(self) -> List[Tuple[str, str]]:
        """Rows for the detection-summary table, formatted for display."""
        return [
            ("Model", self.model_label),
            ("Objects detected", f"{self.total}"),
            ("Weeds", f"{self.weed_count}"),
            ("Soybean plants", f"{self.crop_count}"),
            ("Weed share", f"{self.weed_share:.1f}%"),
            ("Mean confidence", f"{self.mean_confidence * 100:.1f}%"),
            ("Lowest confidence", f"{self.min_confidence * 100:.1f}%"),
            ("Inference time", f"{self.inference_ms:.0f} ms"),
            ("Confidence threshold", f"{self.confidence_threshold:.2f}"),
            ("Image size", f"{self.image_size[0]} × {self.image_size[1]} px"),
        ]


# --------------------------------------------------------------------------
# Image preparation
# --------------------------------------------------------------------------

def prepare_image(image: Image.Image, max_edge: int = MAX_INPUT_EDGE) -> Tuple[Image.Image, bool]:
    """
    Normalise an uploaded image for inference.

    Applies EXIF orientation, converts to RGB and downscales very large photos
    on the long edge. Returns the prepared image and whether it was resized.
    """
    if image is None:
        raise InferenceError("No image was provided.")

    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    if width < 8 or height < 8:
        raise InferenceError("The image is too small to run detection on.")

    longest = max(width, height)
    if longest > max_edge:
        scale = max_edge / float(longest)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        image = image.resize(new_size, Image.LANCZOS)
        return image, True

    return image, False


def open_image(file_like) -> Image.Image:
    """Open an uploaded file, raising a readable error for unsupported input."""
    try:
        image = Image.open(file_like)
        image.load()
    except Exception as exc:
        raise InferenceError(
            "That file could not be read as an image. Upload a JPG, PNG, BMP or WEBP file."
        ) from exc
    return image


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def _extract_detections(result, class_names: Dict[int, str]) -> List[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    try:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
    except Exception as exc:  # pragma: no cover - defensive
        raise InferenceError(f"The model returned results in an unexpected format: {exc}") from exc

    detections: List[Detection] = []
    for box, conf, class_id in zip(xyxy, confs, classes):
        class_id = int(class_id)
        name = class_names.get(class_id, f"class_{class_id}")
        detections.append(
            Detection(
                class_id=class_id,
                class_name=name,
                category=categorise(name),
                confidence=float(conf),
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
        )
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def run_detection(
    model,
    image: Image.Image,
    *,
    model_key: str,
    model_label: str,
    class_names: Optional[Dict[int, str]] = None,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    source_name: str = "",
    max_edge: int = MAX_INPUT_EDGE,
) -> Tuple[DetectionResult, Image.Image]:
    """
    Run one inference pass and return the measured result plus the exact image
    the model saw (so annotations always align with the boxes).
    """
    original_size = image.size
    prepared, was_resized = prepare_image(image, max_edge=max_edge)

    started = time.perf_counter()
    try:
        raw = model.predict(
            source=prepared,
            conf=float(confidence),
            iou=float(iou),
            imgsz=int(imgsz),
            device="cpu",
            verbose=False,
        )
    except Exception as exc:
        raise InferenceError(f"Detection failed: {exc}") from exc
    total_ms = (time.perf_counter() - started) * 1000.0

    if not raw:
        raise InferenceError("The model returned no result for this image.")
    result = raw[0]

    names = class_names or {}
    if not names:
        model_names = getattr(result, "names", None) or getattr(model, "names", None) or {}
        if isinstance(model_names, dict):
            names = {int(k): str(v) for k, v in model_names.items()}
        elif isinstance(model_names, (list, tuple)):
            names = {i: str(v) for i, v in enumerate(model_names)}

    detections = _extract_detections(result, names)

    # Ultralytics reports the pure forward-pass time; fall back to wall clock.
    speed = getattr(result, "speed", None) or {}
    inference_ms = float(speed.get("inference") or 0.0)
    if inference_ms <= 0:
        inference_ms = total_ms

    detection_result = DetectionResult(
        detections=detections,
        model_key=model_key,
        model_label=model_label,
        image_size=prepared.size,
        original_size=original_size,
        inference_ms=inference_ms,
        total_ms=total_ms,
        confidence_threshold=float(confidence),
        iou_threshold=float(iou),
        imgsz=int(imgsz),
        source_name=source_name,
        was_resized=was_resized,
    )
    return detection_result, prepared


def detections_to_rows(result: DetectionResult, limit: Optional[int] = None) -> List[Dict[str, object]]:
    """Flatten detections into table rows for display or CSV export."""
    rows = []
    for index, det in enumerate(result.detections[: limit or len(result.detections)], start=1):
        cx, cy = det.centroid
        rows.append(
            {
                "#": index,
                "Class": det.class_name,
                "Category": "Weed" if det.is_weed else "Soybean plant",
                "Confidence": round(det.confidence * 100, 1),
                "Centre x (px)": round(cx, 1),
                "Centre y (px)": round(cy, 1),
                "Width (px)": round(det.box[2] - det.box[0], 1),
                "Height (px)": round(det.box[3] - det.box[1], 1),
            }
        )
    return rows
