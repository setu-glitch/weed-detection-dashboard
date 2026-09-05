"""
Detection rendering.

Draws bounding boxes with PIL rather than the Ultralytics plotter so the
annotated image uses exactly the same two class colours as the surrounding
interface, and so the renderer works without OpenCV on a slim host.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from utils.config import CATEGORY_COLORS, CATEGORY_WEED, PALETTE
from utils.detection import Detection

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    try:  # Pillow >= 10.1 can scale the built-in bitmap font.
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - older Pillow
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except AttributeError:  # pragma: no cover - very old Pillow
        return draw.textsize(text, font=font)


def _scale_for(image: Image.Image) -> Tuple[int, int, int]:
    """Line width, maximum font size and label padding for an image."""
    longest = max(image.size)
    line_width = max(2, round(longest / 560))
    font_size = min(26, max(11, round(longest / 62)))
    padding = max(2, round(font_size / 3.5))
    return line_width, font_size, padding


def _fit_label(det: Detection, box_width: float, max_size: int, show_confidence: bool):
    """
    Choose the label text and type size for one box.

    Field images hold dozens of small objects, so a label is sized to the box it
    belongs to. When even the short form would not fit, the box is left bare and
    the class colour carries the meaning.
    """
    full = f"{det.class_name} {det.confidence:.2f}" if show_confidence else det.class_name
    short = f"{det.confidence:.2f}" if show_confidence else det.class_name[:1].upper()

    for text in (full, short):
        # Roughly 0.58 em per character in a bold grotesque.
        size = int(min(max_size, box_width / max(1, len(text) * 0.58)))
        if size >= 10:
            return text, size
    return "", 0


def annotate(
    image: Image.Image,
    detections: Sequence[Detection],
    *,
    show_confidence: bool = True,
    show_labels: bool = True,
    highlight: str = "",
    mark_weed_centres: bool = False,
) -> Image.Image:
    """
    Return a new annotated copy of ``image``.

    ``highlight`` dims every detection outside the given category, which the
    autonomous-weeding view uses to isolate intervention targets.
    """
    canvas = image.convert("RGB").copy()
    if not detections:
        return canvas

    draw = ImageDraw.Draw(canvas, "RGBA")
    line_width, max_font_size, padding = _scale_for(canvas)
    width, height = canvas.size
    font_cache = {}

    # Draw lower-confidence boxes first so the strongest detections sit on top.
    ordered: List[Detection] = sorted(detections, key=lambda d: d.confidence)

    for det in ordered:
        colour = CATEGORY_COLORS.get(det.category, PALETTE["institution"])
        dimmed = bool(highlight) and det.category != highlight

        x1, y1, x2, y2 = det.box
        x1 = max(0.0, min(x1, width - 1))
        y1 = max(0.0, min(y1, height - 1))
        x2 = max(0.0, min(x2, width - 1))
        y2 = max(0.0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        outline = colour if not dimmed else _with_alpha(colour, 90)
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=line_width)

        if not dimmed:
            # A faint fill keeps boxes readable against busy soil textures.
            draw.rectangle([x1, y1, x2, y2], fill=_with_alpha(colour, 26))

        if mark_weed_centres and det.category == CATEGORY_WEED:
            cx, cy = det.centroid
            radius = max(3, line_width * 2)
            draw.line([cx - radius * 2, cy, cx + radius * 2, cy], fill=colour, width=line_width)
            draw.line([cx, cy - radius * 2, cx, cy + radius * 2], fill=colour, width=line_width)

        if not show_labels or dimmed:
            continue

        label, font_size = _fit_label(det, x2 - x1, max_font_size, show_confidence)
        if not label:
            continue
        font = font_cache.get(font_size)
        if font is None:
            font = font_cache[font_size] = _load_font(font_size)

        text_w, text_h = _text_size(draw, label, font)
        chip_w = text_w + padding * 2
        chip_h = text_h + padding * 2

        # Prefer the label above the box; drop it inside when there is no room.
        chip_x = min(x1, width - chip_w)
        chip_x = max(0, chip_x)
        chip_y = y1 - chip_h
        if chip_y < 0:
            chip_y = min(y1, height - chip_h)

        draw.rectangle([chip_x, chip_y, chip_x + chip_w, chip_y + chip_h], fill=colour)
        draw.text((chip_x + padding, chip_y + padding), label, fill="#FFFFFF", font=font)

    return canvas


def _with_alpha(hex_colour: str, alpha: int) -> Tuple[int, int, int, int]:
    hex_colour = hex_colour.lstrip("#")
    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (0, 2, 4))
    return (r, g, b, alpha)


def density_overlay(
    image: Image.Image,
    detections: Iterable[Detection],
    grid: int = 6,
    category: str = CATEGORY_WEED,
) -> Image.Image:
    """
    Shade a coarse grid by how many objects of ``category`` fall in each cell.

    A first, image-level step towards the field-level weed-density maps listed
    as future work; it reports only what the current detection found.
    """
    canvas = image.convert("RGB").copy()
    width, height = canvas.size
    cell_w, cell_h = width / grid, height / grid

    counts = [[0 for _ in range(grid)] for _ in range(grid)]
    for det in detections:
        if det.category != category:
            continue
        cx, cy = det.centroid
        col = min(grid - 1, max(0, int(cx // cell_w)))
        row = min(grid - 1, max(0, int(cy // cell_h)))
        counts[row][col] += 1

    peak = max((c for row in counts for c in row), default=0)
    if peak == 0:
        return canvas

    draw = ImageDraw.Draw(canvas, "RGBA")
    colour = CATEGORY_COLORS.get(category, PALETTE["weed"])
    for row in range(grid):
        for col in range(grid):
            if counts[row][col] == 0:
                continue
            alpha = int(40 + 130 * (counts[row][col] / peak))
            draw.rectangle(
                [col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h],
                fill=_with_alpha(colour, alpha),
            )
    return canvas


def placeholder_frame(
    size: Tuple[int, int] = (960, 640),
    message: str = "No monitoring frame available",
) -> Image.Image:
    """A neutral stand-in shown when the monitoring asset is missing."""
    canvas = Image.new("RGB", size, PALETTE["paper"])
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(14, size[0] // 40))
    text_w, text_h = _text_size(draw, message, font)
    draw.rectangle([1, 1, size[0] - 2, size[1] - 2], outline=PALETTE["border_strong"], width=2)
    draw.text(
        ((size[0] - text_w) / 2, (size[1] - text_h) / 2),
        message,
        fill=PALETTE["muted"],
        font=font,
    )
    return canvas


def to_png_bytes(image: Image.Image) -> bytes:
    """Encode an image for the download button."""
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
