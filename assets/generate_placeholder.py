"""
Generate the stand-in monitoring frame used by the Live monitoring panel.

This produces a synthetic top-down view of a soybean bed so the panel has
something to show before a real capture is available. Replace
``dummy_monitoring.jpg`` with an actual frame from the FarmBot camera as soon as
you have one — detection results on a synthetic image mean nothing.

    python assets/generate_placeholder.py
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

WIDTH, HEIGHT = 1280, 854
OUTPUT = Path(__file__).resolve().parent / "dummy_monitoring.jpg"
SEED = 20250905


def soil_base(rng: random.Random) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (108, 84, 63))
    draw = ImageDraw.Draw(canvas)

    # Broad tonal variation across the bed.
    for _ in range(220):
        x, y = rng.randrange(WIDTH), rng.randrange(HEIGHT)
        radius = rng.randint(60, 220)
        shade = rng.randint(-22, 18)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(max(0, 108 + shade), max(0, 84 + shade), max(0, 63 + shade)),
        )
    canvas = canvas.filter(ImageFilter.GaussianBlur(26))

    # Clods and grit.
    draw = ImageDraw.Draw(canvas)
    for _ in range(5200):
        x, y = rng.randrange(WIDTH), rng.randrange(HEIGHT)
        radius = rng.randint(1, 6)
        shade = rng.randint(-38, 34)
        draw.ellipse(
            [x, y, x + radius, y + radius],
            fill=(
                max(0, min(255, 112 + shade)),
                max(0, min(255, 88 + shade)),
                max(0, min(255, 66 + shade)),
            ),
        )
    return canvas.filter(ImageFilter.GaussianBlur(0.6))


def leaflet(draw, cx, cy, length, angle, colour, width_ratio=0.55):
    """Draw one leaf as a rotated ellipse approximated by a polygon."""
    half_w = length * width_ratio / 2
    points = []
    for step in range(0, 21):
        t = step / 20 * math.pi * 2
        lx = math.cos(t) * length / 2
        ly = math.sin(t) * half_w
        px = cx + lx * math.cos(angle) - ly * math.sin(angle) + math.cos(angle) * length / 2
        py = cy + lx * math.sin(angle) + ly * math.cos(angle) + math.sin(angle) * length / 2
        points.append((px, py))
    draw.polygon(points, fill=colour, outline=(int(colour[0] * 0.8), int(colour[1] * 0.8), int(colour[2] * 0.8)))


def soybean(draw, cx, cy, scale, rng):
    """A trifoliate seedling: three rounded leaflets from a common centre."""
    base = (
        rng.randint(58, 82),
        rng.randint(112, 142),
        rng.randint(52, 74),
    )
    start = rng.uniform(0, math.pi * 2)
    for i in range(3):
        angle = start + i * (2 * math.pi / 3) + rng.uniform(-0.16, 0.16)
        tint = rng.randint(-12, 16)
        colour = (
            max(0, min(255, base[0] + tint)),
            max(0, min(255, base[1] + tint)),
            max(0, min(255, base[2] + tint)),
        )
        leaflet(draw, cx, cy, scale * rng.uniform(0.85, 1.1), angle, colour, 0.62)
    draw.ellipse(
        [cx - scale * 0.09, cy - scale * 0.09, cx + scale * 0.09, cy + scale * 0.09],
        fill=(base[0] - 14, base[1] - 18, base[2] - 12),
    )


def weed(draw, cx, cy, scale, rng):
    """A weed: more, narrower, unevenly spread blades in a yellower green."""
    base = (
        rng.randint(96, 128),
        rng.randint(138, 170),
        rng.randint(44, 66),
    )
    blades = rng.randint(6, 10)
    start = rng.uniform(0, math.pi * 2)
    for i in range(blades):
        angle = start + i * (2 * math.pi / blades) + rng.uniform(-0.4, 0.4)
        tint = rng.randint(-16, 20)
        colour = (
            max(0, min(255, base[0] + tint)),
            max(0, min(255, base[1] + tint)),
            max(0, min(255, base[2] + tint)),
        )
        leaflet(draw, cx, cy, scale * rng.uniform(0.6, 1.15), angle, colour, 0.2)


def build() -> Image.Image:
    rng = random.Random(SEED)
    canvas = soil_base(rng)
    plants = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(plants)

    # Three crop rows running left to right, as a FarmBot bed would be sown.
    row_ys = [HEIGHT * 0.26, HEIGHT * 0.53, HEIGHT * 0.80]
    for row_y in row_ys:
        x = rng.uniform(60, 130)
        while x < WIDTH - 40:
            jitter_y = row_y + rng.uniform(-16, 16)
            soybean(draw, x, jitter_y, rng.uniform(74, 104), rng)
            x += rng.uniform(112, 156)

    # Weeds scattered between and within the rows.
    for _ in range(17):
        wx = rng.uniform(50, WIDTH - 50)
        wy = rng.uniform(50, HEIGHT - 50)
        weed(draw, wx, wy, rng.uniform(42, 74), rng)

    plants = plants.filter(ImageFilter.GaussianBlur(0.9))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), plants).convert("RGB")

    # A gentle vignette, as a fixed overhead camera would produce.
    vignette = Image.new("L", (WIDTH, HEIGHT), 0)
    vdraw = ImageDraw.Draw(vignette)
    vdraw.ellipse(
        [-WIDTH * 0.18, -HEIGHT * 0.18, WIDTH * 1.18, HEIGHT * 1.18], fill=255
    )
    vignette = vignette.filter(ImageFilter.GaussianBlur(160))
    dark = Image.new("RGB", (WIDTH, HEIGHT), (52, 42, 32))
    canvas = Image.composite(canvas, dark, vignette)
    return canvas


if __name__ == "__main__":
    image = build()
    image.save(OUTPUT, "JPEG", quality=88, optimize=True)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB)")
