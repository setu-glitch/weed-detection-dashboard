"""
FarmBot integration boundary.

This module defines the seam between detection and actuation. It deliberately
does not talk to hardware: every method that would move a physical machine
raises :class:`FarmBotNotConnected` until a real client is implemented, so the
dashboard can never imply that weeding took place when it did not.

To go live, implement :class:`FarmBotClient` against the FarmBot Python API and
return an instance from :func:`get_client`. Nothing in the UI needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

from utils.detection import Detection, DetectionResult


class FarmBotNotConnected(RuntimeError):
    """Raised when an actuator command is issued without a connected device."""


@dataclass(frozen=True)
class WeedTarget:
    """A weed to be actioned, in image space and in normalised coordinates."""

    index: int
    confidence: float
    pixel_x: float
    pixel_y: float
    norm_x: float
    norm_y: float
    width_px: float
    height_px: float

    @property
    def label(self) -> str:
        return f"W{self.index:02d}"


@dataclass(frozen=True)
class FarmBotStatus:
    connected: bool
    mode: str
    device_name: Optional[str] = None
    firmware: Optional[str] = None
    last_contact: Optional[datetime] = None
    detail: str = ""


class FarmBotClient:
    """Interface every concrete FarmBot client must satisfy."""

    def status(self) -> FarmBotStatus:  # pragma: no cover - interface
        raise NotImplementedError

    def move_to(self, x: float, y: float, z: float) -> None:  # pragma: no cover
        raise NotImplementedError

    def run_weeding_sequence(self, targets: Sequence[WeedTarget]) -> None:  # pragma: no cover
        raise NotImplementedError


class DisconnectedFarmBot(FarmBotClient):
    """Default client used while the dashboard runs without hardware."""

    def status(self) -> FarmBotStatus:
        return FarmBotStatus(
            connected=False,
            mode="Prototype",
            detail=(
                "No FarmBot device is configured. Detection and target planning "
                "run normally; actuator commands are blocked."
            ),
            last_contact=datetime.now(timezone.utc),
        )

    def move_to(self, x: float, y: float, z: float) -> None:
        raise FarmBotNotConnected("Cannot move the gantry: no FarmBot device is connected.")

    def run_weeding_sequence(self, targets: Sequence[WeedTarget]) -> None:
        raise FarmBotNotConnected(
            "Cannot start a weeding pass: no FarmBot device is connected."
        )


_client: FarmBotClient = DisconnectedFarmBot()


def get_client() -> FarmBotClient:
    """Return the active client. Swap the implementation here to go live."""
    return _client


def set_client(client: FarmBotClient) -> None:
    """Install a real client (used by future hardware integration and tests)."""
    global _client
    _client = client


def weed_targets(result: DetectionResult, min_confidence: float = 0.0) -> List[WeedTarget]:
    """
    Convert detected weeds into intervention targets.

    Coordinates are image-space only. Translating them into FarmBot gantry
    coordinates requires the camera calibration and bed geometry, which the
    hardware integration will supply.
    """
    width, height = result.image_size
    targets: List[WeedTarget] = []
    weeds: Sequence[Detection] = sorted(result.weeds, key=lambda d: -d.confidence)
    for index, det in enumerate(weeds, start=1):
        if det.confidence < min_confidence:
            continue
        cx, cy = det.centroid
        # A box may extend past the frame edge; the reachable target does not.
        cx = min(max(cx, 0.0), float(max(width - 1, 0)))
        cy = min(max(cy, 0.0), float(max(height - 1, 0)))
        targets.append(
            WeedTarget(
                index=index,
                confidence=det.confidence,
                pixel_x=round(cx, 1),
                pixel_y=round(cy, 1),
                norm_x=round(cx / width, 4) if width else 0.0,
                norm_y=round(cy / height, 4) if height else 0.0,
                width_px=round(det.box[2] - det.box[0], 1),
                height_px=round(det.box[3] - det.box[1], 1),
            )
        )
    return targets


def intervention_decision(
    result: DetectionResult,
    weed_threshold: int = 1,
    confidence_threshold: float = 0.5,
) -> Tuple[bool, str]:
    """
    Decide whether the current frame warrants an intervention.

    Returns the decision and the reason behind it, so the interface can explain
    itself rather than presenting a bare verdict.
    """
    confident_weeds = [d for d in result.weeds if d.confidence >= confidence_threshold]
    if not result.detections:
        return False, "No objects were detected in this frame."
    if not confident_weeds:
        return False, (
            f"No weed reached the {confidence_threshold:.0%} confidence needed to act on."
        )
    if len(confident_weeds) < weed_threshold:
        return False, (
            f"{len(confident_weeds)} confident weed(s) detected, below the "
            f"threshold of {weed_threshold}."
        )
    return True, (
        f"{len(confident_weeds)} weed(s) at or above {confidence_threshold:.0%} "
        "confidence — a weeding pass would be queued."
    )
