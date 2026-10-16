"""Drawing helpers for the annotated output video."""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

#: Stable colour per COCO class id for the classes people actually film.
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (86, 196, 92),     # person
    1: (60, 170, 250),    # bicycle
    2: (200, 110, 60),    # car
    3: (120, 140, 240),   # motorcycle
    5: (60, 60, 200),     # bus
    7: (170, 190, 60),    # truck
    32: (40, 180, 255),   # sports ball
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(cls: int, seed: int = 0) -> Tuple[int, int, int]:
    if cls in CLASS_COLORS:
        return CLASS_COLORS[cls]
    hue = (int(cls) * 53 + int(seed) * 17) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 190, 235]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_detections(
    frame: np.ndarray,
    detections: Iterable,
    names: Optional[Dict[int, str]] = None,
    show_track_id: bool = False,
) -> np.ndarray:
    """Draw boxes with a filled label chip. Modifies ``frame`` in place."""
    names = names or {}
    for det in detections:
        color = color_for(det.cls)
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = names.get(det.cls, str(det.cls))
        label += f" {det.conf:.2f}"
        if show_track_id and det.track_id is not None:
            label = f"#{det.track_id} " + label

        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.52, 1)
        top = max(0, y1 - th - baseline - 8)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 10, top + th + baseline + 6), color, -1)
        cv2.putText(frame, label, (x1 + 5, top + th + 2), FONT, 0.52, (22, 24, 28), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, lines: Sequence[str], origin=(12, 12)) -> np.ndarray:
    """Translucent top-left statistics panel."""
    if not lines:
        return frame

    padding, line_h, scale = 10, 20, 0.5
    width = max(cv2.getTextSize(t, FONT, scale, 1)[0][0] for t in lines) + padding * 2
    height = line_h * len(lines) + padding * 2 - 4
    x, y = origin

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (28, 32, 38), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (110, 160, 220), 1)

    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x + padding, y + padding + line_h * i + 12),
                    FONT, scale, (235, 238, 242), 1, cv2.LINE_AA)
    return frame


def draw_progress_bar(frame: np.ndarray, fraction: float, color=(110, 170, 240)) -> np.ndarray:
    """Thin progress bar pinned to the bottom edge (useful in still frames)."""
    fraction = float(max(0.0, min(1.0, fraction)))
    height, width = frame.shape[:2]
    y = height - 6
    cv2.rectangle(frame, (0, y), (width, height), (38, 42, 48), -1)
    cv2.rectangle(frame, (0, y), (int(width * fraction), height), color, -1)
    return frame
