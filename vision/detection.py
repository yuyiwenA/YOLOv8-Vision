"""YOLO inference wrapper supporting both plain detection and tracking modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class Detection:
    """One detected object in a frame."""

    bbox: tuple  # x1, y1, x2, y2 (int)
    conf: float
    cls: int
    track_id: Optional[int] = None

    @property
    def center(self):
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self):
        return self.width * self.height


class VisionModel:
    """Thin wrapper that normalises Ultralytics output into :class:`Detection`."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        classes: Optional[Sequence[int]] = None,
        device: Optional[str] = None,
        track: bool = False,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.classes = list(classes) if classes else None
        self.device = device
        self.track = bool(track)
        self.tracker = tracker

    @property
    def names(self) -> dict:
        names = getattr(self.model, "names", None) or {}
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
        return dict(names)

    def resolve_classes(self, spec) -> Optional[List[int]]:
        """Map ``"person,car"`` or ``"0,2"`` to class ids; ``None`` means no filter."""
        if spec is None:
            return None
        names = self.names
        lookup = {str(v).lower(): k for k, v in names.items()}
        ids: List[int] = []
        for raw in spec:
            token = str(raw).strip()
            if not token:
                continue
            if token.lower() in ("all", "*"):
                return None
            if token.lstrip("-").isdigit():
                ids.append(int(token))
            elif token.lower() in lookup:
                ids.append(lookup[token.lower()])
            else:
                raise ValueError(f"unknown class {token!r}; known: {sorted(names.values())}")
        return sorted(set(ids)) or None

    def infer(self, frame: np.ndarray) -> List[Detection]:
        common = dict(
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        if self.track:
            results = self.model.track(
                frame, persist=True, tracker=self.tracker, **common
            )
        else:
            results = self.model.predict(frame, **common)

        height, width = frame.shape[:2]
        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            ids = boxes.id
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                track_id = None
                if ids is not None and i < len(ids):
                    track_id = int(ids[i])
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        conf=float(box.conf[0]),
                        cls=int(box.cls[0]),
                        track_id=track_id,
                    )
                )
        return detections
