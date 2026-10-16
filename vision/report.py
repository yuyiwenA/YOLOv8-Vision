"""Detection statistics collection and reporting.

Adds what the original single-file script lacked entirely: any measured
throughput, any per-class breakdown, and any artifact readable without opening
a CSV. Outputs ``detections_summary.json`` and ``analysis.png``.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PALETTE = ["#2f6fed", "#e0533d", "#12a37a", "#c9821a", "#8a5cd6", "#3d9be0"]


class DetectionReport:
    """Accumulates per-frame detection data and renders the summary."""

    def __init__(self, fps: float, frame_size: Tuple[int, int], names: Optional[Dict[int, str]] = None):
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.frame_size = (int(frame_size[1]), int(frame_size[0]))  # width, height
        self.names = names or {}

        self.frames = 0
        self.elapsed_s = 0.0
        #: Time between the first and last processed frame. Throughput is
        #: derived from this instead of total wall time, which also contains
        #: model loading and first-inference warm-up and can understate the
        #: steady-state rate several-fold.
        self.processing_span_s = 0.0
        self.class_counts: Counter = Counter()
        self.confidences: Dict[int, List[float]] = defaultdict(list)
        self.per_frame_totals: List[int] = []
        self.total_detections = 0
        self.unique_tracks: set = set()

    # ------------------------------------------------------------------ ingest
    def update(self, detections: Sequence) -> None:
        self.frames += 1
        self.per_frame_totals.append(len(detections))
        for det in detections:
            self.class_counts[det.cls] += 1
            self.confidences[det.cls].append(det.conf)
            self.total_detections += 1
            if getattr(det, "track_id", None) is not None:
                self.unique_tracks.add(det.track_id)

    # ------------------------------------------------------------------ output
    def build(self, extras: Optional[dict] = None) -> dict:
        span = self.processing_span_s or self.elapsed_s
        frames = self.frames
        rate = (frames - 1) / span if frames > 1 and span > 0 else None

        mean_conf = {}
        for cls, values in self.confidences.items():
            mean_conf[self.names.get(cls, str(cls))] = round(float(np.mean(values)), 4)

        report = {
            "video": {
                "fps": round(self.fps, 3),
                "width": self.frame_size[0],
                "height": self.frame_size[1],
                "frames_processed": frames,
                "duration_s": round(frames / self.fps, 2),
            },
            "throughput": {
                "elapsed_s": round(self.elapsed_s, 3),
                "processing_span_s": round(span, 3),
                "processing_fps": round(rate, 2) if rate else None,
                "realtime_factor": round(rate / self.fps, 3) if rate else None,
                "warmup_excluded": bool(self.processing_span_s),
            },
            "totals": {
                "detections": self.total_detections,
                "unique_tracks": len(self.unique_tracks) or None,
                "per_class": {self.names.get(c, str(c)): int(n) for c, n in sorted(self.class_counts.items())},
                "mean_confidence_per_class": mean_conf,
                "busiest_frame": max(self.per_frame_totals) if self.per_frame_totals else 0,
                "mean_boxes_per_frame": round(
                    float(np.mean(self.per_frame_totals)), 3
                ) if self.per_frame_totals else 0.0,
            },
        }
        if extras:
            report.update(extras)
        return report

    def write_json(self, path: str, extras: Optional[dict] = None) -> dict:
        report = self.build(extras)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        return report

    def write_chart(self, path: str) -> Optional[str]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return None
        if not self.class_counts:
            return None

        plt.rcParams.update({
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d8dde3",
            "axes.labelcolor": "#333a44",
            "text.color": "#1d232b",
            "xtick.color": "#5b6673",
            "ytick.color": "#5b6673",
            "axes.grid": True,
            "grid.color": "#eceff3",
            "grid.linewidth": 0.8,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
        })

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
        fig.suptitle("YOLOv8-Vision: detection analysis", fontsize=12,
                     fontweight="bold", color="#12171d", x=0.02, ha="left")

        # 1. per-class totals
        ax = axes[0]
        labels = [self.names.get(c, str(c)) for c in self.class_counts]
        values = list(self.class_counts.values())
        ax.bar(labels, values, color=PALETTE[: len(labels)], width=0.55)
        ax.set_title("Detections per class")
        ax.set_ylabel("boxes")
        ax.grid(axis="x", visible=False)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        # 2. confidence distribution
        ax = axes[1]
        for i, (cls, values_) in enumerate(sorted(self.confidences.items())):
            ax.hist(values_, bins=22, range=(0, 1), alpha=0.55,
                    color=PALETTE[i % len(PALETTE)], label=self.names.get(cls, str(cls)))
        ax.set_title("Confidence distribution")
        ax.set_xlabel("confidence")
        ax.set_ylabel("count")
        if self.confidences:
            ax.legend(frameon=False, fontsize=8)

        # 3. boxes per frame over time
        ax = axes[2]
        if self.per_frame_totals:
            ax.plot(np.arange(len(self.per_frame_totals)) / self.fps,
                    self.per_frame_totals, color="#2f6fed", linewidth=0.9)
            ax.fill_between(np.arange(len(self.per_frame_totals)) / self.fps,
                            self.per_frame_totals, color="#2f6fed", alpha=0.15)
        ax.set_title("Objects per frame")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("boxes")

        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
