"""The detection pipeline: read video -> infer -> annotate -> export."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Optional

import cv2

from .detection import VisionModel
from .draw import draw_detections, draw_hud, draw_progress_bar
from .report import DetectionReport

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="YOLOv8-Vision: video object detection with CSV trajectory export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", default=None,
                        help="input video; auto-detects the first video file in the project folder")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference resolution")
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu / cuda:0")
    parser.add_argument("--classes", default="all",
                        help="comma separated class names or ids to keep ('all' = no filter)")
    parser.add_argument("--track", action="store_true", help="assign persistent track ids (ByteTrack)")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--out", default="outputs", help="output directory")
    parser.add_argument("--no-video", dest="save_video", action="store_false", help="skip the result video")
    parser.add_argument("--no-csv", dest="save_csv", action="store_false", help="skip the CSV")
    parser.add_argument("--no-chart", dest="save_chart", action="store_false", help="skip the summary chart")
    parser.add_argument("--show", action="store_true", help="open a preview window (needs a display)")
    parser.set_defaults(save_video=True, save_csv=True, save_chart=True)
    return parser


def _autodetect_video(root: str) -> Optional[str]:
    for name in sorted(os.listdir(root)):
        if name.lower().endswith(VIDEO_SUFFIXES):
            return os.path.join(root, name)
    return None


def run(args: argparse.Namespace, video_name: str = "result.mp4",
        csv_name: str = "trajectory.csv") -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(root)

    video_path = args.video
    if video_path is None:
        video_path = _autodetect_video(project_root) or _autodetect_video(os.getcwd())
        if video_path is None:
            print("[error] no --video given and no video file found in the project folder",
                  file=sys.stderr)
            return 2
        print(f"[info] auto-detected video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] cannot open video: {video_path}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 1e-3:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[info] {width}x{height} @ {fps:.2f} fps, {total or '?'} frames")

    os.makedirs(args.out, exist_ok=True)

    model = VisionModel(
        model_path=args.model, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        device=args.device, track=args.track, tracker=args.tracker,
    )
    try:
        class_ids = model.resolve_classes(args.classes.split(","))
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    model.classes = class_ids
    names = model.names
    print(f"[info] mode: {'tracking' if args.track else 'detection'}"
          f" | classes: {[names.get(c, c) for c in class_ids] if class_ids else 'all'}")

    report = DetectionReport(fps=fps, frame_size=(height, width), names=names)

    writer = None
    video_out = os.path.join(args.out, video_name)
    if args.save_video:
        writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            print("[warn] video writer failed to open; continuing without video output")
            writer = None

    csv_handle = csv_writer = None
    csv_out = os.path.join(args.out, csv_name)
    if args.save_csv:
        csv_handle = open(csv_out, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerow(
            ["frame_id", "time_s", "track_id", "class_id", "class", "confidence",
             "x1", "y1", "x2", "y2", "cx", "cy", "width", "height"]
        )

    window = "yolov8-vision"
    show = bool(args.show)
    if show:
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        except cv2.error:
            print("[warn] no display available; running headless")
            show = False

    frame_id = 0
    processed = 0
    t_first = t_last = None
    started = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.stride > 1 and frame_id % args.stride:
                frame_id += 1
                continue

            detections = model.infer(frame)
            report.update(detections)

            if csv_writer is not None:
                for det in detections:
                    cx, cy = det.center
                    csv_writer.writerow([
                        frame_id, round(frame_id / fps, 4),
                        det.track_id if det.track_id is not None else "",
                        det.cls, names.get(det.cls, str(det.cls)), round(det.conf, 4),
                        det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
                        round(cx, 1), round(cy, 1), det.width, det.height,
                    ])

            canvas = frame.copy()
            draw_detections(canvas, detections, names=names, show_track_id=args.track)
            now = time.time()
            span = now - t_first if t_first else 0.0
            processed_rate = (processed - 1) / span if span > 0.5 else 0.0
            counts = {}
            for det in detections:
                counts[names.get(det.cls, det.cls)] = counts.get(names.get(det.cls, det.cls), 0) + 1
            summary = "  ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no objects"
            draw_hud(canvas, [
                f"frame {frame_id}  |  {frame_id / fps:6.2f} s",
                f"objects {len(detections)}   {summary}",
                f"throughput {processed_rate:5.1f} fps   {'tracking' if args.track else 'detection'}",
                f"source {os.path.basename(video_path)}",
            ])
            if total:
                draw_progress_bar(canvas, frame_id / float(total))

            if writer is not None:
                writer.write(canvas)
            if show:
                cv2.imshow(window, canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[info] interrupted by user")
                    break

            processed += 1
            now = time.time()
            if t_first is None:
                t_first = now
            t_last = now
            if processed % 30 == 0:
                span = max(now - t_first, 1e-6)
                print(f"[info] {processed}/{total or '?'} frames  ({(processed - 1) / span:.1f} fps)")

            frame_id += 1
            if args.max_frames and processed >= args.max_frames:
                print(f"[info] reached --max-frames={args.max_frames}")
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_handle is not None:
            csv_handle.close()
        if show:
            cv2.destroyAllWindows()

    elapsed = time.time() - started
    report.elapsed_s = elapsed
    report.frames = processed
    report.processing_span_s = (t_last - t_first) if (t_first and t_last) else 0.0
    report.per_frame_totals = report.per_frame_totals[:processed]

    json_path = os.path.join(args.out, "detections_summary.json")
    summary = report.write_json(json_path, extras={
        "source": os.path.basename(video_path),
        "model": args.model,
        "mode": "tracking" if args.track else "detection",
    })

    artifacts = [video_out if writer else None, csv_out if csv_writer else None, json_path]
    if args.save_chart:
        chart_path = os.path.join(args.out, "analysis.png")
        if report.write_chart(chart_path):
            artifacts.append(chart_path)

    span = report.processing_span_s
    steady = (processed - 1) / span if (processed > 1 and span > 0) else 0.0
    print("\n" + "=" * 62)
    print("  DETECTION SUMMARY")
    print("=" * 62)
    print(f"  frames processed : {processed} ({processed / fps:.2f} s of footage)")
    print(f"  wall time        : {elapsed:.2f} s (model load + warm-up included)")
    print(f"  steady throughput: {steady:.2f} fps  ({steady / fps:.2f}x real time)")
    print(f"  detections       : {report.total_detections}")
    if report.unique_tracks:
        print(f"  unique tracks    : {len(report.unique_tracks)}")
    for cls, count in sorted(report.class_counts.items()):
        print(f"    - {names.get(cls, cls):<16} {count}")
    print("\n  artifacts")
    for path in artifacts:
        if path:
            print(f"    - {path}")
    print("=" * 62)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n[info] aborted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
