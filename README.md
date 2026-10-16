# YOLOv8-Vision

Video object detection built on YOLOv8 and OpenCV. It runs frame-wise inference,
writes an annotated video, and exports frame-level detection records to CSV.

## What changed in this revision

The original project was a single `main.py` with a hard-coded `test_video.mp4`
and a fixed output name. The README referred to a file that did not exist in the
repository, and there was no way to change the confidence threshold, filter
classes or measure throughput.

| Original | Now |
|---|---|
| `main.py` monolith, hard-coded paths | `vision/` package plus a thin `main.py` entry point |
| `VIDEO_PATH = "test_video.mp4"` hard-coded | auto-detects the video, or `--video PATH` |
| no confidence, IoU or resolution control | `--conf`, `--iou`, `--imgsz`, `--device` |
| all 80 COCO classes always drawn | `--classes person,car`, by name or id |
| no object identity | `--track` for persistent ByteTrack ids |
| no performance measurement | throughput in fps plus `realtime_factor` in the JSON report |
| empty video produced a CSV with no header | header always written, plus a JSON summary and a chart |
| no class breakdown | per-class counts, mean confidence, objects-per-frame timeline |
| `result.mp4` and `trajectory.csv` fixed | all artifacts written under `outputs/` |

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py                                    # auto-detects the video in this folder
python main.py --video 14488533-hd_1920_1080_25fps.mp4
python main.py --track --classes person --conf 0.4
python main.py --max-frames 300 --device cpu
```

`python main.py --help` lists every option.

### Programmatic use

The original signature still works:

```python
from main import main
main("test_video.mp4", output_video="result.mp4", output_csv="trajectory.csv")
```

### Reference run

The results in `outputs/` come from the night traffic clip
(`14488533-hd_1920_1080_25fps.mp4`, 1920x1080, 25 fps, 301 frames):

```bash
python main.py --video 14488533-hd_1920_1080_25fps.mp4 --track --conf 0.25
# 301 frames in 152 s wall time, 2.16 fps steady state (0.09x real time, CPU)
# 3081 detections: car 3007, bus 33, truck 28, person 13
# 188 unique track ids
```

Night footage is a hard case for the confidence threshold. Headlights raise the
score of distant vehicles far above their visual certainty, and the histogram in
`analysis.png` spreads from 0.25 to 0.9 as a result. Raising the threshold trades
away real vehicles along with the noise, so filtering by class is the more
effective control when the scene is known.

## Output

```
outputs/
├── result.mp4                annotated video: boxes, labels, HUD, progress bar
├── trajectory.csv            frame_id, time_s, track_id, class, confidence,
│                             x1, y1, x2, y2, cx, cy, width, height
├── detections_summary.json   throughput, per-class totals, mean confidence
└── analysis.png              three-panel chart: per-class counts, confidence
                              histogram, objects per frame
```

`detections_summary.json` reports throughput twice. `elapsed_s` is total wall
time and includes model loading and first-inference warm-up. `processing_fps` is
derived from the span between the first and last processed frame and describes
steady-state speed. On a short clip the two differ by a factor of five or more,
so reporting only the first would understate the pipeline considerably.

## Screenshots

```bash
python tools/screenshot.py outputs/result.mp4 -n 6 -o outputs/screenshots
```

Writes evenly spaced annotated frames plus a `contact_sheet.png` montage on a
light background, ready for a README or a slide.

## Layout

```
main.py                 CLI and backward-compatible main()
vision/detection.py     VisionModel, Detection dataclass, class resolution
vision/pipeline.py      the run loop and the argument parser
vision/draw.py          box, label, HUD and progress rendering
vision/report.py        statistics collection, JSON and chart output
tools/screenshot.py     frame extraction and contact sheets
```

## Notes

- The YOLOv8n weights download automatically on first run.
- `mp4`, `avi`, `mov`, `mkv` and `webm` inputs are supported.
- Tracking uses the `bytetrack.yaml` config bundled with Ultralytics.

## Relationship to the sports toolkit

This repository is the generic detection layer. It finds objects, keeps ids and
exports records, and stops there.

Analysis on top of it lives in the sibling project `Sports-Tracking-Toolkit`:
continuous ball paths, jersey-colour team assignment, match events, possession
and label-free quality scoring. That project also documents, with measurements,
which footage this class of pipeline handles and which it does not.
