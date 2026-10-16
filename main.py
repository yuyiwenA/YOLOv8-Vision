"""YOLOv8-Vision: detection pipeline entry point.

Command line (recommended)::

    python main.py                       # auto-detects the video in this folder
    python main.py --video test_video.mp4 --track
    python main.py --help

Programmatic use (backward compatible with the original script's signature)::

    from main import main
    main("test_video.mp4", output_video="result.mp4", output_csv="trajectory.csv")
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision.pipeline import build_parser, run  # noqa: E402


def main(video_path=None, output_video=None, output_csv=None):
    """Legacy-compatible wrapper around :func:`vision.pipeline.run`.

    ``video_path``, ``output_video`` and ``output_csv`` mirror the parameters of
    the original monolithic script, but every option is now available through
    the command line as well.
    """
    args = build_parser().parse_args([])
    if video_path is not None:
        args.video = video_path

    out_dir = args.out
    video_name, csv_name = "result.mp4", "trajectory.csv"
    if output_video:
        out_dir = os.path.dirname(os.path.abspath(output_video)) or out_dir
        video_name = os.path.basename(output_video)
    if output_csv:
        out_dir = os.path.dirname(os.path.abspath(output_csv)) or out_dir
        csv_name = os.path.basename(output_csv)
    args.out = out_dir

    return run(args, video_name=video_name, csv_name=csv_name)


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
