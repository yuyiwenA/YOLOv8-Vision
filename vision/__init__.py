"""YOLOv8-Vision: a small, organised detection pipeline.

The original repository was a single 66-line ``main.py`` with a hard-coded
video path and a fixed output name. It worked for one file layout only. This
package keeps the same behaviour as its default while making the pipeline
configurable, measurable and reportable.
"""

__version__ = "0.2.0"

__all__ = ["Detection", "VisionModel", "run"]
