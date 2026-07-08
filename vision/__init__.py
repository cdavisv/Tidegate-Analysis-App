"""Vision detection subsystem for the Tidegate Analysis App.

This package turns a set of camera-trap images into the wide-format *camera CSV*
that the analysis pipeline consumes. Detection can be performed by:

* :class:`vision.demo_detector.DemoDetector` -- a dependency-free synthetic
  detector for testing the full pipeline without heavy models.
* :class:`vision.megadetector.MegaDetectorDetector` -- MegaDetector (via
  Pytorch-Wildlife) for animal/person/vehicle detection, optionally paired with
  a SpeciesNet / Pytorch-Wildlife classifier for species identification. This is
  the AddaxAI-style local computer-vision path.
* :class:`vision.llm_openai.OpenAIVisionDetector` -- a multimodal LLM
  (OpenAI GPT) that identifies species directly from the image.

All detectors return the same :class:`vision.schema.ImageDetection` objects, so
they are interchangeable and can be compared against each other.

Typical usage::

    from vision import get_detector, run_detection
    detector = get_detector("demo")
    camera_df, detections = run_detection("/path/to/images", detector)
    camera_df.to_csv("camera_from_cv.csv", index=False)
"""

from __future__ import annotations

from .schema import (
    CATEGORY_ANIMAL,
    CATEGORY_EMPTY,
    CATEGORY_PERSON,
    CATEGORY_VEHICLE,
    DetectionItem,
    ImageDetection,
    to_scientific_name,
    COMMON_TO_SCIENTIFIC,
)
from .camera_csv import detections_to_camera_df, detections_to_long_df, save_camera_csv
from .image_source import ImageRef, iter_image_paths, read_capture_datetime, collect_images
from .base import Detector, DetectorUnavailableError
from .demo_detector import DemoDetector
from .pipeline import DETECTORS, get_detector, run_detection, summarize_detections

__all__ = [
    # schema
    "DetectionItem",
    "ImageDetection",
    "to_scientific_name",
    "COMMON_TO_SCIENTIFIC",
    "CATEGORY_ANIMAL",
    "CATEGORY_PERSON",
    "CATEGORY_VEHICLE",
    "CATEGORY_EMPTY",
    # csv
    "detections_to_camera_df",
    "detections_to_long_df",
    "save_camera_csv",
    # image source
    "ImageRef",
    "iter_image_paths",
    "read_capture_datetime",
    "collect_images",
    # detectors
    "Detector",
    "DetectorUnavailableError",
    "DemoDetector",
    # pipeline
    "DETECTORS",
    "get_detector",
    "run_detection",
    "summarize_detections",
]


def __getattr__(name):  # pragma: no cover - thin lazy import shim
    """Lazily expose optional detectors that carry heavy/optional imports.

    ``MegaDetectorDetector`` and ``OpenAIVisionDetector`` live in modules that
    import optional third-party libraries. We import them lazily so that simply
    importing :mod:`vision` never fails when those libraries are absent.
    """
    if name == "MegaDetectorDetector":
        from .megadetector import MegaDetectorDetector

        return MegaDetectorDetector
    if name == "OpenAIVisionDetector":
        from .llm_openai import OpenAIVisionDetector

        return OpenAIVisionDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
