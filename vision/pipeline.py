"""Detection orchestration: images -> detections -> camera DataFrame.

Ties the pieces together and exposes a small registry so the UI (and tests) can
select a detector by name.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .base import Detector
from .camera_csv import detections_to_camera_df
from .demo_detector import DemoDetector
from .image_source import ImageRef, collect_images
from .schema import ImageDetection

# Registry of detector classes. Optional back-ends are referenced lazily via
# factory functions so importing this module never pulls in torch/openai.
def _make_megadetector(**kwargs) -> Detector:
    from .megadetector import MegaDetectorDetector

    return MegaDetectorDetector(**kwargs)


def _make_openai(**kwargs) -> Detector:
    from .llm_openai import OpenAIVisionDetector

    return OpenAIVisionDetector(**kwargs)


DETECTORS: Dict[str, Callable[..., Detector]] = {
    "demo": DemoDetector,
    "megadetector": _make_megadetector,
    "openai": _make_openai,
}

DETECTOR_LABELS = {
    "demo": "Demo (synthetic, no model needed)",
    "megadetector": "MegaDetector + SpeciesNet (local CV)",
    "openai": "OpenAI GPT vision (LLM)",
}


def get_detector(name: str, **kwargs) -> Detector:
    """Instantiate a detector by registry name.

    Args:
        name: One of :data:`DETECTORS` (``"demo"``, ``"megadetector"``,
            ``"openai"``).
        **kwargs: Passed to the detector constructor.

    Raises:
        KeyError: If ``name`` is not a registered detector.
    """
    if name not in DETECTORS:
        raise KeyError(f"Unknown detector '{name}'. Options: {list(DETECTORS)}")
    return DETECTORS[name](**kwargs)


def _resolve_images(source, recursive: bool, limit: Optional[int]) -> List[ImageRef]:
    if isinstance(source, str):
        return collect_images(source, recursive=recursive, limit=limit)
    refs: List[ImageRef] = []
    for item in source:
        if isinstance(item, ImageRef):
            refs.append(item)
        else:
            from .image_source import read_capture_datetime

            dt, src = read_capture_datetime(str(item))
            refs.append(ImageRef(path=str(item), datetime=dt, datetime_source=src))
    if limit is not None:
        refs = refs[:limit]
    return refs


def run_detection(
    source,
    detector: Detector,
    *,
    recursive: bool = True,
    limit: Optional[int] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
    corrected_prefix: str = "",
) -> Tuple[pd.DataFrame, List[ImageDetection]]:
    """Run a detector over an image source and build the camera DataFrame.

    Args:
        source: A folder path, or a list of :class:`ImageRef`/path strings.
        detector: A ready :class:`Detector` instance.
        recursive: Recurse into sub-folders when ``source`` is a directory.
        limit: Optionally cap the number of images processed.
        progress: Optional ``(i, total, path)`` progress callback.
        stop_flag: Optional cancel hook.
        corrected_prefix: Value for the camera CSV ``Corrected`` column.

    Returns:
        ``(camera_df, detections)`` -- the wide-format camera DataFrame ready for
        ``data_loader.load_and_prepare_camera_data``, and the raw per-image
        detection objects (for previews/exports).
    """
    images = _resolve_images(source, recursive, limit)
    detections = detector.detect_batch(images, progress=progress, stop_flag=stop_flag)
    camera_df = detections_to_camera_df(detections, corrected_prefix=corrected_prefix)
    return camera_df, detections


def summarize_detections(detections: Sequence[ImageDetection]) -> Dict[str, Any]:
    """Compute summary statistics over a batch of detections (for the UI)."""
    n = len(detections)
    n_animal = sum(1 for d in detections if d.has_animal)
    n_error = sum(1 for d in detections if d.error)
    n_empty = n - n_animal - n_error
    species: Counter = Counter()
    total_animals = 0
    times = []
    for d in detections:
        for it in d.animals:
            species[it.label] += it.count
            total_animals += it.count
        if d.datetime is not None:
            times.append(d.datetime)
    time_range = (min(times), max(times)) if times else (None, None)
    return {
        "n_images": n,
        "n_with_animals": n_animal,
        "n_empty": n_empty,
        "n_errors": n_error,
        "detection_rate": (n_animal / n * 100.0) if n else 0.0,
        "total_animals": total_animals,
        "unique_species": len(species),
        "species_counts": dict(species.most_common()),
        "time_range": time_range,
    }
