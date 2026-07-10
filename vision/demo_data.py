"""Synthetic image references for a zero-setup detection demo.

The :class:`~vision.demo_detector.DemoDetector` fabricates detections from the
*path* of each image and never opens the pixels, so the whole image -> camera
dataset -> analysis pipeline can be exercised without any real image files.

``synthetic_image_refs`` produces a set of :class:`~vision.image_source.ImageRef`
objects with realistic camera-trap filenames whose timestamps span a chosen date
window. The default window matches the bundled Willanch Slough sensor CSV, so a
generated demo detection set overlaps the demo tide/weather timeline for a
coherent end-to-end run.
"""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta
from typing import List, Sequence, Union

from .image_source import ImageRef

# Date span of the bundled ``willanch_sensor_final.csv`` (2021-06-15 .. 2022-06-23).
# Used as the default demo window so synthetic camera timestamps line up with the
# demo sensor data the Analysis page loads.
DEMO_START = datetime(2021, 6, 15)
DEMO_END = datetime(2022, 6, 24)

_DEFAULT_CAMERAS = ("CAM01", "CAM02", "CAM03")

_DateLike = Union[datetime, date, str, None]


def _as_datetime(value: _DateLike, fallback: datetime, *, end: bool = False) -> datetime:
    """Coerce a date/datetime/ISO-string (or ``None``) to a ``datetime``."""
    if value is None:
        return fallback
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time(23, 59, 59) if end else time(0, 0, 0))
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return fallback


def synthetic_image_refs(
    count: int = 120,
    start: _DateLike = None,
    end: _DateLike = None,
    *,
    seed: int = 1234,
    cameras: Sequence[str] = _DEFAULT_CAMERAS,
) -> List[ImageRef]:
    """Build ``count`` synthetic :class:`ImageRef` objects for the demo detector.

    Args:
        count: Number of images to fabricate (clamped to at least 1).
        start: Window start as ``datetime``/``date``/ISO string; defaults to the
            bundled sensor's first day.
        end: Window end; defaults to the bundled sensor's last day.
        seed: Seed for reproducible timestamps/filenames.
        cameras: Camera folder names to round-robin across.

    Returns:
        A time-sorted list of :class:`ImageRef` with ``datetime_source='filename'``.
        The paths are synthetic (no files on disk) -- fine for the demo detector,
        which never reads image pixels.
    """
    count = max(1, int(count))
    start_dt = _as_datetime(start, DEMO_START)
    end_dt = _as_datetime(end, DEMO_END, end=True)
    if end_dt <= start_dt:
        end_dt = start_dt + timedelta(days=1)

    cams = tuple(cameras) or _DEFAULT_CAMERAS
    rng = random.Random(seed)
    span_seconds = (end_dt - start_dt).total_seconds()

    times = sorted(
        start_dt + timedelta(seconds=rng.uniform(0, span_seconds)) for _ in range(count)
    )

    refs: List[ImageRef] = []
    for i, t in enumerate(times):
        # Whole-second timestamps so the filename round-trips through the
        # filename-timestamp parser exactly.
        t = t.replace(microsecond=0)
        cam = cams[i % len(cams)]
        fname = f"{cam}_{t.strftime('%Y%m%d_%H%M%S')}.JPG"
        refs.append(
            ImageRef(
                path=f"demo_images/{cam}/{fname}",
                datetime=t,
                datetime_source="filename",
            )
        )
    return refs
