"""Detector base class shared by all detection back-ends."""

from __future__ import annotations

import abc
import os
from typing import Callable, List, Optional, Sequence

from .image_source import ImageRef, read_capture_datetime
from .schema import ImageDetection

# progress callback: (index, total, current_path) -> None
ProgressCallback = Optional[Callable[[int, int, str], None]]


class DetectorUnavailableError(RuntimeError):
    """Raised when a detector is invoked but its dependencies are missing."""


class Detector(abc.ABC):
    """Abstract base for image detectors.

    Subclasses implement :meth:`detect_image`. The base class provides
    :meth:`detect_batch` with per-image error isolation and progress reporting,
    and :meth:`available` so callers (and the UI) can check readiness before
    running anything expensive.
    """

    #: Short, stable identifier used in registries and the UI.
    name: str = "detector"
    #: One-line human description.
    description: str = ""

    def available(self) -> tuple[bool, str]:
        """Return ``(is_ready, reason)``.

        The default implementation reports available. Detectors with optional
        dependencies or API keys should override this.
        """
        return True, "ready"

    @abc.abstractmethod
    def detect_image(self, image: ImageRef) -> ImageDetection:
        """Run detection on a single image and return an :class:`ImageDetection`."""

    def _coerce(self, image) -> ImageRef:
        if isinstance(image, ImageRef):
            return image
        # Accept a bare path string.
        path = str(image)
        dt, src = read_capture_datetime(path)
        return ImageRef(path=path, datetime=dt, datetime_source=src)

    def detect_batch(
        self,
        images: Sequence,
        progress: ProgressCallback = None,
        stop_flag: Optional[Callable[[], bool]] = None,
    ) -> List[ImageDetection]:
        """Detect over a sequence of images or paths.

        Failures on individual images are captured into that image's
        :attr:`ImageDetection.error` rather than aborting the whole batch.

        Args:
            images: Sequence of :class:`ImageRef` or path strings.
            progress: Optional ``(i, total, path)`` progress callback.
            stop_flag: Optional callable; when it returns ``True`` the batch
                stops early (used for a UI cancel button).

        Returns:
            A list of :class:`ImageDetection`, one per processed image.
        """
        ready, reason = self.available()
        if not ready:
            raise DetectorUnavailableError(f"{self.name} unavailable: {reason}")

        total = len(images)
        results: List[ImageDetection] = []
        for i, raw in enumerate(images):
            if stop_flag is not None and stop_flag():
                break
            ref = self._coerce(raw)
            if progress is not None:
                try:
                    progress(i, total, ref.path)
                except Exception:
                    pass
            try:
                det = self.detect_image(ref)
            except Exception as exc:  # never let one image kill the run
                det = ImageDetection(
                    image_path=ref.path,
                    datetime=ref.datetime,
                    items=[],
                    detector=self.name,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(det)
        if progress is not None:
            try:
                progress(total, total, "")
            except Exception:
                pass
        return results
