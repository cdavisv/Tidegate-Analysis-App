"""Local computer-vision detector: MegaDetector (+ optional SpeciesNet).

This is the AddaxAI-style path. It uses `Pytorch-Wildlife
<https://github.com/microsoft/CameraTraps>`_ to run **MegaDetector** for
animal/person/vehicle localization, and (optionally) **SpeciesNet**
(``pip install speciesnet``) or a Pytorch-Wildlife classifier to turn animal
crops into species labels.

Design goals
------------
* **Import-safe.** Heavy libraries (``torch``, ``PytorchWildlife``,
  ``speciesnet``) are imported lazily inside methods. Importing this module
  never fails, so the Streamlit app loads even on a machine with no models.
* **Graceful degradation.** :meth:`available` reports exactly what is missing
  with install hints; running a batch when unavailable raises a clear error
  instead of a cryptic traceback.
* **Testable parsing.** The raw-result -> :class:`DetectionItem` conversion is a
  pure function (:func:`parse_megadetector_result`) that can be unit-tested
  without any model weights.

Install (on the user's machine, ideally a GPU box)::

    pip install PytorchWildlife            # MegaDetector v5/v6
    pip install speciesnet                 # optional species classifier
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, List, Optional

from .base import Detector, DetectorUnavailableError
from .schema import (
    CATEGORY_ANIMAL,
    CATEGORY_PERSON,
    CATEGORY_VEHICLE,
    DetectionItem,
    ImageDetection,
    to_scientific_name,
)

# MegaDetector's three top-level classes, by integer id and by string.
_MD_CATEGORY = {
    0: CATEGORY_ANIMAL,
    1: CATEGORY_PERSON,
    2: CATEGORY_VEHICLE,
    "animal": CATEGORY_ANIMAL,
    "person": CATEGORY_PERSON,
    "vehicle": CATEGORY_VEHICLE,
}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def parse_megadetector_result(
    result: Dict[str, Any],
    threshold: float = 0.2,
    source: str = "megadetector",
) -> List[DetectionItem]:
    """Convert a Pytorch-Wildlife single-image result into detection items.

    This is intentionally tolerant of the small differences between
    Pytorch-Wildlife versions. It accepts a dict containing a ``supervision``
    ``Detections`` object under ``"detections"`` (with ``.xyxy``,
    ``.confidence`` and ``.class_id`` arrays) and/or parallel ``"labels"``.

    Args:
        result: The dict returned by ``single_image_detection``.
        threshold: Minimum confidence to keep a box.
        source: Detector name to stamp onto each item.

    Returns:
        A list of :class:`DetectionItem` (one per surviving box).
    """
    items: List[DetectionItem] = []
    dets = result.get("detections") if isinstance(result, dict) else None
    if dets is None:
        return items

    xyxy = getattr(dets, "xyxy", None)
    conf = getattr(dets, "confidence", None)
    class_id = getattr(dets, "class_id", None)
    labels = result.get("labels") if isinstance(result, dict) else None

    n = 0
    for candidate in (conf, class_id, xyxy, labels):
        if candidate is not None:
            try:
                n = max(n, len(candidate))
            except TypeError:
                pass
    for i in range(n):
        c = float(conf[i]) if conf is not None and i < len(conf) else None
        if c is not None and c < threshold:
            continue
        cat_key = None
        if class_id is not None and i < len(class_id):
            cat_key = int(class_id[i])
        category = _MD_CATEGORY.get(cat_key, CATEGORY_ANIMAL)
        items.append(
            DetectionItem(
                category=category,
                label="" if category == CATEGORY_ANIMAL else category.title(),
                count=1,
                confidence=c,
                source=source,
            )
        )
    return items


class MegaDetectorDetector(Detector):
    """MegaDetector-based local detector with optional species classification."""

    name = "megadetector"
    description = "MegaDetector (Pytorch-Wildlife) + optional SpeciesNet species ID."

    def __init__(
        self,
        version: str = "MDV6-yolov9-c",
        detection_threshold: float = 0.2,
        classification_threshold: float = 0.3,
        device: Optional[str] = None,
        use_speciesnet: bool = True,
        country: Optional[str] = "USA",
        admin1_region: Optional[str] = "OR",
    ) -> None:
        self.version = version
        self.detection_threshold = float(detection_threshold)
        self.classification_threshold = float(classification_threshold)
        self.device = device
        self.use_speciesnet = bool(use_speciesnet)
        self.country = country
        self.admin1_region = admin1_region
        self._det_model = None
        self._clf = None
        self._resolved_device = None

    # -- readiness ---------------------------------------------------------
    def available(self) -> tuple[bool, str]:
        if not _module_available("torch"):
            return False, "PyTorch not installed. Install with: pip install torch"
        if not _module_available("PytorchWildlife"):
            return (
                False,
                "Pytorch-Wildlife not installed. Install with: pip install PytorchWildlife",
            )
        return True, "ready"

    def speciesnet_available(self) -> bool:
        return self.use_speciesnet and _module_available("speciesnet")

    # -- lazy model loading -----------------------------------------------
    def _resolve_device(self) -> str:
        if self._resolved_device:
            return self._resolved_device
        dev = self.device
        if dev is None:
            try:
                import torch  # type: ignore

                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev = "cpu"
        self._resolved_device = dev
        return dev

    def _load_detector(self):
        if self._det_model is not None:
            return self._det_model
        device = self._resolve_device()
        from PytorchWildlife.models import detection as pw_detection  # type: ignore

        # Prefer MegaDetector v6; fall back to v5 for older installs.
        last_err = None
        for ctor_name, kwargs in (
            ("MegaDetectorV6", {"device": device, "pretrained": True, "version": self.version}),
            ("MegaDetectorV6", {"device": device, "pretrained": True}),
            ("MegaDetectorV5", {"device": device, "pretrained": True}),
        ):
            ctor = getattr(pw_detection, ctor_name, None)
            if ctor is None:
                continue
            try:
                self._det_model = ctor(**kwargs)
                return self._det_model
            except Exception as exc:  # pragma: no cover - needs weights
                last_err = exc
        raise DetectorUnavailableError(
            f"Could not initialize a MegaDetector model: {last_err}"
        )

    def _load_classifier(self):
        if not self.speciesnet_available():
            return None
        if self._clf is not None:
            return self._clf
        try:  # pragma: no cover - requires speciesnet weights
            from speciesnet import SpeciesNet  # type: ignore

            self._clf = SpeciesNet()
        except Exception:
            self._clf = None
        return self._clf

    # -- inference ---------------------------------------------------------
    def detect_image(self, image) -> ImageDetection:  # pragma: no cover - needs weights
        ref = self._coerce(image)
        ready, reason = self.available()
        if not ready:
            raise DetectorUnavailableError(reason)

        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        model = self._load_detector()
        img = np.array(Image.open(ref.path).convert("RGB"))

        # Pytorch-Wildlife API shifted across versions; try the common shapes.
        result = None
        for call in (
            lambda: model.single_image_detection(img, img_path=ref.path),
            lambda: model.single_image_detection(img, ref.path),
            lambda: model.single_image_detection(img),
        ):
            try:
                result = call()
                break
            except TypeError:
                continue
        if result is None:
            raise DetectorUnavailableError("single_image_detection call failed for this Pytorch-Wildlife version")

        items = parse_megadetector_result(
            result, threshold=self.detection_threshold, source=self.name
        )

        # Optional species classification of animal crops via SpeciesNet.
        clf = self._load_classifier()
        if clf is not None and items:
            items = self._classify_animals(clf, ref.path, img, result, items)

        h, w = img.shape[0], img.shape[1]
        return ImageDetection(
            image_path=ref.path,
            datetime=ref.datetime,
            width=w,
            height=h,
            items=items,
            detector=self.name,
            raw=None,
        )

    def _classify_animals(self, clf, path, img, result, items):  # pragma: no cover
        """Best-effort species labelling of animal boxes with SpeciesNet.

        SpeciesNet's exact API varies by release; failures here fall back to the
        generic ``Animal`` label rather than aborting detection.
        """
        try:
            preds = clf.predict(instances_dict={"instances": [{"filepath": path}]})
            # SpeciesNet returns a top prediction string like
            # "uuid;class;order;family;genus;species;common name".
            pred_list = preds.get("predictions") if isinstance(preds, dict) else None
            common = None
            if pred_list:
                p0 = pred_list[0]
                label = p0.get("prediction") or ""
                score = p0.get("prediction_score", 0.0)
                if score >= self.classification_threshold and ";" in label:
                    common = label.split(";")[-1].strip() or None
            if common:
                for it in items:
                    if it.category == CATEGORY_ANIMAL:
                        it.label = common.title()
                        it.scientific_name = to_scientific_name(common)
        except Exception:
            pass
        # Ensure animal items at least carry a generic label so they count.
        for it in items:
            if it.category == CATEGORY_ANIMAL and not it.label:
                it.label = "Animal"
                it.scientific_name = "Animalia"
        return items
