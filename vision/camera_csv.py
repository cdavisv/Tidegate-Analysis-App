"""Convert detection results into the wide-format *camera CSV*.

The analysis pipeline (``data_loader.load_and_prepare_camera_data``) expects a
camera CSV shaped like the project's hand-labelled files:

    relative_path, Full Path, Corrected, absolute_path, data_type,
    Species 1, Species 2, ..., Species N,
    Species 1 Count, Species 2 Count, ..., Species N Count,
    Species 1 Distance, ..., Species 1 Activity, ...,
    Weather, Water Conditions, Water Notes, Temp [C], Notes, DateTime

Key rules that this writer honours (so the downstream loader behaves correctly):

* One row per image.
* Images with animals populate ``Species 1..k`` and the matching
  ``Species k Count`` columns (one column group per distinct species).
* Images with **no** animal leave the ``Species`` columns blank -- the loader
  treats these as valid "no animals detected" camera-activity records, which the
  dual-framework analysis depends on.
* ``DateTime`` is written in ISO-8601, which ``pandas.to_datetime`` parses
  reliably.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from .schema import ImageDetection, DetectionItem

# Base (non-species) columns emitted for schema fidelity with the hand-labelled
# files. ``DateTime`` and the path columns are the ones the loader actually uses.
_BASE_COLUMNS = [
    "relative_path",
    "Full Path",
    "Corrected",
    "absolute_path",
    "data_type",
]
_TRAILING_COLUMNS = [
    "Weather",
    "Water Conditions",
    "Water Notes",
    "Temp [C]",
    "Notes",
    "DateTime",
]


def _aggregate_species(det: ImageDetection) -> List[DetectionItem]:
    """Collapse an image's animal items into one entry per distinct species.

    Counts for the same species are summed; the max confidence and the first
    non-empty activity/distance are retained. Ordering is by descending count so
    the dominant species lands in ``Species 1``.
    """
    grouped: dict[str, DetectionItem] = {}
    for it in det.animals:
        key = (it.label or "").strip().lower()
        if not key:
            continue
        if key not in grouped:
            grouped[key] = DetectionItem(
                category=it.category,
                label=it.label.strip(),
                scientific_name=it.scientific_name,
                count=it.count,
                confidence=it.confidence,
                activity=it.activity,
                distance=it.distance,
                source=it.source,
            )
        else:
            g = grouped[key]
            g.count += it.count
            if it.confidence is not None:
                g.confidence = max(g.confidence or 0.0, it.confidence)
            if not g.activity and it.activity:
                g.activity = it.activity
            if not g.distance and it.distance:
                g.distance = it.distance
    return sorted(
        grouped.values(),
        key=lambda x: (x.count, x.confidence or 0.0),
        reverse=True,
    )


def _fmt_dt(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    return pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S")


def detections_to_camera_df(
    detections: Sequence[ImageDetection],
    *,
    corrected_prefix: str = "",
    include_confidence_notes: bool = True,
) -> pd.DataFrame:
    """Build the wide-format camera DataFrame from detection results.

    Args:
        detections: The per-image detection results.
        corrected_prefix: Optional value for the ``Corrected`` column (mirrors
            the hand-labelled files, which stored the image-root prefix there).
        include_confidence_notes: When ``True``, per-image detector confidence
            and any error are summarized into the ``Notes`` column.

    Returns:
        A DataFrame with a stable column order, ready to ``to_csv`` and feed to
        ``data_loader.load_and_prepare_camera_data``.
    """
    rows: List[dict] = []
    max_species = 1  # always emit at least Species 1 so the loader finds it

    for det in detections:
        species = _aggregate_species(det)
        max_species = max(max_species, len(species))

        abspath = os.path.abspath(det.image_path) if det.image_path else ""
        row = {
            "relative_path": os.path.basename(det.image_path) if det.image_path else "",
            "Full Path": det.image_path or "",
            "Corrected": corrected_prefix,
            "absolute_path": abspath,
            "data_type": "img",
            "Weather": "",
            "Water Conditions": "",
            "Water Notes": "",
            "Temp [C]": "",
            "DateTime": _fmt_dt(det.datetime),
        }

        note_bits: List[str] = []
        if det.error:
            note_bits.append(f"detection_error: {det.error}")
        elif not species:
            note_bits.append("No animals detected")
        if include_confidence_notes and species:
            confs = [f"{s.label}:{s.confidence:.2f}" for s in species if s.confidence is not None]
            if confs:
                note_bits.append("conf " + ", ".join(confs))
        if det.detector:
            note_bits.append(f"[{det.detector}]")
        row["Notes"] = "; ".join(note_bits)

        for i, sp in enumerate(species, start=1):
            row[f"Species {i}"] = sp.label
            row[f"Species {i} Count"] = sp.count
            if sp.distance is not None:
                row[f"Species {i} Distance"] = sp.distance
            if sp.activity is not None:
                row[f"Species {i} Activity"] = sp.activity
        rows.append(row)

    # Assemble a stable, loader-friendly column order.
    species_cols: List[str] = []
    for i in range(1, max_species + 1):
        species_cols.append(f"Species {i}")
    for i in range(1, max_species + 1):
        species_cols.append(f"Species {i} Count")
    for i in range(1, max_species + 1):
        species_cols.extend([f"Species {i} Distance", f"Species {i} Activity"])

    ordered = _BASE_COLUMNS + species_cols + _TRAILING_COLUMNS
    df = pd.DataFrame(rows)
    for col in ordered:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[ordered]
    return df


def detections_to_long_df(detections: Iterable[ImageDetection]) -> pd.DataFrame:
    """Return a tidy long-format table (one row per detected animal item).

    Useful for previews, review UIs and detector-vs-detector comparison.
    """
    records: List[dict] = []
    for det in detections:
        if det.animals:
            for it in det.animals:
                records.append(
                    {
                        "image": os.path.basename(det.image_path) if det.image_path else "",
                        "DateTime": det.datetime,
                        "category": it.category,
                        "species": it.label,
                        "scientific_name": it.scientific_name,
                        "count": it.count,
                        "confidence": it.confidence,
                        "activity": it.activity,
                        "detector": det.detector,
                        "error": det.error,
                    }
                )
        else:
            records.append(
                {
                    "image": os.path.basename(det.image_path) if det.image_path else "",
                    "DateTime": det.datetime,
                    "category": "empty" if not det.error else "error",
                    "species": None,
                    "scientific_name": None,
                    "count": 0,
                    "confidence": None,
                    "activity": None,
                    "detector": det.detector,
                    "error": det.error,
                }
            )
    return pd.DataFrame.from_records(records)


def save_camera_csv(
    detections: Sequence[ImageDetection],
    path: str,
    **kwargs,
) -> str:
    """Write the wide-format camera CSV to ``path`` and return the path."""
    df = detections_to_camera_df(detections, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    return path
