"""Shared detection data structures and species-name canonicalization.

Every detector in this package returns :class:`ImageDetection` objects composed
of :class:`DetectionItem` records. Keeping a single schema means the demo
detector, the local MegaDetector/SpeciesNet path, and the LLM path are all
interchangeable and directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Detection categories (mirrors the MegaDetector top-level classes)
# ---------------------------------------------------------------------------
CATEGORY_ANIMAL = "animal"
CATEGORY_PERSON = "person"
CATEGORY_VEHICLE = "vehicle"
CATEGORY_EMPTY = "empty"

ALL_CATEGORIES = (CATEGORY_ANIMAL, CATEGORY_PERSON, CATEGORY_VEHICLE, CATEGORY_EMPTY)


# ---------------------------------------------------------------------------
# Common-name -> scientific-name mapping
#
# Extends the mapping historically embedded in ``data_loader`` with the estuary
# / tide-gate species relevant to the Coos Bay (Willanch / Palouse) sites. Keys
# are lower-cased and stripped for lookup.
# ---------------------------------------------------------------------------
COMMON_TO_SCIENTIFIC: Dict[str, str] = {
    "unknown": "Unknown",
    "brant": "Branta bernicla",
    "canada goose": "Branta canadensis",
    "canada geese": "Branta canadensis",
    "cackling goose": "Branta hutchinsii",
    "great egret": "Ardea alba",
    "great blue heron": "Ardea herodias",
    "belted kingfisher": "Megaceryle alcyon",
    "double-crested cormorant": "Nannopterum auritus",
    "double crested cormorant": "Nannopterum auritus",
    "pelagic cormorant": "Urile pelagicus",
    "cormorant": "Phalacrocoracidae",
    "river otter": "Lontra canadensis",
    "columbian black-tailed deer": "Odocoileus hemionus columbianus",
    "black-tailed deer": "Odocoileus hemionus columbianus",
    "black tailed deer": "Odocoileus hemionus columbianus",
    "mule deer": "Odocoileus hemionus",
    "turkey vulture": "Cathartes aura",
    "red-necked grebe": "Podiceps grisegena",
    "common loon": "Gavia immer",
    "common merganser": "Mergus merganser",
    "hooded merganser": "Lophodytes cucullatus",
    "bufflehead": "Bucephala albeola",
    "mallard": "Anas platyrhynchos",
    "american crow": "Corvus brachyrhynchos",
    "american wigeon": "Mareca americana",
    "northern pintail": "Anas acuta",
    "green-winged teal": "Anas crecca",
    "gadwall": "Mareca strepera",
    "western sandpiper": "Calidris mauri",
    "dunlin": "Calidris alpina",
    "killdeer": "Charadrius vociferus",
    "greater yellowlegs": "Tringa melanoleuca",
    "spotted sandpiper": "Actitis macularius",
    "western gull": "Larus occidentalis",
    "glaucous-winged gull": "Larus glaucescens",
    "gull": "Larus",
    "osprey": "Pandion haliaetus",
    "bald eagle": "Haliaeetus leucocephalus",
    "northern harrier": "Circus hudsonius",
    "raccoon": "Procyon lotor",
    "coyote": "Canis latrans",
    "north american beaver": "Castor canadensis",
    "beaver": "Castor canadensis",
    "nutria": "Myocastor coypus",
    "muskrat": "Ondatra zibethicus",
    "harbor seal": "Phoca vitulina",
    "great horned owl": "Bubo virginianus",
    "song sparrow": "Melospiza melodia",
    "american robin": "Turdus migratorius",
    "human": "Homo sapiens",
    "person": "Homo sapiens",
}


def to_scientific_name(common_name: Optional[str]) -> str:
    """Return the canonical scientific name for a common name.

    Falls back to a Title-Cased version of the input when the species is not in
    the mapping (matching the historical behaviour of ``data_loader``). ``None``
    or blank input returns an empty string.

    Args:
        common_name: A common species name such as ``"Great Blue Heron"``.

    Returns:
        The canonical scientific name, or a Title-Cased fallback.
    """
    if common_name is None:
        return ""
    key = str(common_name).strip().lower()
    if not key or key == "nan":
        return ""
    if key in COMMON_TO_SCIENTIFIC:
        return COMMON_TO_SCIENTIFIC[key]
    return str(common_name).strip().title()


@dataclass
class DetectionItem:
    """A single detected subject within an image.

    Attributes:
        category: One of ``animal``, ``person``, ``vehicle`` or ``empty``.
        label: Human-readable label (common name for animals). May be blank
            when a detector localizes an animal but cannot classify it.
        scientific_name: Canonical scientific name (filled from ``label`` when
            not provided by the detector).
        count: Number of individuals this item represents (default 1).
        confidence: Detector confidence in ``[0, 1]`` when available.
        bbox: Optional normalized bounding box ``(x, y, w, h)`` in ``[0, 1]``.
        activity: Optional coarse behaviour label (e.g. ``"foraging"``).
        distance: Optional distance/zone annotation.
        source: Name of the detector that produced this item.
    """

    category: str = CATEGORY_ANIMAL
    label: str = ""
    scientific_name: str = ""
    count: int = 1
    confidence: Optional[float] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    activity: Optional[str] = None
    distance: Optional[str] = None
    source: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.scientific_name and self.label:
            self.scientific_name = to_scientific_name(self.label)
        try:
            self.count = int(self.count)
        except (TypeError, ValueError):
            self.count = 1
        if self.count < 1:
            self.count = 1

    @property
    def is_animal(self) -> bool:
        return self.category == CATEGORY_ANIMAL


@dataclass
class ImageDetection:
    """The full detection result for one image.

    Attributes:
        image_path: Absolute or relative path to the source image.
        datetime: Capture timestamp (from EXIF/filename/mtime) if known.
        width: Image width in pixels, if known.
        height: Image height in pixels, if known.
        items: List of detected subjects. An empty list means "no detections".
        detector: Name of the detector used.
        error: Error message if detection failed for this image.
        raw: Optional raw detector payload (kept for debugging/audit).
    """

    image_path: str
    datetime: Optional[datetime] = None
    width: Optional[int] = None
    height: Optional[int] = None
    items: List[DetectionItem] = field(default_factory=list)
    detector: Optional[str] = None
    error: Optional[str] = None
    raw: Optional[Any] = None

    @property
    def animals(self) -> List[DetectionItem]:
        """Return only the animal detections (people/vehicles excluded)."""
        return [it for it in self.items if it.category == CATEGORY_ANIMAL and it.label]

    @property
    def has_animal(self) -> bool:
        return len(self.animals) > 0

    @property
    def total_count(self) -> int:
        return sum(it.count for it in self.animals)

    def top_species(self) -> Optional[str]:
        """Return the highest-count / highest-confidence animal label."""
        if not self.animals:
            return None
        best = max(
            self.animals,
            key=lambda it: (it.count, it.confidence if it.confidence is not None else 0.0),
        )
        return best.label

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["datetime"] = self.datetime.isoformat() if self.datetime else None
        return d
