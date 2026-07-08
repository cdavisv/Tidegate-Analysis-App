"""A dependency-free synthetic detector for end-to-end testing.

``DemoDetector`` fabricates *deterministic* but plausible detections so the full
image -> camera-CSV -> analysis pipeline can be exercised without downloading or
running any heavy models. Results are seeded from a hash of the image path, so a
given image always yields the same detection (reproducible demos and tests).

This is clearly synthetic data and must not be used for real ecological
inference -- the UI labels it as such.
"""

from __future__ import annotations

import hashlib
import random
from typing import List, Optional, Sequence

from .base import Detector
from .schema import (
    CATEGORY_ANIMAL,
    CATEGORY_PERSON,
    DetectionItem,
    ImageDetection,
    to_scientific_name,
)

# Species pool weighted toward the Coos Bay tide-gate community. Weights are
# relative frequencies used to make the synthetic distribution look estuary-like.
_DEFAULT_SPECIES_WEIGHTS = {
    "Canada Goose": 22,
    "Mallard": 16,
    "Great Blue Heron": 12,
    "American Crow": 10,
    "Great Egret": 8,
    "Belted Kingfisher": 6,
    "Bufflehead": 6,
    "Double-crested Cormorant": 5,
    "Common Merganser": 4,
    "River Otter": 3,
    "Black-tailed Deer": 3,
    "Turkey Vulture": 2,
    "Western Gull": 3,
    "Northern Pintail": 2,
    "Raccoon": 2,
}


class DemoDetector(Detector):
    """Deterministic synthetic detector (no external dependencies)."""

    name = "demo"
    description = "Synthetic detector for testing the pipeline without models."

    def __init__(
        self,
        detection_probability: float = 0.30,
        person_probability: float = 0.02,
        max_species_per_image: int = 2,
        species_weights: Optional[dict] = None,
        seed: int = 1234,
    ) -> None:
        self.detection_probability = float(detection_probability)
        self.person_probability = float(person_probability)
        self.max_species_per_image = int(max_species_per_image)
        self.species_weights = dict(species_weights or _DEFAULT_SPECIES_WEIGHTS)
        self.seed = seed
        self._species = list(self.species_weights.keys())
        self._weights = list(self.species_weights.values())

    def available(self) -> tuple[bool, str]:
        return True, "ready (synthetic)"

    def _rng(self, key: str) -> random.Random:
        h = hashlib.sha256(f"{self.seed}:{key}".encode("utf-8")).hexdigest()
        return random.Random(int(h[:16], 16))

    def detect_image(self, image) -> ImageDetection:
        ref = self._coerce(image)
        rng = self._rng(ref.path)
        items: List[DetectionItem] = []

        roll = rng.random()
        if roll < self.person_probability:
            items.append(
                DetectionItem(
                    category=CATEGORY_PERSON,
                    label="Human",
                    scientific_name=to_scientific_name("human"),
                    count=1,
                    confidence=round(rng.uniform(0.70, 0.98), 3),
                    source=self.name,
                )
            )
        elif roll < self.person_probability + self.detection_probability:
            n_species = 1
            if self.max_species_per_image > 1 and rng.random() < 0.25:
                n_species = rng.randint(2, self.max_species_per_image)
            chosen = self._weighted_sample(rng, n_species)
            for sp in chosen:
                items.append(
                    DetectionItem(
                        category=CATEGORY_ANIMAL,
                        label=sp,
                        scientific_name=to_scientific_name(sp),
                        count=rng.randint(1, 4),
                        confidence=round(rng.uniform(0.55, 0.97), 3),
                        activity=rng.choice(["foraging", "resting", "moving", "flying", None]),
                        source=self.name,
                    )
                )
        # else: empty image (no items)

        return ImageDetection(
            image_path=ref.path,
            datetime=ref.datetime,
            items=items,
            detector=self.name,
        )

    def _weighted_sample(self, rng: random.Random, k: int) -> List[str]:
        """Sample ``k`` distinct species by weight (no replacement)."""
        species = list(self._species)
        weights = list(self._weights)
        picked: List[str] = []
        for _ in range(min(k, len(species))):
            total = sum(weights)
            r = rng.uniform(0, total)
            upto = 0.0
            for idx, w in enumerate(weights):
                upto += w
                if upto >= r:
                    picked.append(species.pop(idx))
                    weights.pop(idx)
                    break
        return picked
