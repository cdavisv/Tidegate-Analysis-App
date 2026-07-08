"""Multimodal-LLM detector using OpenAI GPT vision models.

Sometimes a general multimodal model reasons about a tricky camera-trap frame
better than a single specialized CV model. This detector sends each image to an
OpenAI GPT model and asks for a strict-JSON list of species with counts.

* Model is **user-configurable** (default :data:`DEFAULT_VISION_MODEL`). Any
  vision-capable OpenAI model string works; set it in the UI or via the
  ``OPENAI_VISION_MODEL`` environment variable.
* The API key is read from the ``OPENAI_API_KEY`` environment variable or passed
  explicitly. It is never logged.
* JSON parsing is isolated in :func:`parse_llm_json` so it can be unit-tested
  without any network access, and a fake client can be injected for testing the
  request flow.

Cost note: each image is one API call. Large batches cost money and take time;
the UI exposes a max-image cap and downsizes images before upload.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any, Dict, List, Optional

from .base import Detector, DetectorUnavailableError
from .schema import CATEGORY_ANIMAL, CATEGORY_PERSON, DetectionItem, ImageDetection, to_scientific_name

# Current OpenAI flagship multimodal model (2026). User-overridable.
DEFAULT_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-5.5")

_PROMPT = (
    "You are an expert wildlife biologist analyzing a single camera-trap image "
    "from a tidal estuary / tide-gate site in coastal Oregon, USA. "
    "Identify every distinct animal visible. Prefer species-level identification; "
    "if unsure, give the most specific taxon you are confident about. "
    "Count individuals of each species. Ignore vegetation, water and equipment.\n\n"
    "Respond with STRICT JSON only, no prose, in exactly this shape:\n"
    '{"detections": [{"common_name": str, "scientific_name": str, "count": int, '
    '"confidence": number between 0 and 1, "activity": str or null, '
    '"category": "animal" | "person" | "vehicle"}]}\n'
    'If no animals (or people/vehicles) are present, return {"detections": []}.'
)


def parse_llm_json(text: str, source: str = "openai") -> List[DetectionItem]:
    """Parse the model's JSON response into :class:`DetectionItem` objects.

    Tolerant of common wrapping (markdown code fences, leading prose). Returns an
    empty list for an explicitly empty result or unparseable output.

    Args:
        text: Raw text returned by the model.
        source: Detector name stamped onto each item.
    """
    if not text:
        return []
    cleaned = text.strip()
    # Strip ```json ... ``` fences if present.
    cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    data: Any = None
    try:
        data = json.loads(cleaned)
    except Exception:
        # Last resort: grab the first {...} or [...] blob.
        m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except Exception:
                return []
    if data is None:
        return []

    if isinstance(data, dict):
        rows = data.get("detections", data.get("animals", []))
    elif isinstance(data, list):
        rows = data
    else:
        return []

    items: List[DetectionItem] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        common = (row.get("common_name") or row.get("species") or "").strip()
        category = (row.get("category") or CATEGORY_ANIMAL).strip().lower()
        if category not in ("animal", "person", "vehicle"):
            category = CATEGORY_ANIMAL
        if category == CATEGORY_ANIMAL and not common:
            continue
        conf = row.get("confidence")
        try:
            conf = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf = None
        sci = (row.get("scientific_name") or "").strip() or to_scientific_name(common)
        items.append(
            DetectionItem(
                category=category,
                label=common or category.title(),
                scientific_name=sci,
                count=row.get("count", 1),
                confidence=conf,
                activity=(row.get("activity") or None),
                source=source,
            )
        )
    return items


class OpenAIVisionDetector(Detector):
    """Species identification via an OpenAI multimodal model."""

    name = "openai"
    description = "OpenAI GPT vision model identifies species directly from images."

    def __init__(
        self,
        model: str = DEFAULT_VISION_MODEL,
        api_key: Optional[str] = None,
        max_image_px: int = 1024,
        detail: str = "auto",
        client: Any = None,
    ) -> None:
        self.model = model or DEFAULT_VISION_MODEL
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_image_px = int(max_image_px)
        self.detail = detail
        self._client = client  # allow injection for testing

    # -- readiness ---------------------------------------------------------
    def available(self) -> tuple[bool, str]:
        try:
            import openai  # noqa: F401
        except Exception:
            return False, "openai package not installed. Install with: pip install openai"
        if self._client is None and not self.api_key:
            return False, "No OpenAI API key. Set OPENAI_API_KEY or enter a key in the UI."
        return True, "ready"

    def _get_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI  # type: ignore

        self._client = OpenAI(api_key=self.api_key)
        return self._client

    # -- image encoding ----------------------------------------------------
    def _encode_image(self, path: str) -> str:
        """Return a base64 ``data:`` URL, downscaling large images if possible."""
        try:
            from PIL import Image  # type: ignore

            with Image.open(path) as im:
                im = im.convert("RGB")
                if max(im.size) > self.max_image_px:
                    ratio = self.max_image_px / float(max(im.size))
                    im = im.resize((int(im.size[0] * ratio), int(im.size[1] * ratio)))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=85)
                data = buf.getvalue()
        except Exception:
            with open(path, "rb") as fh:
                data = fh.read()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _build_messages(self, data_url: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": self.detail}},
                ],
            }
        ]

    def _call_model(self, messages: List[Dict[str, Any]]) -> str:
        """Call the chat completions API and return the message text."""
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    # -- inference ---------------------------------------------------------
    def detect_image(self, image) -> ImageDetection:
        ref = self._coerce(image)
        ready, reason = self.available()
        if not ready:
            raise DetectorUnavailableError(reason)
        data_url = self._encode_image(ref.path)
        messages = self._build_messages(data_url)
        text = self._call_model(messages)
        items = parse_llm_json(text, source=self.name)
        return ImageDetection(
            image_path=ref.path,
            datetime=ref.datetime,
            items=items,
            detector=self.name,
            raw=text,
        )
