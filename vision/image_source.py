"""Image ingestion helpers: folder walking, uploads, and capture timestamps.

Camera-trap timestamps are essential -- the analysis pipeline joins detections
to tide/weather data on ``DateTime``. This module extracts the capture time from
(in order of preference): EXIF ``DateTimeOriginal``, a timestamp embedded in the
filename, then the filesystem modification time.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp")

# Common camera-trap filename timestamp patterns, e.g.
# "IMG_20220121_164500.JPG", "2022-01-21 16-45-00.jpg", "01210001.JPG" (MMDD....)
_FILENAME_DT_PATTERNS = [
    (re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_ ]?(\d{2})[-_:]?(\d{2})[-_:]?(\d{2})"),
     ("Y", "m", "d", "H", "M", "S")),
    (re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})"),
     ("Y", "m", "d")),
]


@dataclass
class ImageRef:
    """A reference to one source image plus its resolved capture time."""

    path: str
    datetime: Optional[datetime] = None
    datetime_source: str = "unknown"  # exif | filename | mtime | unknown

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def iter_image_paths(folder: str, recursive: bool = True) -> List[str]:
    """List image file paths under ``folder``.

    Args:
        folder: Directory to scan.
        recursive: Recurse into sub-directories when ``True``.

    Returns:
        Sorted list of absolute image paths.
    """
    if not folder or not os.path.isdir(folder):
        return []
    out: List[str] = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    out.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            full = os.path.join(folder, f)
            if os.path.isfile(full) and f.lower().endswith(IMAGE_EXTENSIONS):
                out.append(full)
    return sorted(out)


def _parse_filename_datetime(name: str) -> Optional[datetime]:
    """Try to recover a timestamp from a filename."""
    for pattern, fields in _FILENAME_DT_PATTERNS:
        m = pattern.search(name)
        if not m:
            continue
        vals = {k: int(v) for k, v in zip(fields, m.groups())}
        try:
            return datetime(
                vals.get("Y", 1900),
                vals.get("m", 1),
                vals.get("d", 1),
                vals.get("H", 0),
                vals.get("M", 0),
                vals.get("S", 0),
            )
        except ValueError:
            continue
    return None


def _read_exif_datetime(path: str) -> Optional[datetime]:
    """Read EXIF DateTimeOriginal/DateTime using Pillow, if available."""
    try:
        from PIL import Image, ExifTags  # type: ignore
    except Exception:
        return None
    tag_ids = {v: k for k, v in ExifTags.TAGS.items()}
    want = [tag_ids.get("DateTimeOriginal"), tag_ids.get("DateTimeDigitized"), tag_ids.get("DateTime")]
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            # DateTimeOriginal lives in the Exif IFD on many cameras.
            candidates = {}
            try:
                for k, v in exif.items():
                    candidates[k] = v
                ifd = exif.get_ifd(0x8769)  # ExifIFD
                for k, v in ifd.items():
                    candidates[k] = v
            except Exception:
                pass
            for tid in want:
                if tid and tid in candidates and candidates[tid]:
                    raw = str(candidates[tid]).strip()
                    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                        try:
                            return datetime.strptime(raw, fmt)
                        except ValueError:
                            continue
    except Exception:
        return None
    return None


def read_capture_datetime(path: str) -> tuple[Optional[datetime], str]:
    """Resolve an image's capture time.

    Returns a ``(datetime, source)`` tuple where ``source`` is one of
    ``exif``, ``filename``, ``mtime`` or ``unknown``.
    """
    dt = _read_exif_datetime(path)
    if dt is not None:
        return dt, "exif"
    dt = _parse_filename_datetime(os.path.basename(path))
    if dt is not None:
        return dt, "filename"
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)), "mtime"
    except Exception:
        return None, "unknown"


def collect_images(
    folder: str,
    recursive: bool = True,
    limit: Optional[int] = None,
    resolve_time: bool = True,
) -> List[ImageRef]:
    """Collect images from a folder as :class:`ImageRef` objects with times.

    Args:
        folder: Directory containing the camera-trap images.
        recursive: Recurse into sub-directories.
        limit: Optionally cap the number of images (useful for quick tests).
        resolve_time: Resolve each image's capture datetime.

    Returns:
        A list of :class:`ImageRef`.
    """
    paths = iter_image_paths(folder, recursive=recursive)
    if limit is not None:
        paths = paths[:limit]
    refs: List[ImageRef] = []
    for p in paths:
        if resolve_time:
            dt, src = read_capture_datetime(p)
        else:
            dt, src = None, "unknown"
        refs.append(ImageRef(path=p, datetime=dt, datetime_source=src))
    return refs


def save_uploaded_files(uploaded_files, dest_dir: str) -> List[str]:
    """Persist Streamlit ``UploadedFile`` objects to ``dest_dir``.

    Returns the list of written paths. Safe to call with an empty/None input.
    """
    if not uploaded_files:
        return []
    os.makedirs(dest_dir, exist_ok=True)
    out: List[str] = []
    for uf in uploaded_files:
        name = getattr(uf, "name", None) or f"upload_{len(out)}.jpg"
        dest = os.path.join(dest_dir, os.path.basename(name))
        try:
            data = uf.getbuffer()
        except Exception:
            data = uf.read()
        with open(dest, "wb") as fh:
            fh.write(data)
        out.append(dest)
    return out
