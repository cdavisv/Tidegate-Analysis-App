"""Central configuration for the Tidegate Analysis App.

Holds site metadata (used to fetch weather/tide data), default model names, and
output locations. Secrets (API keys) are read from the environment only and are
never persisted to the optional ``config.json`` file.

Load order (later overrides earlier):

1. Built-in defaults (Coos Bay / Willanch tide-gate area, Oregon).
2. ``config.json`` in the project root, if present.
3. Environment variables (``TIDEGATE_LAT``, ``TIDEGATE_LON``,
   ``OPENAI_VISION_MODEL``, ``OPENAI_API_KEY``, ``SYNOPTIC_TOKEN``).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

CONFIG_FILENAME = "config.json"


@dataclass
class AppConfig:
    """Runtime configuration for the app."""

    # --- Field site (defaults: Willanch Slough tide gate, Coos Bay, OR) ---
    site_name: str = "Willanch Slough Tide Gate (Coos Bay, OR)"
    latitude: float = 43.45
    longitude: float = -124.20
    timezone: str = "America/Los_Angeles"

    # --- NOAA CO-OPS tide station (Charleston, OR = 9432780) ---
    noaa_tide_station: str = "9432780"

    # --- Detection defaults ---
    default_detector: str = "demo"
    openai_vision_model: str = "gpt-5.5"
    megadetector_version: str = "MDV6-yolov9-c"
    detection_threshold: float = 0.2

    # --- Weather defaults ---
    default_weather_source: str = "open-meteo"

    # --- Paths ---
    output_dir: str = "output_plots"

    # --- Secrets (populated from env; not written to config.json) ---
    openai_api_key: Optional[str] = field(default=None, repr=False)
    synoptic_token: Optional[str] = field(default=None, repr=False)

    def public_dict(self) -> dict:
        """Return config without secrets (safe to display/serialize)."""
        d = asdict(self)
        d.pop("openai_api_key", None)
        d.pop("synoptic_token", None)
        return d

    def save(self, path: Optional[str] = None) -> str:
        """Persist non-secret config to ``config.json``."""
        path = path or CONFIG_FILENAME
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.public_dict(), fh, indent=2)
        return path


def _apply_env(cfg: AppConfig) -> AppConfig:
    def _f(name, cast, cur):
        val = os.environ.get(name)
        if val is None or val == "":
            return cur
        try:
            return cast(val)
        except (TypeError, ValueError):
            return cur

    cfg.latitude = _f("TIDEGATE_LAT", float, cfg.latitude)
    cfg.longitude = _f("TIDEGATE_LON", float, cfg.longitude)
    cfg.timezone = os.environ.get("TIDEGATE_TZ", cfg.timezone)
    cfg.site_name = os.environ.get("TIDEGATE_SITE", cfg.site_name)
    cfg.openai_vision_model = os.environ.get("OPENAI_VISION_MODEL", cfg.openai_vision_model)
    cfg.openai_api_key = os.environ.get("OPENAI_API_KEY", cfg.openai_api_key)
    cfg.synoptic_token = os.environ.get("SYNOPTIC_TOKEN", cfg.synoptic_token)
    return cfg


def load_config(path: Optional[str] = None) -> AppConfig:
    """Build an :class:`AppConfig` from defaults + ``config.json`` + env."""
    cfg = AppConfig()
    cfg_path = path or CONFIG_FILENAME
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for k, v in data.items():
                if hasattr(cfg, k) and k not in ("openai_api_key", "synoptic_token"):
                    setattr(cfg, k, v)
        except Exception:
            pass
    return _apply_env(cfg)
