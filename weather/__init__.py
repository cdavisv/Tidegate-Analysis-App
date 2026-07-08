"""Self-contained weather import package for the Tidegate wildlife-analysis app.

This package fetches free weather (and tide-gauge) data, normalizes it into a
single standardized schema, and merges it into the app's sensor datasets. It
depends only on ``pandas``, ``numpy`` and ``requests``.

Standardized schema (see :data:`SCHEMA_COLUMNS`)
------------------------------------------------
``DateTime`` (tz-naive), ``Air_Temp_C``, ``Wind_Speed_km_h``,
``Precipitation_cm``, ``Barometric_Pressure_mbar``, ``Humidity_pct``,
``Solar_Radiation_W_m2`` (optional), ``Water_Level_m`` (optional).

Quick start
-----------
.. code-block:: python

    import weather

    src = weather.get_source("open-meteo")
    df = src.fetch(lat=43.345, lon=-124.323, start="2023-06-01", end="2023-06-07")

    merged = weather.merge_weather_into_sensor(sensor_df, df, prefer="sensor")

Sources are also available directly (:data:`SOURCES`) and every public name
listed in ``__all__`` is importable from ``weather`` itself.
"""

from __future__ import annotations

import logging

# Exceptions + schema + normalization / merge helpers
from .normalize import (
    NUMERIC_COLUMNS,
    SCHEMA_COLUMNS,
    WeatherConfigError,
    WeatherError,
    WeatherFetchError,
    empty_weather_frame,
    load_weather_csv,
    merge_weather_into_sensor,
    resample_hourly,
    to_sensor_frame,
)

# Source connectors + registry
from .sources import (
    NOAASource,
    OpenMeteoSource,
    SOURCES,
    SynopticSource,
    WeatherSource,
    get_source,
)

# A library should not configure logging handlers itself.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "1.0.0"

__all__ = [
    # exceptions
    "WeatherError",
    "WeatherFetchError",
    "WeatherConfigError",
    # schema
    "SCHEMA_COLUMNS",
    "NUMERIC_COLUMNS",
    # sources + registry
    "WeatherSource",
    "OpenMeteoSource",
    "NOAASource",
    "SynopticSource",
    "SOURCES",
    "get_source",
    # csv / normalize / merge
    "load_weather_csv",
    "merge_weather_into_sensor",
    "resample_hourly",
    "to_sensor_frame",
    "empty_weather_frame",
]
