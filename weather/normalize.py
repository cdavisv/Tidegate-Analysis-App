"""Normalization, CSV import, and merge utilities for the weather package.

This module defines the *standardized weather schema* used across the whole
package and provides the pure-``pandas`` helpers that every data source and the
CSV importer funnel their output through. Keeping the schema, the exception
hierarchy, and the frame-shaping logic here (with no network dependencies) lets
:mod:`weather.sources` and :mod:`weather` import from a single, side-effect-free
foundation.

Standardized schema
-------------------
Every source connector and :func:`load_weather_csv` return a
:class:`pandas.DataFrame` with the following columns (missing variables are
filled with ``NaN``):

======================== ========= ===========================================
Column                   dtype     Meaning
======================== ========= ===========================================
``DateTime``             datetime  Observation time, tz-naive (see note below)
``Air_Temp_C``           float     Air temperature in degrees Celsius
``Wind_Speed_km_h``      float     Wind speed in kilometres per hour
``Precipitation_cm``     float     Precipitation for the interval, centimetres
``Barometric_Pressure_mbar`` float Barometric/surface pressure in millibar (hPa)
``Humidity_pct``         float     Relative humidity, 0-100 %
``Solar_Radiation_W_m2`` float     Shortwave/solar radiation in W/m^2 (optional)
``Water_Level_m``        float     Water level in metres (NOAA CO-OPS, optional)
======================== ========= ===========================================

Time zone assumption
--------------------
``DateTime`` is always **tz-naive**. The *wall-clock meaning* depends on the
producer and is documented on each source:

* :func:`load_weather_csv` -- taken verbatim from the file and assumed to be
  local station time. If the file carries an explicit tz offset the offset is
  dropped while preserving the wall-clock value.
* ``OpenMeteoSource`` -- local station time (Open-Meteo ``timezone=auto``).
* ``NOAASource`` -- UTC (both NWS observations and CO-OPS use GMT).
* ``SynopticSource`` -- UTC (``obtimezone=utc``).
"""

from __future__ import annotations

import io
import logging
import re
from typing import Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "WeatherError",
    "WeatherFetchError",
    "WeatherConfigError",
    "SCHEMA_COLUMNS",
    "NUMERIC_COLUMNS",
    "to_sensor_frame",
    "empty_weather_frame",
    "load_weather_csv",
    "merge_weather_into_sensor",
    "resample_hourly",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class WeatherError(Exception):
    """Base class for all errors raised by the :mod:`weather` package."""


class WeatherFetchError(WeatherError):
    """Raised when a network fetch fails or returns an unusable payload.

    Network calls never leak a bare :class:`requests.RequestException`; they are
    wrapped in this exception with a human-readable, context-rich message.
    """


class WeatherConfigError(WeatherError):
    """Raised on configuration problems (missing token, unknown source, ...)."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA_COLUMNS = [
    "DateTime",
    "Air_Temp_C",
    "Wind_Speed_km_h",
    "Precipitation_cm",
    "Barometric_Pressure_mbar",
    "Humidity_pct",
    "Solar_Radiation_W_m2",
    "Water_Level_m",
]

#: Numeric (non-``DateTime``) columns of the schema, in canonical order.
NUMERIC_COLUMNS = SCHEMA_COLUMNS[1:]


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------
def _coerce_naive_datetime(values) -> pd.Series:
    """Coerce a column of timestamps to a tz-naive ``datetime64[ns]`` series.

    Values that already carry a timezone (either a homogeneous tz-aware dtype or
    a mix of offsets) are converted to a naive representation. Homogeneous
    tz-aware input has its offset dropped while preserving the wall-clock value
    (via :meth:`~pandas.Series.dt.tz_localize`); genuinely mixed offsets are
    normalized through UTC so the result is always a valid datetime column.

    Args:
        values: Anything accepted by :func:`pandas.to_datetime` (list, Series,
            array of strings/timestamps).

    Returns:
        A tz-naive ``datetime64[ns]`` :class:`pandas.Series`. Unparseable
        entries become ``NaT``.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    parsed = pd.to_datetime(series, errors="coerce")
    dtype = getattr(parsed, "dtype", None)
    if isinstance(dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_localize(None)
    if dtype == object:
        # Mixed tz offsets survive as an object column; normalize through UTC.
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            return parsed.dt.tz_localize(None)
    return parsed


# ---------------------------------------------------------------------------
# Frame shaping
# ---------------------------------------------------------------------------
def to_sensor_frame(
    df: pd.DataFrame,
    *,
    sort: bool = True,
    drop_invalid_datetime: bool = True,
) -> pd.DataFrame:
    """Coerce an arbitrary frame into the standardized weather schema.

    Ensures every schema column is present (adding ``NaN`` columns for anything
    absent), casts numeric columns to ``float64`` and ``DateTime`` to a tz-naive
    ``datetime64[ns]``, drops non-schema columns, and returns the columns in
    canonical order. This is the single funnel used by all sources and by
    :func:`load_weather_csv`.

    Args:
        df: Input frame. Must contain a ``DateTime`` column.
        sort: If ``True`` (default) sort ascending by ``DateTime``.
        drop_invalid_datetime: If ``True`` (default) drop rows whose
            ``DateTime`` could not be parsed (``NaT``).

    Returns:
        A new :class:`pandas.DataFrame` limited to :data:`SCHEMA_COLUMNS`.

    Raises:
        WeatherError: If ``df`` is not a DataFrame or lacks a ``DateTime``
            column.
    """
    if not isinstance(df, pd.DataFrame):
        raise WeatherError("to_sensor_frame() expects a pandas DataFrame.")
    if "DateTime" not in df.columns:
        raise WeatherError("Input frame is missing the required 'DateTime' column.")

    out = df.copy()
    out["DateTime"] = _coerce_naive_datetime(out["DateTime"])
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        else:
            out[col] = np.nan
    out = out[SCHEMA_COLUMNS]

    if drop_invalid_datetime:
        out = out[out["DateTime"].notna()]
    if sort:
        out = out.sort_values("DateTime")
    return out.reset_index(drop=True)


def empty_weather_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the standardized schema and dtypes."""
    frame = pd.DataFrame({col: pd.Series(dtype="float64") for col in NUMERIC_COLUMNS})
    frame.insert(0, "DateTime", pd.Series(dtype="datetime64[ns]"))
    return frame


# ---------------------------------------------------------------------------
# CSV import
# ---------------------------------------------------------------------------
_BRACKET_RE = re.compile(r"\[.*?\]|\(.*?\)")
_NONALNUM_RE = re.compile(r"[^a-z0-9]")
_UNIT_RE = re.compile(r"[\[(]([^\])]*)[\])]")


def _norm_key(name: object) -> str:
    """Normalize a header for fuzzy matching: strip BOM/units/punctuation.

    ``"Air Temp [C]"`` -> ``"airtemp"``; ``"Air_Temp_C"`` -> ``"airtempc"``;
    ``"Wind Speed [km/h]"`` -> ``"windspeed"``.
    """
    text = str(name).replace("﻿", "").strip().lower()
    text = _BRACKET_RE.sub("", text)
    return _NONALNUM_RE.sub("", text)


def _extract_unit(name: object) -> str:
    """Extract the bracketed/parenthesized unit token from a header, lowercased.

    ``"Wind Speed [km/h]"`` -> ``"km/h"``; ``"Air Temp [C]"`` -> ``"c"``. Returns
    an empty string when no unit annotation is present.
    """
    match = _UNIT_RE.search(str(name))
    return match.group(1).strip().lower() if match else ""


# Canonical target -> ordered list of (predicate) rules is expressed inline in
# _match_canonical to keep the exclusion logic readable.
def _match_canonical(norm: str) -> Optional[str]:
    """Map a normalized header to a schema column, or ``None`` if no match."""
    if norm in {"datetime", "timestamp", "obstime", "datetimeutc", "datetimelocal"}:
        return "DateTime"
    if "windspeed" in norm and "max" not in norm and "gust" not in norm:
        return "Wind_Speed_km_h"
    if "airtemp" in norm or ("air" in norm and "temp" in norm):
        return "Air_Temp_C"
    if "precip" in norm or norm in {"rain", "rainfall"}:
        return "Precipitation_cm"
    if "barometric" in norm or (
        "pressure" in norm and "vapor" not in norm and "vapour" not in norm
    ):
        return "Barometric_Pressure_mbar"
    if "humidity" in norm:
        return "Humidity_pct"
    if "solar" in norm or "shortwave" in norm or "sunshine" in norm:
        return "Solar_Radiation_W_m2"
    if "waterlevel" in norm or ("water" in norm and "level" in norm) or (
        "tidal" in norm and "level" in norm
    ):
        return "Water_Level_m"
    return None


def _convert_units(series: pd.Series, canonical: str, unit: str) -> pd.Series:
    """Convert a numeric column to schema units based on a detected unit token.

    Only well-recognized non-canonical units are converted; anything else is
    passed through untouched (the value is assumed to already be in the schema
    unit).

    Args:
        series: Numeric values (will be coerced with :func:`pandas.to_numeric`).
        canonical: Target schema column name.
        unit: Lowercased unit token extracted from the header (may be empty).

    Returns:
        The converted :class:`pandas.Series`.
    """
    values = pd.to_numeric(series, errors="coerce")
    unit = (unit or "").replace(" ", "")

    if canonical == "Air_Temp_C":
        if unit in {"f", "degf", "°f", "fahrenheit"}:
            return (values - 32.0) * 5.0 / 9.0
    elif canonical == "Wind_Speed_km_h":
        if unit in {"m/s", "ms", "mps", "ms-1", "m/s-1"}:
            return values * 3.6
        if unit in {"mph", "mi/h"}:
            return values * 1.609344
        if unit in {"kn", "kt", "knot", "knots"}:
            return values * 1.852
    elif canonical == "Precipitation_cm":
        if unit == "mm":
            return values / 10.0
        if unit == "m":
            return values * 100.0
        if unit in {"in", "inch", "inches", '"'}:
            return values * 2.54
    elif canonical == "Barometric_Pressure_mbar":
        if unit == "pa":
            return values / 100.0
        if unit == "kpa":
            return values * 10.0
        if unit in {"inhg", "inches", "in"}:
            return values * 33.8638866667
    return values


def load_weather_csv(
    file_or_buffer,
    column_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Load a weather CSV and normalize it into the standardized schema.

    Header matching is case-insensitive and tolerant of the bracketed-unit style
    used across this project (e.g. ``"Air Temp [C]"``, ``"Wind Speed [km/h]"``,
    ``"Precipitation [cm]"``, ``"Barometric Pressure [mbar]"``,
    ``"Humidity [%]"``). A UTF-8 BOM on the header is handled (the file is read
    with ``utf-8-sig`` and BOM characters are stripped from column names).

    The ``DateTime`` column is sourced from an explicit ``DateTime`` column when
    present, otherwise from a ``Date`` + ``Time`` pair, otherwise from a lone
    ``Date`` column. Where a header annotates a non-schema unit (``[mm]``,
    ``[m/s]``, ``[F]``, ``[Pa]`` ...) the values are converted to schema units.

    Args:
        file_or_buffer: Path, URL, or file-like object accepted by
            :func:`pandas.read_csv`.
        column_map: Optional explicit override mapping *source column name* ->
            *schema column name*. Entries here take precedence over
            auto-detection.

    Returns:
        A normalized weather :class:`pandas.DataFrame` (see module docstring).

    Raises:
        WeatherError: If the CSV cannot be read or no usable ``DateTime`` can be
            constructed.
    """
    try:
        raw = pd.read_csv(file_or_buffer, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        # StringIO / already-decoded buffers do not honour the encoding arg.
        if hasattr(file_or_buffer, "seek"):
            file_or_buffer.seek(0)
        raw = pd.read_csv(file_or_buffer, low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive
        raise WeatherError(f"Failed to read weather CSV: {exc}") from exc

    # Strip any residual BOM from the header so explicit column_map keys match.
    raw.columns = [str(col).replace("﻿", "") for col in raw.columns]

    column_map = dict(column_map or {})
    rename: dict[str, str] = {}
    used_targets: set[str] = set()

    # 1) Honour explicit overrides first.
    for source_col, target in column_map.items():
        if source_col in raw.columns and target not in used_targets:
            rename[source_col] = target
            used_targets.add(target)

    # 2) Auto-detect the remaining columns.
    date_col: Optional[str] = None
    time_col: Optional[str] = None
    for col in raw.columns:
        if col in rename:
            continue
        norm = _norm_key(col)
        if norm == "date" and date_col is None:
            date_col = col
            continue
        if norm == "time" and time_col is None:
            time_col = col
            continue
        canonical = _match_canonical(norm)
        if canonical and canonical not in used_targets:
            rename[col] = canonical
            used_targets.add(canonical)

    work = raw.rename(columns=rename)

    # 3) Build the DateTime column.
    if "DateTime" in work.columns:
        datetime_series = work["DateTime"]
    elif date_col is not None and time_col is not None:
        datetime_series = (
            raw[date_col].astype(str).str.strip()
            + " "
            + raw[time_col].astype(str).str.strip()
        )
    elif date_col is not None:
        datetime_series = raw[date_col]
    else:
        raise WeatherError(
            "Could not locate a 'DateTime' column, a 'Date'+'Time' pair, or a "
            "'Date' column in the CSV. Provide a column_map to disambiguate."
        )
    work["DateTime"] = datetime_series

    # 4) Apply unit conversions for numeric columns using original headers.
    for source_col, target in rename.items():
        if target in NUMERIC_COLUMNS:
            unit = _extract_unit(source_col)
            work[target] = _convert_units(work[target], target, unit)

    return to_sensor_frame(work)


# ---------------------------------------------------------------------------
# Merge / resample
# ---------------------------------------------------------------------------
def _require_nonempty_with_datetime(df: pd.DataFrame, label: str) -> None:
    """Validate that ``df`` is a non-empty DataFrame carrying ``DateTime``."""
    if not isinstance(df, pd.DataFrame):
        raise WeatherError(f"{label} must be a pandas DataFrame.")
    if df.empty:
        raise WeatherError(f"{label} is empty; nothing to align.")
    if "DateTime" not in df.columns:
        raise WeatherError(f"{label} is missing a 'DateTime' column.")


def merge_weather_into_sensor(
    sensor_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    tolerance_minutes: Union[int, float] = 30,
    prefer: str = "sensor",
) -> pd.DataFrame:
    """Align weather rows onto a sensor frame by nearest timestamp.

    Uses :func:`pandas.merge_asof` (``direction="nearest"``) to attach each
    weather observation to the closest sensor row within ``tolerance_minutes``.
    Weather columns that do not exist in ``sensor_df`` are added outright.
    Columns that already exist are reconciled per ``prefer``:

    * ``prefer="sensor"`` (default) -- keep existing sensor values, only filling
      ``NaN`` gaps from the weather frame.
    * ``prefer="weather"`` -- overwrite with the weather value wherever the
      weather frame has one, falling back to the sensor value otherwise.

    The original sensor row order is preserved. Rows whose ``DateTime`` cannot
    be parsed are retained (with ``NaN`` weather values) rather than dropped.

    Args:
        sensor_df: The sensor/environmental frame to enrich.
        weather_df: Normalized weather frame (schema of this package).
        tolerance_minutes: Maximum absolute time gap for a match.
        prefer: ``"sensor"`` or ``"weather"`` (see above).

    Returns:
        A new merged :class:`pandas.DataFrame`.

    Raises:
        WeatherError: If either frame is empty, missing ``DateTime``, or if
            ``prefer`` is not one of ``{"sensor", "weather"}``.
    """
    if prefer not in {"sensor", "weather"}:
        raise WeatherError("prefer must be 'sensor' or 'weather'.")
    _require_nonempty_with_datetime(sensor_df, "sensor_df")
    _require_nonempty_with_datetime(weather_df, "weather_df")

    sensor = sensor_df.copy()
    weather = weather_df.copy()
    sensor["DateTime"] = _coerce_naive_datetime(sensor["DateTime"])
    weather["DateTime"] = _coerce_naive_datetime(weather["DateTime"])

    weather = weather.dropna(subset=["DateTime"]).sort_values("DateTime")
    if weather.empty:
        raise WeatherError("weather_df has no valid DateTime values to align.")

    weather_value_cols = [c for c in weather.columns if c != "DateTime"]
    suffixed = {c: f"__w__{c}" for c in weather_value_cols}
    right = weather[["DateTime", *weather_value_cols]].rename(columns=suffixed)

    sensor["__order__"] = np.arange(len(sensor))
    valid_mask = sensor["DateTime"].notna()
    left = sensor.loc[valid_mask].sort_values("DateTime")

    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    merged = pd.merge_asof(
        left,
        right,
        on="DateTime",
        direction="nearest",
        tolerance=tolerance,
    )

    for col in weather_value_cols:
        wcol = suffixed[col]
        if col in sensor.columns:
            if prefer == "weather":
                merged[col] = merged[wcol].where(merged[wcol].notna(), merged[col])
            else:
                merged[col] = merged[col].where(merged[col].notna(), merged[wcol])
            merged = merged.drop(columns=[wcol])
        else:
            merged = merged.rename(columns={wcol: col})

    invalid = sensor.loc[~valid_mask]
    if not invalid.empty:
        invalid = invalid.copy()
        for col in weather_value_cols:
            if col not in invalid.columns:
                invalid[col] = np.nan
        combined = pd.concat([merged, invalid], ignore_index=True)
    else:
        combined = merged

    combined = combined.sort_values("__order__").drop(columns="__order__")
    return combined.reset_index(drop=True)


def resample_hourly(
    df: pd.DataFrame,
    *,
    precipitation_how: str = "sum",
) -> pd.DataFrame:
    """Resample a normalized weather/sensor frame to a regular hourly grid.

    Numeric columns are aggregated with the mean, except ``Precipitation_cm``
    which is summed by default (precipitation is an interval accumulation).

    Args:
        df: Frame with a ``DateTime`` column and one or more numeric columns.
        precipitation_how: Aggregation for ``Precipitation_cm`` (``"sum"`` or
            any pandas aggregation name, e.g. ``"mean"``).

    Returns:
        An hourly-resampled :class:`pandas.DataFrame` with ``DateTime`` restored
        as a column.

    Raises:
        WeatherError: If ``df`` is empty, lacks ``DateTime``, or has no numeric
            columns to aggregate.
    """
    _require_nonempty_with_datetime(df, "df")
    out = df.copy()
    out["DateTime"] = _coerce_naive_datetime(out["DateTime"])
    out = out.dropna(subset=["DateTime"]).sort_values("DateTime").set_index("DateTime")

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise WeatherError("resample_hourly() found no numeric columns to aggregate.")

    agg = {col: "mean" for col in numeric_cols}
    if "Precipitation_cm" in agg:
        agg["Precipitation_cm"] = precipitation_how

    resampled = out.resample("1h").agg(agg)
    return resampled.reset_index()
