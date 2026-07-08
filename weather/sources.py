"""Weather data source connectors.

This module implements the free weather/water data sources used by the app and
a small registry to look them up by name. Every connector returns a normalized
:class:`pandas.DataFrame` in the standardized schema defined in
:mod:`weather.normalize`.

Sources
-------
* :class:`OpenMeteoSource` -- primary, no API key. Automatically selects the
  historical *archive* endpoint or the *forecast* endpoint based on how recent
  the requested range is.
* :class:`NOAASource` -- no API key. NWS observations (via ``api.weather.gov``)
  plus optional NOAA CO-OPS water level when a ``station_id`` is supplied.
* :class:`SynopticSource` -- free token via the ``SYNOPTIC_TOKEN`` environment
  variable.

All network access goes through :func:`_request_json`, which enforces a short
timeout, sends a descriptive ``User-Agent`` (required by NWS), and converts any
failure into a :class:`~weather.normalize.WeatherFetchError` with a helpful
message -- callers never see a bare :mod:`requests` exception.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests

from .normalize import (
    WeatherConfigError,
    WeatherError,
    WeatherFetchError,
    empty_weather_frame,
    to_sensor_frame,
)

logger = logging.getLogger(__name__)

__all__ = [
    "WeatherSource",
    "OpenMeteoSource",
    "NOAASource",
    "SynopticSource",
    "SOURCES",
    "get_source",
]

#: Default network timeout (seconds) for all source requests.
DEFAULT_TIMEOUT = 20

#: Descriptive User-Agent. NWS (api.weather.gov) rejects requests without one.
DEFAULT_USER_AGENT = (
    "Tidegate-Analysis-App/1.0 (wildlife weather import; "
    "+https://github.com/Tidegate-Analysis-App)"
)

DateLike = Union[str, date, datetime, pd.Timestamp]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _as_date(value: DateLike) -> date:
    """Coerce a date-like value to a :class:`datetime.date`."""
    if isinstance(value, datetime):  # must precede `date` (datetime subclasses date)
        return value.date()
    if isinstance(value, date):
        return value
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise WeatherError(f"Could not interpret {value!r} as a date.")
    return ts.date()


def _as_datetime(value: DateLike) -> pd.Timestamp:
    """Coerce a date-like value to a tz-naive :class:`pandas.Timestamp`."""
    if isinstance(value, (date, datetime)):
        ts = pd.Timestamp(value)
    else:
        ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise WeatherError(f"Could not interpret {value!r} as a datetime.")
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def _iso_z(value: DateLike) -> str:
    """Format a date-like value as an ISO-8601 UTC string (``...Z``)."""
    return _as_datetime(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_float(value: Any) -> float:
    """Best-effort float conversion returning ``NaN`` on failure."""
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _request_json(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    context: str = "",
) -> Any:
    """Perform a GET request and return parsed JSON, wrapping all failures.

    Args:
        session: A :class:`requests.Session` (carries the User-Agent header).
        url: Fully-qualified request URL.
        params: Optional query parameters.
        timeout: Per-request timeout in seconds.
        context: Short label used to prefix error messages (e.g. ``"Open-Meteo"``).

    Returns:
        The decoded JSON body (``dict`` or ``list``).

    Raises:
        WeatherFetchError: On any transport error, non-200 status, or invalid
            JSON body.
    """
    label = context or "Weather request"
    try:
        response = session.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise WeatherFetchError(f"{label} failed for {url}: {exc}") from exc

    status = getattr(response, "status_code", None)
    if status is not None and status != 200:
        body = ""
        try:
            body = (response.text or "")[:300]
        except Exception:  # pragma: no cover - defensive
            body = ""
        raise WeatherFetchError(
            f"{label} to {url} returned HTTP {status}: {body}".rstrip()
        )

    try:
        return response.json()
    except ValueError as exc:
        raise WeatherFetchError(
            f"{label} to {url} returned a non-JSON / unparseable body: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class WeatherSource(ABC):
    """Abstract base class for weather source connectors.

    Subclasses implement :meth:`fetch` and may override :meth:`available`.

    Args:
        session: Optional pre-built :class:`requests.Session` (useful for
            testing). A new session is created if omitted.
        user_agent: Optional User-Agent override. Falls back to the
            ``WEATHER_USER_AGENT`` environment variable, then a package default.
        timeout: Per-request timeout in seconds.
    """

    #: Registry key / human-readable identifier for the source.
    name: str = "base"

    def __init__(
        self,
        *,
        session: Optional[requests.Session] = None,
        user_agent: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.timeout = timeout
        self.user_agent = (
            user_agent or os.environ.get("WEATHER_USER_AGENT") or DEFAULT_USER_AGENT
        )
        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": self.user_agent, "Accept": "application/json"}
        )

    def available(self) -> bool:
        """Return whether the source is usable (credentials present, etc.)."""
        return True

    @abstractmethod
    def fetch(
        self, lat: float, lon: float, start: DateLike, end: DateLike
    ) -> pd.DataFrame:
        """Fetch normalized weather data for a location and time range."""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} name={self.name!r} available={self.available()}>"


# ---------------------------------------------------------------------------
# Open-Meteo (primary)
# ---------------------------------------------------------------------------
class OpenMeteoSource(WeatherSource):
    """Open-Meteo connector (primary source, no API key required).

    Historical data comes from the ERA5 *archive* endpoint; recent data and
    forecasts come from the *forecast* endpoint. The endpoint is chosen
    automatically from the recency of the requested ``end`` date (the archive
    lags real time by a few days).

    Time zone: ``DateTime`` is **local station time** (request uses
    ``timezone=auto``), returned tz-naive.
    """

    name = "open-meteo"
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    #: The archive endpoint trails real time; ranges ending within this many
    #: days of "today" are served from the forecast endpoint instead.
    ARCHIVE_LATENCY_DAYS = 5

    HOURLY_VARIABLES = [
        "temperature_2m",
        "wind_speed_10m",
        "precipitation",
        "surface_pressure",
        "relative_humidity_2m",
        "shortwave_radiation",
    ]

    #: Open-Meteo hourly variable -> schema column.
    VARIABLE_MAP = {
        "temperature_2m": "Air_Temp_C",
        "wind_speed_10m": "Wind_Speed_km_h",
        "precipitation": "Precipitation_cm",
        "surface_pressure": "Barometric_Pressure_mbar",
        "relative_humidity_2m": "Humidity_pct",
        "shortwave_radiation": "Solar_Radiation_W_m2",
    }

    def _choose_url(self, end_date: date) -> str:
        """Return the archive URL for old ranges, else the forecast URL."""
        cutoff = date.today() - timedelta(days=self.ARCHIVE_LATENCY_DAYS)
        return self.ARCHIVE_URL if end_date < cutoff else self.FORECAST_URL

    def fetch(
        self, lat: float, lon: float, start: DateLike, end: DateLike
    ) -> pd.DataFrame:
        """Fetch hourly Open-Meteo data and normalize to the schema.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            start: Range start (``date`` or ``datetime``).
            end: Range end (``date`` or ``datetime``).

        Returns:
            A normalized weather :class:`pandas.DataFrame`.

        Raises:
            WeatherFetchError: On network failure or an Open-Meteo error body.
        """
        start_date = _as_date(start)
        end_date = _as_date(end)
        url = self._choose_url(end_date)
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(self.HOURLY_VARIABLES),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "timezone": "auto",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",  # explicit: matches Wind_Speed_km_h
            "precipitation_unit": "mm",  # converted to cm below
        }
        logger.debug("Open-Meteo GET %s params=%s", url, params)
        payload = _request_json(
            self.session, url, params=params, timeout=self.timeout, context="Open-Meteo"
        )
        if isinstance(payload, dict) and payload.get("error"):
            raise WeatherFetchError(
                f"Open-Meteo returned an error: {payload.get('reason', 'unknown')}"
            )
        return self._parse(payload)

    def _parse(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """Convert an Open-Meteo hourly payload into a normalized frame."""
        hourly = (payload or {}).get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            logger.warning("Open-Meteo payload contained no hourly data.")
            return empty_weather_frame()

        frame = pd.DataFrame({"DateTime": times})
        for api_key, column in self.VARIABLE_MAP.items():
            values = hourly.get(api_key)
            if values is not None:
                frame[column] = pd.to_numeric(pd.Series(values), errors="coerce")

        if "Precipitation_cm" in frame.columns:
            # Open-Meteo precipitation is millimetres; convert to centimetres.
            frame["Precipitation_cm"] = frame["Precipitation_cm"] / 10.0

        return to_sensor_frame(frame)


# ---------------------------------------------------------------------------
# NOAA (NWS observations + optional CO-OPS water level)
# ---------------------------------------------------------------------------
def _unit_suffix(unit_code: Any) -> str:
    """Return the lowercased unit token after the ``:`` in an NWS unit code."""
    if not unit_code:
        return ""
    return str(unit_code).split(":")[-1].strip().lower()


def _quantity_value(measure: Any) -> Optional[float]:
    """Extract ``value`` from an NWS ``{"value": ..., "unitCode": ...}`` object."""
    if isinstance(measure, dict):
        return measure.get("value")
    return None


def _nws_temp_c(measure: Any) -> float:
    value = _quantity_value(measure)
    if value is None:
        return float("nan")
    unit = _unit_suffix(measure.get("unitCode"))
    if unit in {"degf", "f"}:
        return (float(value) - 32.0) * 5.0 / 9.0
    return float(value)


def _nws_wind_kmh(measure: Any) -> float:
    value = _quantity_value(measure)
    if value is None:
        return float("nan")
    unit = _unit_suffix(measure.get("unitCode"))
    if unit in {"m_s-1", "ms-1", "m/s", "mps"}:
        return float(value) * 3.6
    if unit in {"mph", "mi_h-1"}:
        return float(value) * 1.609344
    if unit in {"kn", "kt", "knot", "knots"}:
        return float(value) * 1.852
    return float(value)  # km_h-1 (or unknown -> assume km/h)


def _nws_pressure_mbar(measure: Any) -> float:
    value = _quantity_value(measure)
    if value is None:
        return float("nan")
    unit = _unit_suffix(measure.get("unitCode"))
    if unit == "pa":
        return float(value) / 100.0
    if unit == "kpa":
        return float(value) * 10.0
    if unit in {"hpa", "mbar", "mb"}:
        return float(value)
    # Unknown unit: NWS reports pascals natively, so use magnitude as a hint.
    return float(value) / 100.0 if float(value) > 2000 else float(value)


def _nws_precip_cm(measure: Any) -> float:
    value = _quantity_value(measure)
    if value is None:
        return float("nan")
    unit = _unit_suffix(measure.get("unitCode"))
    if unit == "mm":
        return float(value) / 10.0
    if unit == "m":
        return float(value) * 100.0
    if unit == "cm":
        return float(value)
    if unit in {"in", "inch"}:
        return float(value) * 2.54
    return float(value) / 10.0  # assume millimetres


class NOAASource(WeatherSource):
    """NOAA connector: NWS observations plus optional CO-OPS water level.

    Weather observations are pulled from the National Weather Service API
    (``api.weather.gov``): the nearest station to ``lat``/``lon`` is discovered
    via ``/points`` -> ``/stations``, then its ``/observations`` are read. When a
    CO-OPS ``station_id`` is configured, tide-gauge water level is fetched from
    the Tides & Currents ``datagetter`` API and aligned onto the observation
    timeline by nearest timestamp.

    The connector is deliberately resilient to sparse data: if NWS observations
    are unavailable but a CO-OPS station is configured, the water-level frame is
    returned on its own (and vice versa).

    Time zone: ``DateTime`` is **UTC** (NWS timestamps are UTC; CO-OPS is
    requested with ``time_zone=gmt``), returned tz-naive.

    Args:
        coops_station_id: Optional NOAA CO-OPS station id (e.g. ``"9432780"``).
        coops_product: CO-OPS product, ``"water_level"`` (default) or
            ``"predictions"``.
        coops_datum: Tidal datum for water-level requests (default ``"MLLW"``).
    """

    name = "noaa"
    NWS_BASE = "https://api.weather.gov"
    COOPS_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    def __init__(
        self,
        *,
        coops_station_id: Optional[str] = None,
        coops_product: str = "water_level",
        coops_datum: str = "MLLW",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.coops_station_id = coops_station_id
        self.coops_product = coops_product
        self.coops_datum = coops_datum

    def fetch(
        self,
        lat: float,
        lon: float,
        start: DateLike,
        end: DateLike,
        station_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch NWS observations and (optionally) CO-OPS water level.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            start: Range start (``date`` or ``datetime``).
            end: Range end (``date`` or ``datetime``).
            station_id: Optional CO-OPS station id overriding the constructor
                value.

        Returns:
            A normalized weather :class:`pandas.DataFrame`.

        Raises:
            WeatherFetchError: If neither data stream yields any data.
        """
        sid = station_id if station_id is not None else self.coops_station_id

        obs_df: Optional[pd.DataFrame] = None
        obs_error: Optional[WeatherFetchError] = None
        try:
            obs_df = self._fetch_nws_observations(lat, lon, start, end)
        except WeatherFetchError as exc:
            obs_error = exc
            logger.warning("NWS observations unavailable: %s", exc)

        water_df: Optional[pd.DataFrame] = None
        if sid is not None:
            water_df = self._fetch_coops_water_level(sid, start, end)

        obs_ok = obs_df is not None and not obs_df.empty
        water_ok = water_df is not None and not water_df.empty

        if not obs_ok and not water_ok:
            if obs_error is not None and sid is None:
                raise obs_error
            raise WeatherFetchError(
                "NOAA returned no usable data for the requested location/range."
            )
        if obs_ok and water_ok:
            return self._attach_water_level(obs_df, water_df)
        if obs_ok:
            return obs_df
        return water_df  # water-only

    def _fetch_nws_observations(
        self, lat: float, lon: float, start: DateLike, end: DateLike
    ) -> pd.DataFrame:
        """Discover the nearest NWS station and read its observations."""
        points = _request_json(
            self.session,
            f"{self.NWS_BASE}/points/{lat:.4f},{lon:.4f}",
            context="NWS points",
        )
        stations_url = (
            (points or {}).get("properties", {}).get("observationStations")
        )
        if not stations_url:
            logger.warning("NWS /points response lacked observationStations.")
            return empty_weather_frame()

        stations = _request_json(
            self.session, stations_url, context="NWS stations"
        )
        features = (stations or {}).get("features") or []
        if not features:
            logger.warning("No NWS observation stations near %s,%s.", lat, lon)
            return empty_weather_frame()

        station_url = features[0].get("id")
        station_ident = (
            features[0].get("properties", {}).get("stationIdentifier", "?")
        )
        logger.debug("Using NWS station %s (%s)", station_ident, station_url)

        observations = _request_json(
            self.session,
            f"{station_url}/observations",
            params={"start": _iso_z(start), "end": _iso_z(end)},
            context="NWS observations",
        )
        return self._parse_observations(observations)

    @staticmethod
    def _parse_observations(payload: Dict[str, Any]) -> pd.DataFrame:
        """Convert an NWS observations feature collection into a frame."""
        features = (payload or {}).get("features") or []
        rows: List[Dict[str, Any]] = []
        for feature in features:
            props = feature.get("properties", {})
            pressure = props.get("barometricPressure")
            if _quantity_value(pressure) is None:
                pressure = props.get("seaLevelPressure")
            rows.append(
                {
                    "DateTime": props.get("timestamp"),
                    "Air_Temp_C": _nws_temp_c(props.get("temperature")),
                    "Wind_Speed_km_h": _nws_wind_kmh(props.get("windSpeed")),
                    "Barometric_Pressure_mbar": _nws_pressure_mbar(pressure),
                    "Humidity_pct": _to_float(
                        _quantity_value(props.get("relativeHumidity"))
                    ),
                    "Precipitation_cm": _nws_precip_cm(
                        props.get("precipitationLastHour")
                    ),
                }
            )
        if not rows:
            return empty_weather_frame()
        return to_sensor_frame(pd.DataFrame(rows))

    def _fetch_coops_water_level(
        self, station_id: str, start: DateLike, end: DateLike
    ) -> pd.DataFrame:
        """Fetch NOAA CO-OPS water level / predictions for a station."""
        params = {
            "product": self.coops_product,
            "application": "Tidegate-Analysis-App",
            "begin_date": _as_date(start).strftime("%Y%m%d"),
            "end_date": _as_date(end).strftime("%Y%m%d"),
            "station": str(station_id),
            "time_zone": "gmt",
            "units": "metric",
            "format": "json",
        }
        if self.coops_product in {"water_level", "predictions"}:
            params["datum"] = self.coops_datum
        if self.coops_product == "predictions":
            params["interval"] = "h"

        payload = _request_json(
            self.session,
            self.COOPS_URL,
            params=params,
            timeout=self.timeout,
            context="NOAA CO-OPS",
        )
        if isinstance(payload, dict) and payload.get("error"):
            message = payload["error"].get("message", "unknown error")
            raise WeatherFetchError(f"NOAA CO-OPS error: {message}")

        data = (payload or {}).get("data") or (payload or {}).get("predictions") or []
        rows = [
            {"DateTime": row.get("t"), "Water_Level_m": _to_float(row.get("v"))}
            for row in data
        ]
        if not rows:
            logger.warning("CO-OPS returned no water-level rows for %s.", station_id)
            return empty_weather_frame()
        return to_sensor_frame(pd.DataFrame(rows))

    @staticmethod
    def _attach_water_level(
        obs_df: pd.DataFrame, water_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Align water level onto the observation timeline (nearest, <=1h)."""
        left = obs_df.sort_values("DateTime").drop(columns=["Water_Level_m"])
        right = (
            water_df[["DateTime", "Water_Level_m"]]
            .dropna(subset=["DateTime"])
            .sort_values("DateTime")
        )
        merged = pd.merge_asof(
            left,
            right,
            on="DateTime",
            direction="nearest",
            tolerance=pd.Timedelta(hours=1),
        )
        return to_sensor_frame(merged)


# ---------------------------------------------------------------------------
# Synoptic (token via SYNOPTIC_TOKEN)
# ---------------------------------------------------------------------------
def _synoptic_scale(values, unit: str, kind: str) -> pd.Series:
    """Convert a Synoptic variable list to schema units.

    Args:
        values: Raw list/array of observation values.
        unit: The unit string from the payload ``UNITS`` map (may be empty).
        kind: One of ``{"temp", "wind", "precip", "pressure"}``. Any other kind
            passes the values through unchanged.

    Returns:
        A converted :class:`pandas.Series`.
    """
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    unit = (unit or "").strip().lower()

    if kind == "temp":
        if "fahrenheit" in unit or unit in {"f", "degf"}:
            return (series - 32.0) * 5.0 / 9.0
        return series
    if kind == "wind":
        if unit in {"m/s", "mps", "m s-1"} or "meters/second" in unit:
            return series * 3.6
        if "mph" in unit or "miles" in unit:
            return series * 1.609344
        if "knot" in unit or unit in {"kn", "kt"}:
            return series * 1.852
        return series  # km/h / kph
    if kind == "precip":
        if "milli" in unit or unit == "mm":
            return series / 10.0
        if "centi" in unit or unit == "cm":
            return series
        if "inch" in unit or unit == "in":
            return series * 2.54
        if unit in {"m", "meters"}:
            return series * 100.0
        return series / 10.0  # metric default is millimetres
    if kind == "pressure":
        if "pascal" in unit or unit == "pa":
            return series / 100.0
        if any(tok in unit for tok in ("hecto", "hpa", "millibar", "mbar", "mb")):
            return series
        # Fall back on magnitude: pascals are ~1e5, millibars ~1e3.
        median = series.dropna().median()
        if pd.notna(median) and median > 2000:
            return series / 100.0
        return series
    return series


class SynopticSource(WeatherSource):
    """Synoptic Data (MesoWest) connector -- requires a free API token.

    The token is read from the ``SYNOPTIC_TOKEN`` environment variable or passed
    explicitly. When no token is present :meth:`available` returns ``False`` and
    :meth:`fetch` raises :class:`~weather.normalize.WeatherConfigError`.

    Time zone: ``DateTime`` is **UTC** (request uses ``obtimezone=utc``),
    returned tz-naive.

    Args:
        token: Explicit API token (overrides ``SYNOPTIC_TOKEN``).
        station_id: Optional Synoptic station id (``stid``). When omitted the
            nearest station within ``radius_miles`` of ``lat``/``lon`` is used.
        radius_miles: Search radius when discovering a station by coordinates.
    """

    name = "synoptic"
    URL = "https://api.synopticdata.com/v2/stations/timeseries"

    REQUEST_VARIABLES = [
        "air_temp",
        "wind_speed",
        "precip_accum_one_hour",
        "sea_level_pressure",
        "pressure",
        "relative_humidity",
        "solar_radiation",
    ]

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        station_id: Optional[str] = None,
        radius_miles: int = 25,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.token = token or os.environ.get("SYNOPTIC_TOKEN")
        self.station_id = station_id
        self.radius_miles = radius_miles

    def available(self) -> bool:
        """Return ``True`` only when an API token is configured."""
        return bool(self.token)

    def fetch(
        self,
        lat: float,
        lon: float,
        start: DateLike,
        end: DateLike,
        station_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch a Synoptic timeseries and normalize to the schema.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            start: Range start (``date`` or ``datetime``).
            end: Range end (``date`` or ``datetime``).
            station_id: Optional Synoptic ``stid`` overriding the constructor.

        Returns:
            A normalized weather :class:`pandas.DataFrame`.

        Raises:
            WeatherConfigError: If no API token is configured.
            WeatherFetchError: On network failure or a Synoptic error response.
        """
        if not self.token:
            raise WeatherConfigError(
                "SynopticSource requires an API token. Set the SYNOPTIC_TOKEN "
                "environment variable or pass token=... to the constructor."
            )

        params: Dict[str, Any] = {
            "token": self.token,
            "start": _as_datetime(start).strftime("%Y%m%d%H%M"),
            "end": _as_datetime(end).strftime("%Y%m%d%H%M"),
            "vars": ",".join(self.REQUEST_VARIABLES),
            "units": "metric",
            "obtimezone": "utc",
        }
        sid = station_id if station_id is not None else self.station_id
        if sid is not None:
            params["stid"] = sid
        else:
            params["radius"] = f"{lat},{lon},{self.radius_miles}"
            params["limit"] = 1

        payload = _request_json(
            self.session, self.URL, params=params, timeout=self.timeout,
            context="Synoptic",
        )
        summary = (payload or {}).get("SUMMARY", {})
        code = summary.get("RESPONSE_CODE")
        if code is not None and code != 1:
            raise WeatherFetchError(
                f"Synoptic error: {summary.get('RESPONSE_MESSAGE', 'unknown')}"
            )
        return self._parse(payload)

    def _parse(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """Convert a Synoptic timeseries payload into a normalized frame."""
        stations = (payload or {}).get("STATION") or []
        if not stations:
            logger.warning("Synoptic response contained no stations.")
            return empty_weather_frame()

        observations = stations[0].get("OBSERVATIONS") or {}
        units = (payload or {}).get("UNITS") or {}
        times = observations.get("date_time") or []
        if not times:
            return empty_weather_frame()

        frame = pd.DataFrame({"DateTime": times})

        def find_key(prefix: str) -> Optional[str]:
            for key in observations:
                if key == "date_time":
                    continue
                if key.startswith(prefix):
                    return key
            return None

        def unit_for(key: str) -> str:
            base = re.sub(r"_set_\d+.*$", "", key)
            return units.get(base, "")

        temp_key = find_key("air_temp")
        wind_key = find_key("wind_speed")
        precip_key = find_key("precip_accum")
        pressure_key = (
            find_key("sea_level_pressure")
            or find_key("pressure")
            or find_key("altimeter")
        )
        humidity_key = find_key("relative_humidity")
        solar_key = find_key("solar_radiation")

        if temp_key:
            frame["Air_Temp_C"] = _synoptic_scale(
                observations[temp_key], unit_for(temp_key), "temp"
            )
        if wind_key:
            frame["Wind_Speed_km_h"] = _synoptic_scale(
                observations[wind_key], unit_for(wind_key), "wind"
            )
        if precip_key:
            frame["Precipitation_cm"] = _synoptic_scale(
                observations[precip_key], unit_for(precip_key), "precip"
            )
        if pressure_key:
            frame["Barometric_Pressure_mbar"] = _synoptic_scale(
                observations[pressure_key], unit_for(pressure_key), "pressure"
            )
        if humidity_key:
            frame["Humidity_pct"] = pd.to_numeric(
                pd.Series(observations[humidity_key]), errors="coerce"
            )
        if solar_key:
            frame["Solar_Radiation_W_m2"] = pd.to_numeric(
                pd.Series(observations[solar_key]), errors="coerce"
            )

        return to_sensor_frame(frame)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#: Registry of source name -> connector class.
SOURCES: Dict[str, type] = {
    "open-meteo": OpenMeteoSource,
    "noaa": NOAASource,
    "synoptic": SynopticSource,
}


def get_source(name: str, **kwargs: Any) -> WeatherSource:
    """Instantiate a weather source by registry name.

    Args:
        name: One of :data:`SOURCES` (e.g. ``"open-meteo"``, ``"noaa"``,
            ``"synoptic"``).
        **kwargs: Forwarded to the connector constructor.

    Returns:
        A constructed :class:`WeatherSource` instance.

    Raises:
        WeatherConfigError: If ``name`` is not a registered source.
    """
    try:
        source_cls = SOURCES[name]
    except KeyError:
        available = ", ".join(sorted(SOURCES))
        raise WeatherConfigError(
            f"Unknown weather source {name!r}. Available sources: {available}."
        ) from None
    return source_cls(**kwargs)
