"""Offline unit tests for the :mod:`weather` package.

These tests never touch the network. Source connectors are exercised by
monkeypatching :meth:`requests.Session.get` to return canned JSON payloads, so
the full HTTP-wrapping / parsing / unit-conversion path is covered while running
fully offline.
"""

from __future__ import annotations

import io
import json
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import requests

# Make the repo root importable regardless of pytest's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weather  # noqa: E402
from weather import (  # noqa: E402
    NOAASource,
    OpenMeteoSource,
    SynopticSource,
    WeatherConfigError,
    WeatherError,
    load_weather_csv,
    merge_weather_into_sensor,
    resample_hourly,
)


# ---------------------------------------------------------------------------
# Fake HTTP plumbing
# ---------------------------------------------------------------------------
class FakeResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):  # pragma: no cover - not used by the wrapper
        if self.status_code != 200:
            raise requests.HTTPError(f"status {self.status_code}")


def install_fake_get(monkeypatch, router, calls=None):
    """Patch ``requests.Session.get`` with a URL-routing fake.

    Args:
        monkeypatch: pytest ``monkeypatch`` fixture.
        router: callable ``(url, params) -> payload`` returning the JSON body.
        calls: optional list that receives ``(url, params)`` tuples.
    """

    def fake_get(self, url, params=None, timeout=None, **kwargs):
        if calls is not None:
            calls.append((url, params))
        payload = router(url, params)
        return FakeResponse(payload)

    monkeypatch.setattr(requests.Session, "get", fake_get)


# ---------------------------------------------------------------------------
# Canned payloads
# ---------------------------------------------------------------------------
OPEN_METEO_PAYLOAD = {
    "latitude": 43.34,
    "longitude": -124.32,
    "hourly_units": {
        "temperature_2m": "°C",
        "wind_speed_10m": "km/h",
        "precipitation": "mm",
        "surface_pressure": "hPa",
        "relative_humidity_2m": "%",
        "shortwave_radiation": "W/m²",
    },
    "hourly": {
        "time": ["2023-06-01T00:00", "2023-06-01T01:00", "2023-06-01T02:00"],
        "temperature_2m": [10.0, 11.5, 12.0],
        "wind_speed_10m": [5.0, 7.2, 3.1],  # already km/h
        "precipitation": [0.0, 2.0, 10.0],  # mm -> cm (/10)
        "surface_pressure": [1013.2, 1012.8, 1011.0],  # hPa == mbar
        "relative_humidity_2m": [80, 82, 85],
        "shortwave_radiation": [0, 50, 120],
    },
}

NWS_POINTS = {
    "properties": {"observationStations": "https://api.weather.gov/gridpoints/X/1,2/stations"}
}
NWS_STATIONS = {
    "features": [
        {
            "id": "https://api.weather.gov/stations/KTST",
            "properties": {"stationIdentifier": "KTST"},
        }
    ]
}
NWS_OBSERVATIONS = {
    "features": [
        {
            "properties": {
                "timestamp": "2023-06-01T00:00:00+00:00",
                "temperature": {"unitCode": "wmoUnit:degC", "value": 12.0},
                "windSpeed": {"unitCode": "wmoUnit:km_h-1", "value": 18.0},
                "barometricPressure": {"unitCode": "wmoUnit:Pa", "value": 101300},
                "relativeHumidity": {"unitCode": "wmoUnit:percent", "value": 77.0},
                "precipitationLastHour": {"unitCode": "wmoUnit:mm", "value": 5.0},
            }
        },
        {
            "properties": {
                "timestamp": "2023-06-01T01:00:00+00:00",
                "temperature": {"unitCode": "wmoUnit:degC", "value": 11.0},
                "windSpeed": {"unitCode": "wmoUnit:m_s-1", "value": 5.0},  # -> 18 km/h
                "barometricPressure": {"unitCode": "wmoUnit:Pa", "value": 101250},
                "relativeHumidity": {"unitCode": "wmoUnit:percent", "value": 80.0},
                "precipitationLastHour": {"unitCode": "wmoUnit:m", "value": 0.001},  # -> 0.1 cm
            }
        },
    ]
}
COOPS_WATER = {
    "data": [
        {"t": "2023-06-01 00:00", "v": "1.234"},
        {"t": "2023-06-01 01:00", "v": "1.500"},
    ]
}

SYNOPTIC_PAYLOAD = {
    "SUMMARY": {"RESPONSE_CODE": 1, "RESPONSE_MESSAGE": "OK"},
    "UNITS": {
        "air_temp": "Celsius",
        "wind_speed": "m/s",
        "precip_accum_one_hour": "Millimeters",
        "sea_level_pressure": "Pascals",
        "relative_humidity": "%",
        "solar_radiation": "W/m**2",
    },
    "STATION": [
        {
            "STID": "TEST1",
            "OBSERVATIONS": {
                "date_time": ["2023-06-01T00:00:00Z", "2023-06-01T01:00:00Z"],
                "air_temp_set_1": [10.0, 11.0],
                "wind_speed_set_1": [5.0, 10.0],  # m/s -> 18, 36 km/h
                "precip_accum_one_hour_set_1": [2.0, 0.0],  # mm -> 0.2, 0.0 cm
                "sea_level_pressure_set_1": [101300.0, 101250.0],  # Pa -> 1013, 1012.5
                "relative_humidity_set_1": [70.0, 72.0],
                "solar_radiation_set_1": [100.0, 200.0],
            },
        }
    ],
}


# ---------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------
def test_open_meteo_parse(monkeypatch):
    install_fake_get(monkeypatch, lambda url, params: OPEN_METEO_PAYLOAD)
    src = OpenMeteoSource()
    df = src.fetch(lat=43.34, lon=-124.32, start="2023-06-01", end="2023-06-01")

    assert list(df.columns) == weather.SCHEMA_COLUMNS
    assert len(df) == 3
    # tz-naive datetime
    assert pd.api.types.is_datetime64_ns_dtype(df["DateTime"])
    assert df["DateTime"].dt.tz is None
    assert df["DateTime"].iloc[0] == pd.Timestamp("2023-06-01 00:00")
    # values / conversions
    assert df["Air_Temp_C"].tolist() == [10.0, 11.5, 12.0]
    assert df["Wind_Speed_km_h"].tolist() == [5.0, 7.2, 3.1]  # km/h passthrough
    # precipitation mm -> cm
    assert df["Precipitation_cm"].tolist() == pytest.approx([0.0, 0.2, 1.0])
    # pressure hPa == mbar passthrough
    assert df["Barometric_Pressure_mbar"].iloc[0] == pytest.approx(1013.2)
    assert df["Humidity_pct"].tolist() == [80.0, 82.0, 85.0]
    assert df["Solar_Radiation_W_m2"].tolist() == [0.0, 50.0, 120.0]


def test_open_meteo_endpoint_selection(monkeypatch):
    calls = []
    install_fake_get(monkeypatch, lambda url, params: OPEN_METEO_PAYLOAD, calls=calls)
    src = OpenMeteoSource()

    old = date.today() - timedelta(days=400)
    src.fetch(lat=1.0, lon=2.0, start=old, end=old)
    assert "archive-api.open-meteo.com" in calls[-1][0]

    recent = date.today() - timedelta(days=1)
    src.fetch(lat=1.0, lon=2.0, start=recent, end=recent)
    assert calls[-1][0] == OpenMeteoSource.FORECAST_URL


def test_open_meteo_error_body(monkeypatch):
    install_fake_get(
        monkeypatch,
        lambda url, params: {"error": True, "reason": "Invalid parameters"},
    )
    with pytest.raises(weather.WeatherFetchError):
        OpenMeteoSource().fetch(lat=1.0, lon=2.0, start="2023-01-01", end="2023-01-02")


# ---------------------------------------------------------------------------
# NOAA (NWS observations + CO-OPS water level)
# ---------------------------------------------------------------------------
def _noaa_router(url, params):
    if "/points/" in url:
        return NWS_POINTS
    if url.endswith("/stations"):
        return NWS_STATIONS
    if url.endswith("/observations"):
        return NWS_OBSERVATIONS
    if "datagetter" in url:
        return COOPS_WATER
    raise AssertionError(f"unexpected url {url}")


def test_noaa_observations_and_water_level(monkeypatch):
    install_fake_get(monkeypatch, _noaa_router)
    src = NOAASource(coops_station_id="9432780")
    df = src.fetch(lat=43.34, lon=-124.32, start="2023-06-01", end="2023-06-02")

    assert list(df.columns) == weather.SCHEMA_COLUMNS
    assert len(df) == 2
    # unit conversions
    assert df["Air_Temp_C"].tolist() == [12.0, 11.0]
    assert df["Wind_Speed_km_h"].tolist() == pytest.approx([18.0, 18.0])  # 5 m/s -> 18
    assert df["Barometric_Pressure_mbar"].tolist() == pytest.approx([1013.0, 1012.5])
    assert df["Precipitation_cm"].tolist() == pytest.approx([0.5, 0.1])  # mm & m -> cm
    assert df["Humidity_pct"].tolist() == [77.0, 80.0]
    # water level aligned onto the observation timeline
    assert df["Water_Level_m"].tolist() == pytest.approx([1.234, 1.500])
    # NWS timestamps are UTC, returned tz-naive
    assert df["DateTime"].iloc[0] == pd.Timestamp("2023-06-01 00:00")


def test_noaa_water_only_when_nws_missing(monkeypatch):
    def router(url, params):
        if "datagetter" in url:
            return COOPS_WATER
        # NWS points has no stations -> observations unavailable
        if "/points/" in url:
            return {"properties": {}}
        raise AssertionError(f"unexpected url {url}")

    install_fake_get(monkeypatch, router)
    src = NOAASource(coops_station_id="9432780")
    df = src.fetch(lat=1.0, lon=2.0, start="2023-06-01", end="2023-06-02")
    assert df["Water_Level_m"].tolist() == pytest.approx([1.234, 1.500])
    assert df["Air_Temp_C"].isna().all()


def test_noaa_raises_when_all_empty(monkeypatch):
    install_fake_get(monkeypatch, lambda url, params: {"properties": {}})
    with pytest.raises(weather.WeatherFetchError):
        NOAASource().fetch(lat=1.0, lon=2.0, start="2023-06-01", end="2023-06-02")


# ---------------------------------------------------------------------------
# Synoptic
# ---------------------------------------------------------------------------
def test_synoptic_available_without_token(monkeypatch):
    monkeypatch.delenv("SYNOPTIC_TOKEN", raising=False)
    src = SynopticSource(token=None)
    assert src.available() is False
    with pytest.raises(WeatherConfigError):
        src.fetch(lat=1.0, lon=2.0, start="2023-06-01", end="2023-06-02")


def test_synoptic_parse_with_token(monkeypatch):
    install_fake_get(monkeypatch, lambda url, params: SYNOPTIC_PAYLOAD)
    src = SynopticSource(token="fake-token")
    assert src.available() is True
    df = src.fetch(lat=1.0, lon=2.0, start="2023-06-01", end="2023-06-02")

    assert len(df) == 2
    assert df["Air_Temp_C"].tolist() == [10.0, 11.0]
    assert df["Wind_Speed_km_h"].tolist() == pytest.approx([18.0, 36.0])  # m/s -> km/h
    assert df["Precipitation_cm"].tolist() == pytest.approx([0.2, 0.0])  # mm -> cm
    assert df["Barometric_Pressure_mbar"].tolist() == pytest.approx([1013.0, 1012.5])
    assert df["Humidity_pct"].tolist() == [70.0, 72.0]
    assert df["Solar_Radiation_W_m2"].tolist() == [100.0, 200.0]


# ---------------------------------------------------------------------------
# CSV importer
# ---------------------------------------------------------------------------
def test_load_weather_csv_bom_and_bracketed_units():
    # BOM on the header + bracketed-unit style + Date/Time pair.
    csv_text = (
        "﻿Date,Time,Air Temp [C],Wind Speed [km/h],Precipitation [cm],"
        "Barometric Pressure [mbar],Humidity [%],Sunshine Radiation [W/m^2]\n"
        "2023-06-01,00:00,10.5,12.0,0.10,1013.2,80,0\n"
        "2023-06-01,01:00,11.0,15.0,0.00,1012.8,82,55\n"
    )
    df = load_weather_csv(io.StringIO(csv_text))

    assert list(df.columns) == weather.SCHEMA_COLUMNS
    assert len(df) == 2
    assert df["DateTime"].iloc[0] == pd.Timestamp("2023-06-01 00:00")
    assert df["DateTime"].iloc[1] == pd.Timestamp("2023-06-01 01:00")
    assert df["Air_Temp_C"].tolist() == [10.5, 11.0]
    assert df["Wind_Speed_km_h"].tolist() == [12.0, 15.0]
    assert df["Precipitation_cm"].tolist() == pytest.approx([0.10, 0.0])
    assert df["Barometric_Pressure_mbar"].tolist() == pytest.approx([1013.2, 1012.8])
    assert df["Humidity_pct"].tolist() == [80.0, 82.0]
    assert df["Solar_Radiation_W_m2"].tolist() == [0.0, 55.0]


def test_load_weather_csv_bom_bytes_and_datetime_column():
    # BOM encoded as bytes (utf-8-sig) + explicit DateTime column.
    csv_text = "DateTime,Air Temp [C]\n2023-06-01 00:00,9.0\n2023-06-01 01:00,9.5\n"
    buffer = io.BytesIO(csv_text.encode("utf-8-sig"))
    df = load_weather_csv(buffer)
    assert df["DateTime"].iloc[0] == pd.Timestamp("2023-06-01 00:00")
    assert df["Air_Temp_C"].tolist() == [9.0, 9.5]


def test_load_weather_csv_unit_conversions():
    # Non-schema units in headers should be converted.
    csv_text = (
        "DateTime,Air Temp [F],Wind Speed [m/s],Precipitation [mm],"
        "Barometric Pressure [Pa]\n"
        "2023-06-01 00:00,50.0,10.0,5.0,101300\n"
    )
    df = load_weather_csv(io.StringIO(csv_text))
    assert df["Air_Temp_C"].iloc[0] == pytest.approx(10.0)  # 50 F -> 10 C
    assert df["Wind_Speed_km_h"].iloc[0] == pytest.approx(36.0)  # 10 m/s -> 36 km/h
    assert df["Precipitation_cm"].iloc[0] == pytest.approx(0.5)  # 5 mm -> 0.5 cm
    assert df["Barometric_Pressure_mbar"].iloc[0] == pytest.approx(1013.0)  # Pa -> mbar


def test_load_weather_csv_column_map_override():
    csv_text = "when,degrees_air,breeze\n2023-06-01 00:00,10.0,12.0\n"
    df = load_weather_csv(
        io.StringIO(csv_text),
        column_map={
            "when": "DateTime",
            "degrees_air": "Air_Temp_C",
            "breeze": "Wind_Speed_km_h",
        },
    )
    assert df["DateTime"].iloc[0] == pd.Timestamp("2023-06-01 00:00")
    assert df["Air_Temp_C"].iloc[0] == 10.0
    assert df["Wind_Speed_km_h"].iloc[0] == 12.0


def test_load_weather_csv_missing_datetime_raises():
    csv_text = "Air Temp [C],Humidity [%]\n10.0,80\n"
    with pytest.raises(WeatherError):
        load_weather_csv(io.StringIO(csv_text))


# ---------------------------------------------------------------------------
# merge_weather_into_sensor
# ---------------------------------------------------------------------------
def test_merge_fills_gaps_without_overwriting_when_prefer_sensor():
    sensor = pd.DataFrame(
        {
            "DateTime": pd.to_datetime(
                ["2023-06-01 00:00", "2023-06-01 01:00", "2023-06-01 02:00"]
            ),
            "Air_Temp_C": [np.nan, 5.0, np.nan],  # middle value must be preserved
            "Gate_Opening": [10, 20, 30],  # non-weather column preserved
        }
    )
    weather_df = pd.DataFrame(
        {
            "DateTime": pd.to_datetime(
                ["2023-06-01 00:05", "2023-06-01 01:03", "2023-06-01 02:10"]
            ),
            "Air_Temp_C": [9.0, 9.0, 9.0],
            "Humidity_pct": [50.0, 55.0, 60.0],
        }
    )

    merged = merge_weather_into_sensor(
        sensor, weather_df, tolerance_minutes=30, prefer="sensor"
    )
    # existing sensor value kept; NaN gaps filled from weather
    assert merged["Air_Temp_C"].tolist() == [9.0, 5.0, 9.0]
    # new weather column added
    assert "Humidity_pct" in merged.columns
    assert merged["Humidity_pct"].tolist() == [50.0, 55.0, 60.0]
    # non-weather sensor column preserved and order intact
    assert merged["Gate_Opening"].tolist() == [10, 20, 30]


def test_merge_prefer_weather_overwrites():
    sensor = pd.DataFrame(
        {
            "DateTime": pd.to_datetime(["2023-06-01 00:00", "2023-06-01 01:00"]),
            "Air_Temp_C": [1.0, 2.0],
        }
    )
    weather_df = pd.DataFrame(
        {
            "DateTime": pd.to_datetime(["2023-06-01 00:00", "2023-06-01 01:00"]),
            "Air_Temp_C": [9.0, 9.5],
        }
    )
    merged = merge_weather_into_sensor(sensor, weather_df, prefer="weather")
    assert merged["Air_Temp_C"].tolist() == [9.0, 9.5]


def test_merge_respects_tolerance():
    sensor = pd.DataFrame(
        {
            "DateTime": pd.to_datetime(["2023-06-01 00:00"]),
            "Air_Temp_C": [np.nan],
        }
    )
    weather_df = pd.DataFrame(
        {
            "DateTime": pd.to_datetime(["2023-06-01 03:00"]),  # 3h away
            "Air_Temp_C": [9.0],
        }
    )
    merged = merge_weather_into_sensor(sensor, weather_df, tolerance_minutes=30)
    assert pd.isna(merged["Air_Temp_C"].iloc[0])  # outside tolerance -> not filled


def test_merge_errors_on_empty_or_missing_datetime():
    good = pd.DataFrame(
        {"DateTime": pd.to_datetime(["2023-06-01 00:00"]), "Air_Temp_C": [1.0]}
    )
    with pytest.raises(WeatherError):
        merge_weather_into_sensor(pd.DataFrame(), good)
    with pytest.raises(WeatherError):
        merge_weather_into_sensor(good, pd.DataFrame())
    with pytest.raises(WeatherError):
        merge_weather_into_sensor(good, pd.DataFrame({"NoTime": [1]}))
    with pytest.raises(WeatherError):
        merge_weather_into_sensor(good, good, prefer="invalid")


def test_merge_keeps_rows_with_unparseable_datetime():
    sensor = pd.DataFrame(
        {
            "DateTime": ["2023-06-01 00:00", "not-a-date"],
            "Air_Temp_C": [np.nan, np.nan],
        }
    )
    weather_df = pd.DataFrame(
        {
            "DateTime": pd.to_datetime(["2023-06-01 00:00"]),
            "Air_Temp_C": [7.0],
        }
    )
    merged = merge_weather_into_sensor(sensor, weather_df)
    assert len(merged) == 2  # invalid row retained
    assert merged["Air_Temp_C"].iloc[0] == 7.0
    assert pd.isna(merged["Air_Temp_C"].iloc[1])


# ---------------------------------------------------------------------------
# resample_hourly
# ---------------------------------------------------------------------------
def test_resample_hourly_sums_precip_means_temp():
    df = pd.DataFrame(
        {
            "DateTime": pd.to_datetime(
                [
                    "2023-06-01 00:00",
                    "2023-06-01 00:30",
                    "2023-06-01 01:00",
                ]
            ),
            "Air_Temp_C": [10.0, 20.0, 5.0],
            "Precipitation_cm": [0.1, 0.3, 1.0],
        }
    )
    out = resample_hourly(df)
    hour0 = out[out["DateTime"] == pd.Timestamp("2023-06-01 00:00")].iloc[0]
    assert hour0["Air_Temp_C"] == pytest.approx(15.0)  # mean of 10 & 20
    assert hour0["Precipitation_cm"] == pytest.approx(0.4)  # sum of 0.1 & 0.3


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_contents_and_get_source():
    assert set(weather.SOURCES) == {"open-meteo", "noaa", "synoptic"}
    assert weather.SOURCES["open-meteo"] is OpenMeteoSource

    src = weather.get_source("open-meteo")
    assert isinstance(src, OpenMeteoSource)
    assert weather.get_source("noaa").name == "noaa"

    with pytest.raises(WeatherConfigError):
        weather.get_source("does-not-exist")


def test_public_api_importable():
    for name in weather.__all__:
        assert hasattr(weather, name), f"weather.{name} should be importable"
