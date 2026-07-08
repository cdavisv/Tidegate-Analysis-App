"""Weather Import page: fetch/import weather & tide data and merge with sensors."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

import ui_common as ui
import data_loader
from weather import (
    SOURCES,
    get_source,
    load_weather_csv,
    merge_weather_into_sensor,
    WeatherError,
)

cfg = ui.get_config()

ui.app_header(
    "Weather & Tide Import",
    "Pull weather and tide data from free sources and align it to your sensor timeline.",
    icon="🌦️",
)
ui.workflow_chips(active="weather")

# -------------------------------------------------------------------------
# Determine a sensible default date range from any loaded sensor data.
# -------------------------------------------------------------------------
def _sensor_date_range():
    df = st.session_state.get(ui.K_SENSOR_DF)
    if df is not None and "DateTime" in df.columns:
        try:
            dt = pd.to_datetime(df["DateTime"], errors="coerce").dropna()
            if len(dt):
                return dt.min().date(), dt.max().date()
        except Exception:
            pass
    today = date.today()
    return today - timedelta(days=30), today


d_start, d_end = _sensor_date_range()

# -------------------------------------------------------------------------
# 1. Site & date range
# -------------------------------------------------------------------------
st.subheader("1 · Site & date range")
c1, c2, c3 = st.columns(3)
lat = c1.number_input("Latitude", value=float(cfg.latitude), format="%.5f")
lon = c2.number_input("Longitude", value=float(cfg.longitude), format="%.5f")
tz = c3.text_input("Timezone", value=cfg.timezone)
c4, c5 = st.columns(2)
start = c4.date_input("Start date", value=d_start)
end = c5.date_input("End date", value=d_end)
st.caption(f"Site: **{cfg.site_name}** · defaults are editable and can be saved in `config.json`.")

st.divider()

# -------------------------------------------------------------------------
# 2. Source
# -------------------------------------------------------------------------
st.subheader("2 · Choose a source")
source_names = list(SOURCES.keys()) + ["csv-upload"]
source_labels = {
    "open-meteo": "Open-Meteo — free, no key (recommended)",
    "noaa": "NOAA — NWS weather + CO-OPS tide (free)",
    "synoptic": "Synoptic / MesoWest — free token",
    "csv-upload": "Upload a weather CSV",
}
source = st.radio("Weather source", source_names,
                  format_func=lambda k: source_labels.get(k, k))

weather_df = None
fetch_kwargs: dict = {}

if source == "synoptic":
    token = st.text_input("Synoptic token", value=cfg.synoptic_token or "", type="password",
                          help="Free token from https://synopticdata.com. Or set SYNOPTIC_TOKEN.")
    fetch_kwargs["token"] = token or None
elif source == "noaa":
    st.text_input("NOAA CO-OPS tide station (optional)", value=cfg.noaa_tide_station,
                  key="noaa_station",
                  help="e.g. 9432780 (Charleston, OR) to also import water level.")

if source == "csv-upload":
    up = st.file_uploader("Weather CSV", type=["csv"],
                          help="Headers like 'Air Temp [C]', 'Wind Speed [km/h]', 'Precipitation [cm]', "
                          "plus a DateTime (or Date + Time) column. Units in brackets are auto-detected.")
    if up is not None:
        try:
            weather_df = load_weather_csv(up)
            st.success(f"Parsed {len(weather_df):,} rows.")
        except WeatherError as exc:
            st.error(f"Could not parse weather CSV: {exc}")
else:
    if st.button("Fetch weather data", type="primary"):
        try:
            with st.spinner(f"Fetching from {source}…"):
                src = get_source(source, **{k: v for k, v in fetch_kwargs.items() if v is not None})
                if source == "noaa":
                    weather_df = src.fetch(lat, lon, start, end,
                                           tide_station=st.session_state.get("noaa_station") or None)
                else:
                    weather_df = src.fetch(lat, lon, start, end)
            st.session_state["_weather_fetch"] = weather_df
            st.success(f"Fetched {len(weather_df):,} rows from {source}.")
        except WeatherError as exc:
            st.error(f"Fetch failed: {exc}")
        except Exception as exc:  # network blocked, etc.
            st.error(f"Fetch failed: {exc}")

# Reuse a prior fetch stored in session (survives reruns)
if weather_df is None:
    weather_df = st.session_state.get("_weather_fetch")

# -------------------------------------------------------------------------
# 3. Preview & merge
# -------------------------------------------------------------------------
if weather_df is not None and not weather_df.empty:
    st.divider()
    st.subheader("3 · Preview & merge")
    st.session_state[ui.K_WEATHER_DF] = weather_df
    st.session_state[ui.K_WEATHER_SRC] = f"{source} ({len(weather_df):,} rows)"

    cols_present = [c for c in weather_df.columns if weather_df[c].notna().any()]
    st.caption("Variables with data: " + ", ".join(c for c in cols_present if c != "DateTime"))
    st.dataframe(weather_df.head(200), use_container_width=True, height=280)
    st.download_button("Download normalized weather CSV",
                       data=weather_df.to_csv(index=False),
                       file_name=f"weather_{source}.csv", mime="text/csv")

    st.markdown("**Merge onto sensor timeline**")
    sensor_loaded = st.session_state.get(ui.K_SENSOR_DF) is not None
    tol = st.slider("Match tolerance (minutes)", 5, 180, 30, 5,
                    help="Weather readings are matched to the nearest sensor timestamp within this window.")
    prefer = st.radio("On overlap, keep", ["sensor", "weather"], horizontal=True,
                      help="'sensor' only fills gaps; 'weather' overwrites matching sensor columns.")

    mc1, mc2 = st.columns(2)
    with mc1:
        if st.button("Merge into loaded sensor data", disabled=not sensor_loaded,
                     use_container_width=True):
            try:
                # Normalize the base sensor first so column names line up and
                # we don't create duplicate columns downstream.
                import tempfile, os
                raw = st.session_state[ui.K_SENSOR_DF]
                with tempfile.TemporaryDirectory() as tmp:
                    p = os.path.join(tmp, "sensor.csv")
                    raw.to_csv(p, index=False)
                    base = data_loader.load_and_prepare_water_data(p)
                merged = merge_weather_into_sensor(base, weather_df,
                                                   tolerance_minutes=tol, prefer=prefer)
                st.session_state[ui.K_SENSOR_DF] = merged
                st.session_state[ui.K_SENSOR_SRC] = (
                    f"sensor + {source} weather ({merged.shape[1]} cols)"
                )
                st.success(f"Merged. Sensor table now has {merged.shape[1]} columns.")
            except Exception as exc:
                st.error(f"Merge failed: {exc}")
        if not sensor_loaded:
            st.caption("Load or upload sensor/tide data first (Analysis page or the demo quick-start).")
    with mc2:
        if st.button("Use weather/tide as the sensor data", use_container_width=True):
            wx = weather_df.copy()
            if "Water_Level_m" in wx.columns and "Depth" not in wx.columns:
                wx["Depth"] = wx["Water_Level_m"]
            st.session_state[ui.K_SENSOR_DF] = wx
            st.session_state[ui.K_SENSOR_SRC] = f"{source} weather/tide only"
            st.success("Weather/tide data set as the sensor table for analysis.")

    st.page_link("views/analysis.py", label="Go to Analysis →", icon=":material/insights:")

with st.expander("About the sources"):
    st.markdown(
        "- **Open-Meteo** — free, no API key; historical archive + forecast. Best default.\n"
        "- **NOAA** — NWS station observations, plus CO-OPS tide water level when you give a "
        "station id (e.g. 9432780 Charleston, OR).\n"
        "- **Synoptic / MesoWest** — dense station network; needs a free token (`SYNOPTIC_TOKEN`).\n"
        "- **CSV upload** — bring your own weather station export; bracketed-unit headers "
        "(`Air Temp [C]`, `Wind Speed [km/h]`) are auto-detected.\n\n"
        "All sources normalize to the same columns: `Air_Temp_C`, `Wind_Speed_km_h`, "
        "`Precipitation_cm`, `Barometric_Pressure_mbar`, `Humidity_pct`, `Solar_Radiation_W_m2`, "
        "`Water_Level_m`."
    )
