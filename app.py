"""Tidegate Wildlife & Tide Gate Analysis — multi-page Streamlit entry point.

Run with::

    streamlit run app.py

The app is organized as a pipeline:

1. **Image Detection** — turn camera-trap images into a species dataset using a
   demo detector, local MegaDetector/SpeciesNet, or an OpenAI GPT vision model.
2. **Weather Import** — pull weather (and tide) data from Open-Meteo / NOAA /
   Synoptic, or upload a CSV, and merge it with the sensor data.
3. **Analysis** — run the dual-framework wildlife/tide-gate analysis and explore
   interactive results.

The legacy single-page app (``main.py``) still works, but ``app.py`` is the new
front door.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Tidegate Wildlife & Tide Gate Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

import ui_common  # noqa: E402  (must follow set_page_config)

ui_common.inject_theme_css()

home = st.Page("views/home.py", title="Home", icon=":material/home:", default=True)
detection = st.Page("views/detection.py", title="Image Detection", icon=":material/photo_camera:")
weather = st.Page("views/weather_import.py", title="Weather Import", icon=":material/rainy:")
analysis = st.Page("views/analysis.py", title="Analysis", icon=":material/insights:")
help_page = st.Page("views/help.py", title="Help & Docs", icon=":material/help:")

pg = st.navigation(
    {
        "Overview": [home],
        "Pipeline": [detection, weather, analysis],
        "Reference": [help_page],
    }
)

# Shared sidebar branding + live session status (rendered on every page).
with st.sidebar:
    st.markdown("## 🐦 Tidegate Analysis")
    st.caption("Wildlife detection · tide gates · weather patterns")
    st.divider()

ui_common.sidebar_status()

with st.sidebar:
    st.divider()
    st.caption("Research preview · [GitHub](https://github.com/cdavisv/Tidegate-Analysis-App)")

pg.run()
