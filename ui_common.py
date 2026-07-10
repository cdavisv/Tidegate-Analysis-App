"""Shared Streamlit UI helpers: theme, header, session keys, sidebar status.

Centralizes styling and the session-state contract used to hand data between
pages (Image Detection -> Weather Import -> Analysis).
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st

from config import load_config

# --- Session-state keys (single source of truth) -------------------------
K_CAMERA_DF = "camera_df"          # wide-format camera table (pre-loader)
K_CAMERA_SRC = "camera_source"     # human description of where it came from
K_DETECTIONS = "detections"        # list[ImageDetection] from a detector run
K_SENSOR_DF = "sensor_df"          # water/tide/sensor table (pre-loader)
K_SENSOR_SRC = "sensor_source"
K_WEATHER_DF = "weather_df"        # normalized weather (optional)
K_WEATHER_SRC = "weather_source_desc"
K_ANALYSIS = "analysis_results"    # dict from pipeline_runner.run_full_analysis
K_ANALYSIS_LOG = "analysis_log"

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# Bundled sample files shipped with the repo (used by the demo quick-starts).
SAMPLE_CAMERA_CSV = os.path.join(REPO_DIR, "willanch_camera_final.csv")
SAMPLE_SENSOR_CSV = os.path.join(REPO_DIR, "willanch_sensor_final.csv")


def get_config():
    """Return a cached :class:`AppConfig` stored in session state."""
    if "app_config" not in st.session_state:
        st.session_state.app_config = load_config()
    return st.session_state.app_config


def inject_theme_css() -> None:
    """Inject shared CSS for a cohesive, polished look."""
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2.2rem; max-width: 1200px; }
          /* Header banner */
          .tg-banner {
            background: linear-gradient(120deg, #1f8a70 0%, #2aa5a0 55%, #3ab0c9 100%);
            color: #ffffff; padding: 1.1rem 1.4rem; border-radius: 14px;
            margin-bottom: 1.2rem; box-shadow: 0 6px 22px rgba(31,138,112,0.22);
          }
          .tg-banner h1 { color:#fff; font-size: 1.55rem; margin: 0 0 .15rem 0; font-weight: 700; }
          .tg-banner p  { color: #eafaf4; margin: 0; font-size: .96rem; }
          /* Step chips */
          .tg-steps { display:flex; gap:.5rem; flex-wrap:wrap; margin:.2rem 0 1rem 0; }
          .tg-chip { background:#eef5f2; color:#16302b; border:1px solid #cfe4dc;
            border-radius: 999px; padding:.22rem .7rem; font-size:.8rem; }
          .tg-chip.active { background:#1f8a70; color:#fff; border-color:#1f8a70; }
          .tg-chip.done   { background:#d6efe6; color:#0f5a45; border-color:#9ed7c3; }
          /* Cards */
          .tg-card { background:#fff; border:1px solid #e6ece9; border-radius:12px;
            padding:1rem 1.1rem; box-shadow:0 2px 10px rgba(20,48,43,0.05); }
          section[data-testid="stSidebar"] { background: #f6faf8; }
          div[data-testid="stMetricValue"] { font-size: 1.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def app_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """Render the gradient banner header used at the top of each page."""
    icon_html = f"{icon} " if icon else ""
    st.markdown(
        f"""
        <div class="tg-banner">
          <h1>{icon_html}{title}</h1>
          {f'<p>{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def workflow_chips(active: str) -> None:
    """Render the 3-step workflow breadcrumb with the current step highlighted.

    Steps: ``detect`` -> ``weather`` -> ``analyze``.
    """
    steps = [("detect", "1 · Detect / Import images"),
             ("weather", "2 · Import weather"),
             ("analyze", "3 · Analyze patterns")]
    done_map = {
        "detect": st.session_state.get(K_CAMERA_DF) is not None,
        "weather": st.session_state.get(K_SENSOR_DF) is not None,
        "analyze": st.session_state.get(K_ANALYSIS) is not None,
    }
    html = ['<div class="tg-steps">']
    for key, label in steps:
        cls = "tg-chip"
        if key == active:
            cls += " active"
        elif done_map.get(key):
            cls += " done"
        html.append(f'<span class="{cls}">{label}</span>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def sidebar_status() -> None:
    """Shared sidebar panel showing what data is currently loaded."""
    with st.sidebar:
        st.markdown("### Session data")
        cam = st.session_state.get(K_CAMERA_DF)
        sen = st.session_state.get(K_SENSOR_DF)
        wx = st.session_state.get(K_WEATHER_DF)
        an = st.session_state.get(K_ANALYSIS)

        def _row(label, ok, detail=""):
            mark = "✅" if ok else "⬜"
            st.caption(f"{mark} **{label}**" + (f" — {detail}" if ok and detail else ""))

        _row("Camera data", cam is not None,
             st.session_state.get(K_CAMERA_SRC, ""))
        _row("Sensor / tide data", sen is not None,
             st.session_state.get(K_SENSOR_SRC, ""))
        _row("Weather imported", wx is not None,
             st.session_state.get(K_WEATHER_SRC, ""))
        _row("Analysis complete", an is not None)

        if cam is not None or sen is not None or an is not None:
            if st.button("Reset session", use_container_width=True):
                for k in [K_CAMERA_DF, K_CAMERA_SRC, K_DETECTIONS, K_SENSOR_DF,
                          K_SENSOR_SRC, K_WEATHER_DF, K_WEATHER_SRC, K_ANALYSIS,
                          K_ANALYSIS_LOG]:
                    st.session_state.pop(k, None)
                st.rerun()


def page_link(path: str, label: str, icon: Optional[str] = None, **kwargs) -> None:
    """Safe wrapper around ``st.page_link``.

    ``st.page_link`` raises ``KeyError('url_pathname')`` when a page is rendered
    outside an ``st.navigation`` context (e.g. a single page opened directly, or
    in a test harness). This wrapper degrades to a simple link/caption instead of
    crashing the whole page.
    """
    try:
        st.page_link(path, label=label, icon=icon, **kwargs)
    except Exception:
        st.caption(f"→ {label}")


def load_sample_data() -> None:
    """Load the bundled Willanch demo CSVs (camera + sensor) into session state."""
    import pandas as pd
    st.session_state[K_CAMERA_DF] = pd.read_csv(
        SAMPLE_CAMERA_CSV, low_memory=False, encoding="utf-8-sig"
    )
    st.session_state[K_CAMERA_SRC] = "Willanch demo camera CSV"
    st.session_state[K_SENSOR_DF] = pd.read_csv(
        SAMPLE_SENSOR_CSV, low_memory=False, encoding="utf-8-sig"
    )
    st.session_state[K_SENSOR_SRC] = "Willanch demo sensor CSV"


def has_sample_data() -> bool:
    return os.path.exists(SAMPLE_CAMERA_CSV) and os.path.exists(SAMPLE_SENSOR_CSV)
