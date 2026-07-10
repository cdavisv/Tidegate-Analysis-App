"""Home / overview page: what the app does + one-click demo quick-starts."""

from __future__ import annotations

import streamlit as st

import ui_common as ui

ui.app_header(
    "Wildlife, Tide Gates & Weather Patterns",
    "Detect species in camera-trap images, import weather & tide data, and reveal how "
    "bird activity tracks the tidal cycle and gate operations.",
    icon="🐦",
)

ui.workflow_chips(active="detect")

st.markdown(
    "This tool builds a complete pipeline — from raw trail-camera images all the way "
    "to ecological insight:"
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        "#### 📷 1. Detect\n"
        "Turn images into a species dataset with a **demo** detector, local "
        "**MegaDetector + SpeciesNet** (AddaxAI-style), or an **OpenAI GPT** vision model."
    )
    ui.page_link("views/detection.py", label="Go to Image Detection", icon=":material/photo_camera:")
with c2:
    st.markdown(
        "#### 🌦️ 2. Import weather\n"
        "Pull weather & tide data from **Open-Meteo**, **NOAA**, or **Synoptic** "
        "(or upload a CSV) and merge it onto your sensor timeline."
    )
    ui.page_link("views/weather_import.py", label="Go to Weather Import", icon=":material/rainy:")
with c3:
    st.markdown(
        "#### 📊 3. Analyze\n"
        "Run the **dual-framework** analysis to separate camera/operational bias from "
        "true wildlife behavior across tides and gate positions."
    )
    ui.page_link("views/analysis.py", label="Go to Analysis", icon=":material/insights:")

st.divider()

# ---- Quick starts -------------------------------------------------------
st.subheader("Quick start")
q1, q2 = st.columns(2)

with q1:
    st.markdown(
        "**Run the bundled demo end-to-end**\n\n"
        "Uses the real Willanch Slough camera + sensor CSVs shipped with the repo. "
        "One click loads the data *and* runs the full dual-framework analysis."
    )
    if ui.has_sample_data():
        if st.button("▶ Load demo & run full analysis", type="primary",
                     use_container_width=True):
            with st.status("Running the demo pipeline…", expanded=True) as _s:
                _s.write("Loading bundled Willanch CSVs…")
                ui.load_sample_data()
                _s.write("Running dual-framework analysis (~30s)…")
                try:
                    import pipeline_runner
                    results = pipeline_runner.run_full_analysis(
                        st.session_state[ui.K_CAMERA_DF],
                        st.session_state[ui.K_SENSOR_DF],
                        progress=lambda m: _s.write(m),
                    )
                    st.session_state[ui.K_ANALYSIS] = results
                    _s.update(label="Demo analysis complete — open Analysis to explore.",
                              state="complete", expanded=False)
                except Exception as exc:
                    _s.update(label="Analysis failed", state="error")
                    st.error(f"Pipeline error: {exc}")
        if st.button("Load demo data only", use_container_width=True):
            ui.load_sample_data()
            st.success("Demo data loaded. Open the **Analysis** page and click Run.")
        ui.page_link("views/analysis.py", label="Go to Analysis →", icon=":material/insights:")
    else:
        st.info("Bundled sample CSVs not found in this checkout.")

with q2:
    st.markdown(
        "**Try image detection**\n\n"
        "Point the detector at a folder of images or upload a few, and watch it build "
        "the camera dataset. The **demo** detector needs no models or API keys."
    )
    ui.page_link("views/detection.py", label="Open Image Detection →", icon=":material/photo_camera:")

st.divider()
with st.expander("About the dual-framework method"):
    st.markdown(
        "- **Camera Activity Pattern Analysis** uses *all* time periods and measures when "
        "cameras were operational — exposing equipment/operational bias.\n"
        "- **Wildlife Detection Efficiency Analysis** restricts to camera-active periods and "
        "measures how often animals were detected — revealing genuine behavior.\n\n"
        "Comparing the two separates *how we watched* from *what the animals did*. "
        "See the **Help & Docs** page for the full methodology and data formats."
    )
