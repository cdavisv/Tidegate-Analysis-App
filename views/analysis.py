"""Analysis page: run the dual-framework pipeline and explore results."""

from __future__ import annotations

import contextlib
import glob
import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import ui_common as ui
import pipeline_runner

ui.app_header(
    "Wildlife & Tide Gate Analysis",
    "Separate operational bias from real behavior across tides, gates, and weather.",
    icon="📊",
)
ui.workflow_chips(active="analyze")

# -------------------------------------------------------------------------
# Data inputs — prefer session data (from Detection / Weather / demo), allow
# uploads to override.
# -------------------------------------------------------------------------
st.subheader("Data inputs")

cam_sess = st.session_state.get(ui.K_CAMERA_DF)
sen_sess = st.session_state.get(ui.K_SENSOR_DF)

col_cam, col_wat = st.columns(2)
with col_cam:
    if cam_sess is not None:
        st.success(f"Camera data ready — {st.session_state.get(ui.K_CAMERA_SRC, 'session')} "
                   f"({len(cam_sess):,} rows)")
    else:
        st.info("No camera data yet. Upload one, or use Image Detection / the demo quick-start.")
    cam_upload = st.file_uploader("Camera CSV (optional override)", type=["csv"], key="an_cam")

with col_wat:
    if sen_sess is not None:
        st.success(f"Sensor data ready — {st.session_state.get(ui.K_SENSOR_SRC, 'session')} "
                   f"({len(sen_sess):,} rows)")
    else:
        st.info("No sensor/tide data yet. Upload one, or import it on the Weather page.")
    wat_upload = st.file_uploader("Water / Tide / Sensor CSV (optional override)", type=["csv"], key="an_wat")

camera_input = cam_upload if cam_upload is not None else cam_sess
water_input = wat_upload if wat_upload is not None else sen_sess

ready = camera_input is not None and water_input is not None
if not ready:
    st.caption("Provide both a camera dataset and a sensor/tide dataset to run the analysis.")

if st.button("Run full analysis", type="primary", disabled=not ready):
    log_buffer = io.StringIO()
    with st.status("Running analysis pipeline…", expanded=True) as status:
        def _p(msg):
            status.write(msg)
        try:
            with contextlib.redirect_stdout(log_buffer):
                results = pipeline_runner.run_full_analysis(camera_input, water_input, progress=_p)
            st.session_state[ui.K_ANALYSIS] = results
            st.session_state[ui.K_ANALYSIS_LOG] = log_buffer.getvalue()
            status.update(label="Analysis complete!", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Analysis failed", state="error")
            st.error(f"Pipeline error: {exc}")
            st.session_state[ui.K_ANALYSIS_LOG] = log_buffer.getvalue()

# -------------------------------------------------------------------------
# Results
# -------------------------------------------------------------------------
results = st.session_state.get(ui.K_ANALYSIS)
if not results:
    st.stop()

comparison = results["comprehensive"]["comparison"]
figures = results["figures"]
combined = results["combined_df"]

st.divider()
st.header("Results dashboard")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total time periods", f"{comparison['total_periods']:,}")
m2.metric("Camera active periods", f"{comparison['camera_periods']:,}")
m3.metric("Animal detections", f"{comparison['animal_detections']:,}")
species_count = "N/A"
if results["species_summary"] is not None and not results["species_summary"].empty:
    species_count = str(len(results["species_summary"]))
m4.metric("Unique species", species_count)

m5, m6, m7, m8 = st.columns(4)
m5.metric("Camera activity rate", f"{comparison['camera_activity_rate']:.2f}%")
m6.metric("Overall detection rate", f"{comparison['overall_detection_rate']:.3f}%")
m7.metric("Camera detection rate", f"{comparison['camera_detection_rate']:.1f}%")
bird_results = results["bird_tide_results"]
if bird_results is not None and not bird_results.empty:
    m8.metric("Peak detection rate", f"{bird_results.values.max():.2f}%")
else:
    m8.metric("Peak detection rate", "N/A")

st.subheader("Key findings")
kf1, kf2 = st.columns(2)
with kf1:
    st.info(
        f"**Camera Activity Pattern Analysis**\n\n"
        f"Cameras operated during **{comparison['camera_activity_rate']:.2f}%** of all "
        f"monitoring periods — revealing equipment performance and operational bias."
    )
with kf2:
    st.success(
        f"**Wildlife Detection Efficiency**\n\n"
        f"When cameras were active, animals were detected "
        f"**{comparison['camera_detection_rate']:.1f}%** of the time — revealing wildlife "
        f"behavior and optimal monitoring conditions."
    )

if bird_results is not None and not bird_results.empty:
    peak_val = bird_results.values.max()
    peak_pos = np.where(bird_results.values == peak_val)
    if peak_pos[0].size and peak_pos[1].size:
        st.success(
            f"**Peak detection rate: {peak_val:.2f}%** when gate is "
            f"**{bird_results.index[peak_pos[0][0]]}** during "
            f"**{bird_results.columns[peak_pos[1][0]]}** tide."
        )

st.divider()

tabs = st.tabs([
    "Species", "Environmental", "Gate Interactions", "Tidal Cycles",
    "Weather Patterns", "Method Comparison", "Combined Dataset", "Console Log",
])

# ---- Species ----
with tabs[0]:
    st.subheader("Species diversity summary")
    ssum = results["species_summary"]
    if ssum is not None and not ssum.empty:
        top3 = ssum.head(3)
        cols = st.columns(min(3, len(top3)))
        for i, (name, row) in enumerate(top3.iterrows()):
            cols[i].metric(name, f"{int(row['Total_Count'])} individuals",
                           f"{int(row['Detection_Events'])} events")
        st.dataframe(ssum.reset_index().rename(columns={"index": "Species"}),
                     use_container_width=True)
        if "species_summary" in figures:
            st.plotly_chart(figures["species_summary"], use_container_width=True)
        if "species_pattern_comparison" in figures:
            st.plotly_chart(figures["species_pattern_comparison"], use_container_width=True)
        st.download_button("Download species summary CSV", ssum.to_csv(),
                           "species_summary.csv", "text/csv", key="dl_species")
    else:
        st.warning("No species data available.")

# ---- Environmental ----
with tabs[1]:
    st.subheader("Environmental factor analysis")
    env = results["env_results"]
    if env:
        mtr_df, hinge_df, tidal_df, temp_df = env
        for title, df_, figkey in [
            ("MTR Gate Detection Rates", mtr_df, "mtr_gate_detection_rate"),
            ("Top Hinge Gate Detection Rates", hinge_df, "hinge_gate_detection_rate"),
            ("Tidal Level Detection Rates", tidal_df, "tidal_level_detection_rate"),
        ]:
            if df_ is not None and not df_.empty:
                st.markdown(f"#### {title}")
                st.dataframe(df_.reset_index(), use_container_width=True)
                if figkey in figures:
                    st.plotly_chart(figures[figkey], use_container_width=True)
        water_keys = sorted(k for k in figures if k.startswith("water_quality_"))
        if water_keys:
            st.markdown("#### Water Quality Time Series")
            for key in water_keys:
                st.plotly_chart(figures[key], use_container_width=True)
        if "environmental_effectiveness" in figures:
            st.markdown("#### Environmental Effectiveness Dashboard")
            st.plotly_chart(figures["environmental_effectiveness"], use_container_width=True)
    else:
        st.warning("No environmental analysis results available.")

# ---- Gate Interactions ----
with tabs[2]:
    st.subheader("Wildlife & tide gate behavior")
    bt = results["bird_tide_results"]
    if bt is not None and not bt.empty:
        st.markdown("#### Detection Rate (%) by Gate Status and Tidal Flow")
        st.dataframe(bt.round(2), use_container_width=True)
        for figkey in ("wildlife_detection_heatmap", "wildlife_detection_scatter"):
            if figkey in figures:
                st.plotly_chart(figures[figkey], use_container_width=True)
        st.markdown("#### Hypothesis Test Visualizations")
        hyp_files = sorted(glob.glob("hypothesis_visualization_*.png"))
        if hyp_files:
            hyp_cols = st.columns(2)
            for i, fpath in enumerate(hyp_files):
                hyp_cols[i % 2].image(fpath, use_container_width=True)
        else:
            st.caption("No hypothesis visualization PNGs generated.")
        st.download_button("Download gate interaction summary CSV", bt.to_csv(),
                           "gate_interaction_summary.csv", "text/csv", key="dl_gate")
    else:
        st.warning("No gate interaction results available.")

# ---- Tidal Cycles ----
with tabs[3]:
    st.subheader("Tidal cycle detection analysis")
    det_tide = results["detection_by_tide"]
    phase_det = results["phase_detection"]
    sp_tide = results["species_tide_table"]
    if det_tide is not None and not det_tide.empty:
        st.markdown("#### Detection Rates by Tidal State")
        st.dataframe(det_tide.round(4), use_container_width=True)
        if "detection_by_tidal_state" in figures:
            st.plotly_chart(figures["detection_by_tidal_state"], use_container_width=True)
    if phase_det is not None and not phase_det.empty:
        st.markdown("#### Detection Rates by Tidal Phase")
        st.dataframe(phase_det.round(4), use_container_width=True)
        if "detection_rate" in phase_det.columns and phase_det["detection_rate"].sum() > 0:
            st.info(f"Peak detection rate: **{phase_det['detection_rate'].max():.1%}** at tidal "
                    f"phase **{phase_det['detection_rate'].idxmax()}**")
        if "detection_by_tidal_phase" in figures:
            st.plotly_chart(figures["detection_by_tidal_phase"], use_container_width=True)
    if sp_tide is not None and not sp_tide.empty:
        st.markdown("#### Species Tidal Preferences")
        st.dataframe(sp_tide.round(1), use_container_width=True)
        if "species_tide_preference_heatmap" in figures:
            st.plotly_chart(figures["species_tide_preference_heatmap"], use_container_width=True)
        st.download_button("Download species tide preferences CSV", sp_tide.to_csv(),
                           "species_tide_preferences.csv", "text/csv", key="dl_tideprefs")

# ---- Weather Patterns (new) ----
with tabs[4]:
    st.subheader("Weather influence on detections")
    st.caption("Detection rate across binned weather variables present in the combined dataset. "
               "Import weather on the Weather page to enrich this view.")
    weather_candidates = {
        "Air_Temp_C": "Air temperature (°C)",
        "Wind_Speed_km_h": "Wind speed (km/h)",
        "Precipitation_cm": "Precipitation (cm)",
        "Barometric_Pressure_mbar": "Barometric pressure (mbar)",
        "Humidity_pct": "Humidity (%)",
        "Solar_Radiation_W_m2": "Solar radiation (W/m²)",
        "Humidity [%]": "Humidity (%)",
        "Barometric Pressure [mbar]": "Barometric pressure (mbar)",
        "Precipitation [cm]": "Precipitation (cm)",
    }
    present = [c for c in weather_candidates if c in combined.columns and combined[c].notna().sum() > 20]
    if not present:
        st.info("No weather variables available yet. Use the **Weather Import** page to add some, "
                "or supply a sensor CSV that includes weather columns.")
    else:
        if "animal_detected" in combined.columns:
            cam = combined[combined["has_camera_data"] == True].copy()  # noqa: E712
            var = st.selectbox("Weather variable", present,
                               format_func=lambda c: weather_candidates.get(c, c))
            series = pd.to_numeric(cam[var], errors="coerce")
            valid = cam.assign(_v=series).dropna(subset=["_v"])
            if len(valid) > 20:
                try:
                    valid["_bin"] = pd.qcut(valid["_v"], q=min(6, valid["_v"].nunique()), duplicates="drop")
                    grp = valid.groupby("_bin", observed=True).agg(
                        detection_rate=("animal_detected", "mean"),
                        observations=("animal_detected", "size"),
                    ).reset_index()
                    grp["detection_rate_pct"] = grp["detection_rate"] * 100
                    grp["bin_label"] = grp["_bin"].astype(str)
                    fig = px.bar(grp, x="bin_label", y="detection_rate_pct",
                                 hover_data=["observations"],
                                 labels={"bin_label": weather_candidates.get(var, var),
                                         "detection_rate_pct": "Detection rate (%)"},
                                 title=f"Detection rate by {weather_candidates.get(var, var)}",
                                 color="detection_rate_pct", color_continuous_scale="Teal")
                    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    corr = valid["_v"].corr(valid["animal_detected"].astype(float))
                    st.caption(f"Point-biserial correlation with detection: **{corr:+.3f}** "
                               f"(n={len(valid):,} camera observations).")
                except Exception as exc:
                    st.warning(f"Could not bin this variable: {exc}")
            else:
                st.info("Not enough camera observations with this variable to analyze.")

# ---- Method Comparison ----
with tabs[5]:
    st.subheader("Analysis method comparison")
    comp_results = results["comprehensive"]
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("#### Camera Activity Pattern Analysis")
        st.markdown("- Uses **all** time periods (including sensor-only data)\n"
                    "- Measures: Camera Active / All Time Periods\n"
                    "- Reveals: Equipment performance and monitoring bias")
        act = comp_results["camera_activity"]["species_summary"]
        if act is not None and not act.empty:
            st.dataframe(act.head(10).reset_index(), use_container_width=True)
    with mc2:
        st.markdown("#### Wildlife Detection Efficiency Analysis")
        st.markdown("- Uses **only** camera observation periods\n"
                    "- Measures: Animal Detections / Camera Observations\n"
                    "- Reveals: Wildlife behavior and optimal conditions")
        eff = comp_results["detection_efficiency"]["species_summary"]
        if eff is not None and not eff.empty:
            st.dataframe(eff.head(10).reset_index(), use_container_width=True)
    for key, title in [
        ("data_overview_dashboard", "Data Overview Dashboard"),
        ("analysis_method_comparison", "Analysis Method Comparison"),
        ("temporal_analysis", "Temporal Activity Patterns"),
        ("camera_performance_dashboard", "Camera Performance Dashboard"),
    ]:
        if key in figures:
            st.markdown(f"#### {title}")
            st.plotly_chart(figures[key], use_container_width=True)

# ---- Combined Dataset ----
with tabs[6]:
    st.subheader("Combined dataset preview")
    st.markdown(f"**{len(combined):,} rows × {len(combined.columns)} columns**")
    all_cols = combined.columns.tolist()
    default_show = [c for c in ["DateTime", "Species", "Count", "has_camera_data",
                                "animal_detected", "Gate_Opening_MTR_Deg", "Depth"]
                    if c in all_cols]
    selected = st.multiselect("Columns to display", all_cols, default=default_show)
    if selected:
        st.dataframe(combined[selected].head(500), use_container_width=True, height=400)
    st.download_button("Download combined dataset CSV", combined.to_csv(index=False),
                       "combined_data_output.csv", "text/csv", key="dl_combined")

# ---- Console Log ----
with tabs[7]:
    st.subheader("Analysis console log")
    log_text = st.session_state.get(ui.K_ANALYSIS_LOG, "")
    st.caption(f"Console output: {log_text.count(chr(10))} lines")
    search = st.text_input("Search log", placeholder="Filter log output…")
    if search:
        filtered = [ln for ln in log_text.split("\n") if search.lower() in ln.lower()]
        st.code("\n".join(filtered), language="text")
    else:
        st.code(log_text, language="text")
    st.download_button("Download analysis log", log_text, "analysis_log.txt", "text/plain",
                       key="dl_log")
