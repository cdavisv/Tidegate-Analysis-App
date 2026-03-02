"""
Main execution pipeline for the comprehensive wildlife monitoring analysis.

This script orchestrates the entire workflow, including:
- Data loading and validation
- Camera and sensor data integration
- Dual-framework comprehensive analysis
- Specialized environmental and gate interaction analyses
- Visualization generation
- Summary report output

The pipeline is designed to compare Camera Activity Pattern Analysis with
Wildlife Detection Efficiency Analysis to clearly separate operational bias
from biological detection behavior.
"""
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import glob
import io
import contextlib

import data_loader
import data_combiner
import species_analysis
import environmental_analysis
import comprehensive_analysis
import gate_combination_analysis
import bird_tide_analysis
import tide_cycle_analysis
import visualization
import additional_visualizations


# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Wildlife & Tide Gate Analysis",
    layout="wide",
)


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def validate_camera_columns(df):
    """Validates that a camera CSV has the required columns.

    Args:
        df: pd.DataFrame loaded from the uploaded camera CSV.

    Returns:
        tuple: (is_valid: bool, messages: list[str])
    """
    messages = []
    has_datetime = "DateTime" in df.columns or (
        "Date" in df.columns and "Time" in df.columns
    )
    has_species = any(col.startswith("Species ") for col in df.columns)

    if not has_datetime:
        messages.append("Missing required column: 'DateTime' (or 'Date' + 'Time').")
    else:
        messages.append("DateTime column detected.")

    if not has_species:
        messages.append(
            "Missing required columns: 'Species 1', 'Species 2', etc."
        )
    else:
        species_cols = [
            c
            for c in df.columns
            if c.startswith("Species ")
            and len(c.split()) == 2
            and c.split()[1].isdigit()
        ]
        messages.append(f"Found {len(species_cols)} species column(s).")

    return has_datetime and has_species, messages


def validate_water_columns(df):
    """Validates that a water/tide CSV has the required columns.

    Args:
        df: pd.DataFrame loaded from the uploaded water CSV.

    Returns:
        tuple: (is_valid: bool, messages: list[str])
    """
    messages = []
    has_datetime = "DateTime" in df.columns or (
        "Date" in df.columns and "Time" in df.columns
    )

    if not has_datetime:
        messages.append("Missing required column: 'DateTime' (or 'Date' + 'Time').")
    else:
        messages.append("DateTime column detected.")

    known_cols = {
        "Gate Opening MTR [Degrees]": "MTR Gate Angle",
        "Gate_Opening_MTR_Deg": "MTR Gate Angle",
        "Gate Opening Top Hinge [Degrees]": "Top Hinge Gate Angle",
        "Gate_Opening_Top_Hinge_Deg": "Top Hinge Gate Angle",
        "Tidal Level Outside Tidegate [m]": "Tidal Depth",
        "Depth": "Tidal Depth",
        "Air Temp [C]": "Air Temperature",
        "Air_Temp_C": "Air Temperature",
    }

    detected = sorted(set(known_cols[c] for c in df.columns if c in known_cols))
    if detected:
        messages.append(f"Detected parameters: {', '.join(detected)}.")
    else:
        messages.append("No recognized environmental columns found.")

    return has_datetime, messages


def run_pipeline(camera_path, water_path, status_container):
    """Executes the full analysis pipeline from data loading through visualization.

    Args:
        camera_path: File path to the uploaded camera CSV.
        water_path: File path to the uploaded water/tide CSV.
        status_container: Streamlit status context for progress updates.

    Returns:
        dict: All analysis results and figures keyed by category.
    """
    # Clean previous outputs
    if os.path.exists("output_plots"):
        for f in glob.glob("output_plots/*.html"):
            os.remove(f)

    figures = {}

    # Step 1 & 2: Load data
    status_container.write("Step 1/7: Loading camera data...")
    camera_df = data_loader.load_and_prepare_camera_data(camera_path)

    status_container.write("Step 2/7: Loading water/tide data...")
    water_df = data_loader.load_and_prepare_water_data(water_path)

    # Step 3: Combine
    status_container.write("Step 3/7: Combining datasets...")
    combined_df = data_combiner.combine_data(camera_df, water_df)
    combined_df.to_csv("combined_data_output.csv", index=False)

    # Step 4: Comprehensive dual-framework analysis
    status_container.update(label="Running comprehensive analysis...")
    status_container.write("Step 4/7: Running dual-framework analysis...")
    comprehensive_results = comprehensive_analysis.run_comprehensive_analysis(
        combined_df
    )

    # Step 5: Specialized analyses
    status_container.write("Step 5/7: Running species & environmental analyses...")
    species_summary_df, species_df = species_analysis.analyze_species_diversity(
        combined_df
    )
    env_results = environmental_analysis.analyze_environmental_factors(combined_df)
    bird_tide_results = bird_tide_analysis.analyze_bird_tide_gate_behavior(combined_df)
    gate_combo_df = gate_combination_analysis.run_gate_combination_analysis(combined_df)

    # Step 6: Tide cycle
    status_container.write("Step 6/7: Running tide cycle analysis...")
    tide_cycle_df, detection_by_tide, phase_detection = (
        tide_cycle_analysis.analyze_tide_cycle_detections(gate_combo_df)
    )
    species_tide_table = tide_cycle_analysis.analyze_species_tide_preferences(
        tide_cycle_df
    )

    # Step 7: Visualizations
    status_container.update(label="Generating visualizations...")
    status_container.write("Step 7/7: Generating visualizations...")

    if species_summary_df is not None and not species_summary_df.empty:
        fig = visualization.plot_species_analysis(species_summary_df)
        if fig:
            figures["species_summary"] = fig

    if env_results:
        env_figs = visualization.plot_environmental_factors(*env_results)
        if env_figs:
            figures.update(env_figs)

    if bird_tide_results is not None and not bird_tide_results.empty:
        bird_figs = visualization.plot_bird_analysis(bird_tide_results, combined_df)
        if bird_figs:
            figures.update(bird_figs)

    water_figs = visualization.create_safe_water_visualizations(combined_df)
    if water_figs:
        figures.update(water_figs)

    visualization.create_hypothesis_visualizations(gate_combo_df)

    tide_figs = visualization.create_tide_cycle_visualizations(
        tide_cycle_df, (detection_by_tide, phase_detection, species_tide_table)
    )
    if tide_figs:
        figures.update(tide_figs)

    addl_figs = additional_visualizations.create_all_additional_visualizations(
        comprehensive_results, combined_df
    )
    if addl_figs:
        figures.update(addl_figs)

    return {
        "combined_df": combined_df,
        "comprehensive": comprehensive_results,
        "species_summary": species_summary_df,
        "species_df": species_df,
        "env_results": env_results,
        "bird_tide_results": bird_tide_results,
        "gate_combo_df": gate_combo_df,
        "tide_cycle_df": tide_cycle_df,
        "detection_by_tide": detection_by_tide,
        "phase_detection": phase_detection,
        "species_tide_table": species_tide_table,
        "figures": figures,
    }


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("Tidegate Analysis")
    st.markdown(
        "Analyze wildlife camera trap data combined with tidal and "
        "environmental sensor data using a dual-framework approach."
    )

    st.divider()

    with st.expander("Instructions"):
        st.markdown(
            """
1. Upload a **Camera CSV** (must contain `DateTime` and `Species 1` columns).
2. Upload a **Water/Tide CSV** (must contain `DateTime` and sensor columns).
3. Preview data to confirm correct format.
4. Click **Run Full Analysis**.
5. Explore results across the tabs.
            """
        )

    st.divider()

    # Analysis status
    if st.session_state.get("analysis_complete"):
        st.success("Analysis complete")
        comp = st.session_state.comprehensive["comparison"]
        st.metric("Total Time Periods", f"{comp['total_periods']:,}")
        st.metric("Camera Active Periods", f"{comp['camera_periods']:,}")
        st.metric("Animal Detections", f"{comp['animal_detections']:,}")

        st.divider()

        # Sidebar downloads
        st.download_button(
            "Download Combined Dataset",
            data=st.session_state.combined_df.to_csv(index=False),
            file_name="combined_data_output.csv",
            mime="text/csv",
            key="sb_dl_combined",
        )
        st.download_button(
            "Download Analysis Log",
            data=st.session_state.console_log,
            file_name="analysis_log.txt",
            mime="text/plain",
            key="sb_dl_log",
        )
        if (
            st.session_state.species_summary is not None
            and not st.session_state.species_summary.empty
        ):
            st.download_button(
                "Download Species Summary",
                data=st.session_state.species_summary.to_csv(),
                file_name="species_summary.csv",
                mime="text/csv",
                key="sb_dl_species",
            )
    else:
        st.info("No analysis run yet")


# --------------------------------------------------
# Main Area: Section 1 — Data Upload & Preview
# --------------------------------------------------
st.header("Data Upload")

col_cam, col_wat = st.columns(2)

camera_file = col_cam.file_uploader(
    "Camera CSV", type=["csv"], help="Camera detections and activity data"
)
water_file = col_wat.file_uploader(
    "Water / Tide CSV", type=["csv"], help="Tide, gate, or environmental sensor data"
)

# Preview & validate camera file
if camera_file is not None:
    try:
        camera_preview = pd.read_csv(camera_file, nrows=10, low_memory=False)
        camera_file.seek(0)
        is_valid, msgs = validate_camera_columns(camera_preview)
        with col_cam:
            if is_valid:
                st.success(
                    f"Loaded preview: {len(camera_preview.columns)} columns"
                )
            else:
                st.error("Validation failed")
            for m in msgs:
                st.caption(m)
            with st.expander("Preview rows"):
                st.dataframe(camera_preview, use_container_width=True)
    except Exception as e:
        col_cam.error(f"Could not read file: {e}")

# Preview & validate water file
if water_file is not None:
    try:
        water_preview = pd.read_csv(water_file, nrows=10, low_memory=False)
        water_file.seek(0)
        is_valid, msgs = validate_water_columns(water_preview)
        with col_wat:
            if is_valid:
                st.success(
                    f"Loaded preview: {len(water_preview.columns)} columns"
                )
            else:
                st.error("Validation failed")
            for m in msgs:
                st.caption(m)
            with st.expander("Preview rows"):
                st.dataframe(water_preview, use_container_width=True)
    except Exception as e:
        col_wat.error(f"Could not read file: {e}")


# --------------------------------------------------
# Main Area: Section 2 — Run Analysis
# --------------------------------------------------
st.divider()

run_btn = st.button(
    "Run Full Analysis",
    type="primary",
    disabled=(camera_file is None or water_file is None),
)

if run_btn:
    if not camera_file or not water_file:
        st.error("Please upload both CSV files.")
    else:
        log_buffer = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp:
            camera_path = os.path.join(tmp, camera_file.name)
            water_path = os.path.join(tmp, water_file.name)

            with open(camera_path, "wb") as f:
                f.write(camera_file.read())
            with open(water_path, "wb") as f:
                f.write(water_file.read())

            with st.status(
                "Running analysis pipeline...", expanded=True
            ) as status:
                with contextlib.redirect_stdout(log_buffer):
                    results = run_pipeline(camera_path, water_path, status)
                status.update(
                    label="Analysis complete!", state="complete", expanded=False
                )

        # Store in session state
        for key in [
            "combined_df",
            "comprehensive",
            "species_summary",
            "species_df",
            "env_results",
            "bird_tide_results",
            "gate_combo_df",
            "tide_cycle_df",
            "detection_by_tide",
            "phase_detection",
            "species_tide_table",
            "figures",
        ]:
            st.session_state[key] = results[key]

        st.session_state.console_log = log_buffer.getvalue()
        st.session_state.analysis_complete = True
        st.rerun()


# --------------------------------------------------
# Main Area: Section 3 — Results (only after analysis)
# --------------------------------------------------
if st.session_state.get("analysis_complete"):
    comparison = st.session_state.comprehensive["comparison"]
    figures = st.session_state.figures

    # ---- KPI Dashboard ----
    st.header("Results Dashboard")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Time Periods", f"{comparison['total_periods']:,}")
    m2.metric("Camera Active Periods", f"{comparison['camera_periods']:,}")
    m3.metric("Animal Detections", f"{comparison['animal_detections']:,}")

    species_count = "N/A"
    if (
        st.session_state.species_summary is not None
        and not st.session_state.species_summary.empty
    ):
        species_count = str(len(st.session_state.species_summary))
    m4.metric("Unique Species", species_count)

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Camera Activity Rate", f"{comparison['camera_activity_rate']:.2f}%")
    m6.metric(
        "Overall Detection Rate", f"{comparison['overall_detection_rate']:.3f}%"
    )
    m7.metric(
        "Camera Detection Rate", f"{comparison['camera_detection_rate']:.1f}%"
    )

    bird_results = st.session_state.bird_tide_results
    if bird_results is not None and not bird_results.empty:
        peak_rate = bird_results.values.max()
        m8.metric("Peak Detection Rate", f"{peak_rate:.2f}%")
    else:
        m8.metric("Peak Detection Rate", "N/A")

    # ---- Key Findings ----
    st.subheader("Key Findings")
    kf1, kf2 = st.columns(2)
    with kf1:
        st.info(
            f"**Camera Activity Pattern Analysis**\n\n"
            f"Cameras operated during **{comparison['camera_activity_rate']:.2f}%** "
            f"of all monitoring periods. This method reveals equipment "
            f"performance and operational bias patterns."
        )
    with kf2:
        st.success(
            f"**Wildlife Detection Efficiency**\n\n"
            f"When cameras were active, animals were detected "
            f"**{comparison['camera_detection_rate']:.1f}%** of the time. "
            f"This method reveals wildlife behavior and optimal "
            f"monitoring conditions."
        )

    # Peak detection highlight
    if bird_results is not None and not bird_results.empty:
        peak_val = bird_results.values.max()
        peak_pos = np.where(bird_results.values == peak_val)
        if peak_pos[0].size > 0 and peak_pos[1].size > 0:
            peak_gate = bird_results.index[peak_pos[0][0]]
            peak_tide = bird_results.columns[peak_pos[1][0]]
            st.success(
                f"**Peak detection rate: {peak_val:.2f}%** "
                f"when gate is **{peak_gate}** during **{peak_tide}** tide."
            )

    st.divider()

    # ---- Tabbed Results ----
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Species Analysis",
            "Environmental Factors",
            "Gate Interactions",
            "Tidal Cycles",
            "Method Comparison",
            "Combined Dataset",
            "Console Log",
        ]
    )

    # ======= Tab 1: Species =======
    with tab1:
        st.subheader("Species Diversity Summary")
        ssum = st.session_state.species_summary

        if ssum is not None and not ssum.empty:
            top3 = ssum.head(3)
            cols = st.columns(min(3, len(top3)))
            for i, (name, row) in enumerate(top3.iterrows()):
                cols[i].metric(
                    name,
                    f"{int(row['Total_Count'])} individuals",
                    f"{int(row['Detection_Events'])} events",
                )

            st.dataframe(
                ssum.reset_index().rename(columns={"index": "Species"}),
                use_container_width=True,
            )

            if "species_summary" in figures:
                st.plotly_chart(figures["species_summary"], use_container_width=True)
            if "species_pattern_comparison" in figures:
                st.plotly_chart(
                    figures["species_pattern_comparison"], use_container_width=True
                )

            st.download_button(
                "Download Species Summary CSV",
                data=ssum.to_csv(),
                file_name="species_summary.csv",
                mime="text/csv",
                key="tab_dl_species",
            )
        else:
            st.warning("No species data available.")

    # ======= Tab 2: Environmental =======
    with tab2:
        st.subheader("Environmental Factor Analysis")
        env = st.session_state.env_results

        if env:
            mtr_df, hinge_df, tidal_df, temp_df = env

            if mtr_df is not None and not mtr_df.empty:
                st.markdown("#### MTR Gate Detection Rates")
                st.dataframe(mtr_df.reset_index(), use_container_width=True)
                if "mtr_gate_detection_rate" in figures:
                    st.plotly_chart(
                        figures["mtr_gate_detection_rate"], use_container_width=True
                    )

            if hinge_df is not None and not hinge_df.empty:
                st.markdown("#### Top Hinge Gate Detection Rates")
                st.dataframe(hinge_df.reset_index(), use_container_width=True)
                if "hinge_gate_detection_rate" in figures:
                    st.plotly_chart(
                        figures["hinge_gate_detection_rate"],
                        use_container_width=True,
                    )

            if tidal_df is not None and not tidal_df.empty:
                st.markdown("#### Tidal Level Detection Rates")
                st.dataframe(tidal_df.reset_index(), use_container_width=True)
                if "tidal_level_detection_rate" in figures:
                    st.plotly_chart(
                        figures["tidal_level_detection_rate"],
                        use_container_width=True,
                    )

            # Water quality time series
            water_keys = sorted(k for k in figures if k.startswith("water_quality_"))
            if water_keys:
                st.markdown("#### Water Quality Time Series")
                for key in water_keys:
                    st.plotly_chart(figures[key], use_container_width=True)

            if "environmental_effectiveness" in figures:
                st.markdown("#### Environmental Effectiveness Dashboard")
                st.plotly_chart(
                    figures["environmental_effectiveness"],
                    use_container_width=True,
                )

            # Downloads
            env_items = [
                ("MTR Gate", mtr_df),
                ("Hinge Gate", hinge_df),
                ("Tidal Level", tidal_df),
                ("Temperature", temp_df),
            ]
            for name, df in env_items:
                if df is not None and not df.empty:
                    st.download_button(
                        f"Download {name} Analysis CSV",
                        data=df.to_csv(),
                        file_name=f"{name.lower().replace(' ', '_')}_analysis.csv",
                        mime="text/csv",
                        key=f"dl_env_{name}",
                    )
        else:
            st.warning("No environmental analysis results available.")

    # ======= Tab 3: Gate Interactions =======
    with tab3:
        st.subheader("Wildlife & Tide Gate Behavior")
        bt = st.session_state.bird_tide_results

        if bt is not None and not bt.empty:
            st.markdown("#### Detection Rate (%) by Gate Status and Tidal Flow")
            st.dataframe(bt.round(2), use_container_width=True)

            peak_val = bt.values.max()
            peak_pos = np.where(bt.values == peak_val)
            if peak_pos[0].size > 0 and peak_pos[1].size > 0:
                st.success(
                    f"Peak detection rate: **{peak_val:.2f}%** at gate "
                    f"'{bt.index[peak_pos[0][0]]}' during "
                    f"'{bt.columns[peak_pos[1][0]]}' tide."
                )

            if "wildlife_detection_heatmap" in figures:
                st.plotly_chart(
                    figures["wildlife_detection_heatmap"],
                    use_container_width=True,
                )
            if "wildlife_detection_scatter" in figures:
                st.plotly_chart(
                    figures["wildlife_detection_scatter"],
                    use_container_width=True,
                )

            # Hypothesis PNG visualizations
            st.markdown("#### Hypothesis Test Visualizations")
            hyp_files = sorted(glob.glob("hypothesis_visualization_*.png"))
            if hyp_files:
                hyp_cols = st.columns(2)
                for i, fpath in enumerate(hyp_files):
                    hyp_cols[i % 2].image(fpath, use_container_width=True)
            else:
                st.caption("No hypothesis visualization PNGs found.")

            st.download_button(
                "Download Gate Interaction Summary CSV",
                data=bt.to_csv(),
                file_name="gate_interaction_summary.csv",
                mime="text/csv",
                key="dl_gate",
            )
        else:
            st.warning("No gate interaction results available.")

    # ======= Tab 4: Tidal Cycles =======
    with tab4:
        st.subheader("Tidal Cycle Detection Analysis")
        det_tide = st.session_state.detection_by_tide
        phase_det = st.session_state.phase_detection
        sp_tide = st.session_state.species_tide_table

        if det_tide is not None and not det_tide.empty:
            st.markdown("#### Detection Rates by Tidal State")
            st.dataframe(det_tide.round(4), use_container_width=True)
            if "detection_by_tidal_state" in figures:
                st.plotly_chart(
                    figures["detection_by_tidal_state"], use_container_width=True
                )

        if phase_det is not None and not phase_det.empty:
            st.markdown("#### Detection Rates by Tidal Phase")
            st.dataframe(phase_det.round(4), use_container_width=True)

            if "detection_rate" in phase_det.columns and phase_det["detection_rate"].sum() > 0:
                peak_phase = phase_det["detection_rate"].idxmax()
                peak_rate = phase_det["detection_rate"].max()
                st.info(
                    f"Peak detection rate: **{peak_rate:.1%}** at tidal phase "
                    f"**{peak_phase}**"
                )

            if "detection_by_tidal_phase" in figures:
                st.plotly_chart(
                    figures["detection_by_tidal_phase"], use_container_width=True
                )

        if sp_tide is not None and not sp_tide.empty:
            st.markdown("#### Species Tidal Preferences")
            st.dataframe(sp_tide.round(1), use_container_width=True)
            if "species_tide_preference_heatmap" in figures:
                st.plotly_chart(
                    figures["species_tide_preference_heatmap"],
                    use_container_width=True,
                )
            st.download_button(
                "Download Species Tide Preferences CSV",
                data=sp_tide.to_csv(),
                file_name="species_tide_preferences.csv",
                mime="text/csv",
                key="dl_tide_prefs",
            )

    # ======= Tab 5: Method Comparison =======
    with tab5:
        st.subheader("Analysis Method Comparison")
        comp_results = st.session_state.comprehensive

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("#### Camera Activity Pattern Analysis")
            st.markdown(
                "- Uses **all** time periods (including sensor-only data)\n"
                "- Measures: Camera Active / All Time Periods\n"
                "- Reveals: Equipment performance and monitoring bias"
            )
            act_summary = comp_results["camera_activity"]["species_summary"]
            if act_summary is not None and not act_summary.empty:
                st.dataframe(
                    act_summary.head(10).reset_index(),
                    use_container_width=True,
                )

        with mc2:
            st.markdown("#### Wildlife Detection Efficiency Analysis")
            st.markdown(
                "- Uses **only** camera observation periods\n"
                "- Measures: Animal Detections / Camera Observations\n"
                "- Reveals: Wildlife behavior and optimal conditions"
            )
            eff_summary = comp_results["detection_efficiency"]["species_summary"]
            if eff_summary is not None and not eff_summary.empty:
                st.dataframe(
                    eff_summary.head(10).reset_index(),
                    use_container_width=True,
                )

        # Additional comparison charts
        chart_keys = [
            ("data_overview_dashboard", "Data Overview Dashboard"),
            ("analysis_method_comparison", "Analysis Method Comparison"),
            ("temporal_analysis", "Temporal Activity Patterns"),
            ("camera_performance_dashboard", "Camera Performance Dashboard"),
        ]
        for key, title in chart_keys:
            if key in figures:
                st.markdown(f"#### {title}")
                st.plotly_chart(figures[key], use_container_width=True)

    # ======= Tab 6: Combined Dataset =======
    with tab6:
        st.subheader("Combined Dataset Preview")
        combined = st.session_state.combined_df

        st.markdown(f"**{len(combined):,} rows x {len(combined.columns)} columns**")

        all_cols = combined.columns.tolist()
        default_show = [
            c
            for c in [
                "DateTime",
                "Species",
                "Count",
                "has_camera_data",
                "animal_detected",
                "Gate_Opening_MTR_Deg",
                "Depth",
            ]
            if c in all_cols
        ]

        selected = st.multiselect(
            "Select columns to display", all_cols, default=default_show
        )

        if selected:
            st.dataframe(
                combined[selected].head(500),
                use_container_width=True,
                height=400,
            )

        st.download_button(
            "Download Combined Dataset CSV",
            data=combined.to_csv(index=False),
            file_name="combined_data_output.csv",
            mime="text/csv",
            key="tab_dl_combined",
        )

    # ======= Tab 7: Console Log =======
    with tab7:
        st.subheader("Analysis Console Log")
        st.caption("Full output from all analysis modules")

        log_text = st.session_state.console_log
        line_count = log_text.count("\n")
        st.info(f"Console output: {line_count} lines")

        search = st.text_input("Search log", placeholder="Filter log output...")

        if search:
            filtered = [
                line
                for line in log_text.split("\n")
                if search.lower() in line.lower()
            ]
            st.code("\n".join(filtered), language="text")
        else:
            st.code(log_text, language="text")

        st.download_button(
            "Download Analysis Log",
            data=log_text,
            file_name="analysis_log.txt",
            mime="text/plain",
            key="tab_dl_log",
        )
