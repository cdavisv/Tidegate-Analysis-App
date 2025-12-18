
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


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Comprehensive Wildlife & Tide Analysis",
    layout="wide"
)

st.title("📊 Comprehensive Wildlife & Environmental Analysis")
st.markdown(
    """
    This application runs the **full camera activity, wildlife detection,
    and environmental analysis pipeline** using camera trap and water/tide data.

    All printed output from the analysis pipeline is captured and displayed below.
    """
)

# -----------------------------
# File Upload Section
# -----------------------------
st.header("1️⃣ Upload Input Data")

camera_file = st.file_uploader(
    "Upload Camera CSV",
    type=["csv"],
    help="Camera detections and activity data"
)

water_file = st.file_uploader(
    "Upload Water / Tide CSV",
    type=["csv"],
    help="Tide, gate, or environmental sensor data"
)

run_analysis = st.button("🚀 Run Full Analysis", type="primary")


# -----------------------------
# Helpers for Plot Rendering
# -----------------------------
def render_plot_html(filepath, height=700):
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=height, scrolling=True)


def render_plot_section(title, patterns):
    st.subheader(title)
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))

    if not files:
        st.info("No plots generated for this section.")
        return

    for f in files:
        name = os.path.basename(f).replace("_", " ").replace(".html", "")
        with st.expander(name):
            render_plot_html(f)


# -----------------------------
# Analysis Runner (unchanged)
# -----------------------------
def run_pipeline(camera_path, water_path):
    if os.path.exists("output_plots"):
        for f in glob.glob("output_plots/*.html"):
            os.remove(f)

    camera_df = data_loader.load_and_prepare_camera_data(camera_path)
    water_df = data_loader.load_and_prepare_water_data(water_path)

    combined_df = data_combiner.combine_data(camera_df, water_df)
    combined_df.to_csv("combined_data_output.csv", index=False)

    comprehensive_results = comprehensive_analysis.run_comprehensive_analysis(combined_df)

    species_summary_df, _ = species_analysis.analyze_species_diversity(combined_df)
    env_results = environmental_analysis.analyze_environmental_factors(combined_df)
    bird_tide_results = bird_tide_analysis.analyze_bird_tide_gate_behavior(combined_df)
    gate_combo_df = gate_combination_analysis.run_gate_combination_analysis(combined_df)

    tide_cycle_df, detection_by_tide, phase_detection = (
        tide_cycle_analysis.analyze_tide_cycle_detections(gate_combo_df)
    )
    species_tide_table = tide_cycle_analysis.analyze_species_tide_preferences(
        tide_cycle_df
    )

    if species_summary_df is not None and not species_summary_df.empty:
        visualization.plot_species_analysis(species_summary_df)

    if env_results:
        visualization.plot_environmental_factors(*env_results)

    if bird_tide_results is not None and not bird_tide_results.empty:
        visualization.plot_bird_analysis(bird_tide_results, combined_df)

    visualization.create_safe_water_visualizations(combined_df)
    visualization.create_hypothesis_visualizations(gate_combo_df)

    visualization.create_tide_cycle_visualizations(
        tide_cycle_df, (detection_by_tide, phase_detection, species_tide_table)
    )

    additional_visualizations.create_all_additional_visualizations(
        comprehensive_results, combined_df
    )

    return combined_df, species_summary_df, comprehensive_results


# -----------------------------
# Run Button Logic
# -----------------------------
if run_analysis:
    if not camera_file or not water_file:
        st.error("Please upload both Camera and Water CSV files.")
    else:
        log_buffer = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp:
            camera_path = os.path.join(tmp, camera_file.name)
            water_path = os.path.join(tmp, water_file.name)

            with open(camera_path, "wb") as f:
                f.write(camera_file.read())
            with open(water_path, "wb") as f:
                f.write(water_file.read())

            with st.spinner("Running full analysis pipeline..."):
                with contextlib.redirect_stdout(log_buffer):
                    combined_df, species_summary, comprehensive = run_pipeline(
                        camera_path, water_path
                    )

        st.success("🎉 Analysis finished successfully!")

        # -----------------------------
        # Captured Console Output
        # -----------------------------
        st.header("2️⃣ Analysis Console Output")

        with st.expander("Show full pipeline output", expanded=True):
            st.code(log_buffer.getvalue(), language="text")

        st.download_button(
            "Download Analysis Log",
            data=log_buffer.getvalue(),
            file_name="analysis_log.txt",
            mime="text/plain"
        )

        # -----------------------------
        # Visualizations
        # -----------------------------
        st.header("3️⃣ Visualizations")

        render_plot_section(
            "Species & Detection Overview",
            ["output_plots/1_*.html", "output_plots/7c_*.html"]
        )

        render_plot_section(
            "Environmental & Gate Effects",
            ["output_plots/2*.html", "output_plots/7d_*.html"]
        )

        render_plot_section(
            "Wildlife Behavior & Gate Interactions",
            ["output_plots/5*.html"]
        )

        render_plot_section(
            "Tidal Cycle & Phase Analysis",
            ["output_plots/6*.html"]
        )

        render_plot_section(
            "Method Comparisons & Performance Dashboards",
            ["output_plots/7*.html"]
        )

        # -----------------------------
        # Downloads
        # -----------------------------
        st.header("4️⃣ Downloads")

        st.download_button(
            "Download Combined Dataset",
            data=combined_df.to_csv(index=False),
            file_name="combined_data_output.csv",
            mime="text/csv"
        )
