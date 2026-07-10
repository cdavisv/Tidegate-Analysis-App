"""Reusable analysis pipeline runner (decoupled from the Streamlit UI).

Extracted from the original ``main.py`` so it can be driven with in-memory
DataFrames (e.g. a camera table produced by the vision detector, or a
weather-merged sensor table) as well as file paths / uploaded buffers.

``run_full_analysis`` returns the same result dictionary the UI expects, with
keys: ``combined_df``, ``comprehensive``, ``species_summary``, ``species_df``,
``env_results``, ``bird_tide_results``, ``gate_combo_df``, ``tide_cycle_df``,
``detection_by_tide``, ``phase_detection``, ``species_tide_table``, ``figures``.
"""

from __future__ import annotations

import glob
import os
import tempfile
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd

import data_loader
import data_combiner
import comprehensive_analysis
import species_analysis
import environmental_analysis
import bird_tide_analysis
import gate_combination_analysis
import tide_cycle_analysis
import visualization
import additional_visualizations

Progress = Optional[Callable[[str], None]]
InputType = Union[str, "os.PathLike", pd.DataFrame, Any]


def _to_path(data: InputType, tmpdir: str, name: str) -> str:
    """Coerce an input (path / uploaded buffer / DataFrame) to a CSV path."""
    if isinstance(data, pd.DataFrame):
        path = os.path.join(tmpdir, name)
        data.to_csv(path, index=False)
        return path
    if hasattr(data, "read"):  # Streamlit UploadedFile / file-like
        path = os.path.join(tmpdir, getattr(data, "name", name))
        try:
            data.seek(0)
        except Exception:
            pass
        with open(path, "wb") as fh:
            fh.write(data.read())
        return path
    return str(data)


def _emit(progress: Progress, msg: str) -> None:
    if progress is not None:
        try:
            progress(msg)
        except Exception:
            pass


def run_full_analysis(
    camera_input: InputType,
    water_input: InputType,
    progress: Progress = None,
    output_dir: str = "output_plots",
    combined_csv_path: str = "combined_data_output.csv",
) -> Dict[str, Any]:
    """Run the full dual-framework analysis pipeline.

    Args:
        camera_input: Camera data as a file path, uploaded buffer, or wide-format
            DataFrame (e.g. from ``vision.detections_to_camera_df``).
        water_input: Water/tide/sensor data as a path, buffer, or DataFrame.
        progress: Optional callback receiving human-readable step messages.
        output_dir: Directory for generated HTML plots.
        combined_csv_path: Where to write the merged dataset CSV.

    Returns:
        The results dictionary consumed by the analysis UI.
    """
    os.makedirs(output_dir, exist_ok=True)
    for f in glob.glob(os.path.join(output_dir, "*.html")):
        try:
            os.remove(f)
        except OSError:
            pass

    figures: Dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as tmp:
        camera_path = _to_path(camera_input, tmp, "camera.csv")
        water_path = _to_path(water_input, tmp, "water.csv")

        _emit(progress, "Step 1/7: Loading camera data...")
        camera_df = data_loader.load_and_prepare_camera_data(camera_path)
        if camera_df is None:
            raise ValueError(
                "Camera data could not be loaded. Ensure it has a 'DateTime' "
                "(or 'Date'+'Time') column and 'Species 1' column."
            )

        _emit(progress, "Step 2/7: Loading water / sensor data...")
        water_df = data_loader.load_and_prepare_water_data(water_path)
        if water_df is None:
            raise ValueError(
                "Water/sensor data could not be loaded. Ensure it has a "
                "'DateTime' (or 'Date'+'Time') column."
            )

    _emit(progress, "Step 3/7: Combining datasets...")
    combined_df = data_combiner.combine_data(camera_df, water_df)
    try:
        combined_df.to_csv(combined_csv_path, index=False)
    except OSError:
        pass

    _emit(progress, "Step 4/7: Running dual-framework analysis...")
    comprehensive_results = comprehensive_analysis.run_comprehensive_analysis(combined_df)

    _emit(progress, "Step 5/7: Running species & environmental analyses...")
    species_summary_df, species_df = species_analysis.analyze_species_diversity(combined_df)
    env_results = environmental_analysis.analyze_environmental_factors(combined_df)
    bird_tide_results = bird_tide_analysis.analyze_bird_tide_gate_behavior(combined_df)
    gate_combo_df = gate_combination_analysis.run_gate_combination_analysis(combined_df)

    _emit(progress, "Step 6/7: Running tide cycle analysis...")
    tide_cycle_df, detection_by_tide, phase_detection = (
        tide_cycle_analysis.analyze_tide_cycle_detections(gate_combo_df)
    )
    species_tide_table = tide_cycle_analysis.analyze_species_tide_preferences(tide_cycle_df)

    _emit(progress, "Step 7/7: Generating visualizations...")
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

    try:
        visualization.create_hypothesis_visualizations(gate_combo_df)
    except Exception as exc:  # non-fatal; PNGs are a nice-to-have
        _emit(progress, f"(hypothesis visualizations skipped: {exc})")

    tide_figs = visualization.create_tide_cycle_visualizations(
        tide_cycle_df, (detection_by_tide, phase_detection, species_tide_table)
    )
    if tide_figs:
        figures.update(tide_figs)

    try:
        addl_figs = additional_visualizations.create_all_additional_visualizations(
            comprehensive_results, combined_df
        )
        if addl_figs:
            figures.update(addl_figs)
    except Exception as exc:  # non-fatal; the extra dashboards are a nice-to-have
        _emit(progress, f"(additional visualizations skipped: {exc})")

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
