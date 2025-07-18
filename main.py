# main.py

import pandas as pd
import data_loader
import data_combiner
import species_analysis
import environmental_analysis
import bird_tide_analysis
import gate_combination_analysis
import tide_cycle_analysis # <-- IMPORT THE NEW MODULE
import visualization

# --- CONFIGURATION ---
# Use this variable to control the maximum time gap (in hours) to fill with interpolation.
MAX_INTERPOLATION_HOURS = .25  # <-- CHANGE THIS VALUE

# Define the file paths for your data files.
# TODO: Update these paths to point to your actual data files.
CAMERA_DATA_PATH = 'Palouse Combined Camera Imageset_final.csv'
WATER_DATA_PATH = 'Palouse_Tide_Data_Combined_newest.csv'


def main():
    """
    Main function to orchestrate the entire data analysis pipeline,
    from data loading and combination to analysis and visualization.
    """
    print("--- Starting Analysis Pipeline ---")

    # 1. Load and Prepare Data
    camera_df = data_loader.load_and_prepare_camera_data(CAMERA_DATA_PATH)
    water_df = data_loader.load_and_prepare_water_data(WATER_DATA_PATH)

    # 2. Combine and Interpolate Data
    combined_df = data_combiner.combine_data(
        camera_df,
        water_df,
        max_interp_hours=MAX_INTERPOLATION_HOURS
    )

    if combined_df.empty:
        print("\nCombined DataFrame is empty. Cannot proceed with analysis. Exiting.")
        return
    
    combined_df.to_csv('combined_data_output.csv', index=False)
    print("\n -> Successfully saved the combined DataFrame to 'combined_data_output.csv'")

    # 3. Perform Analyses
    species_summary, species_df = species_analysis.analyze_species_diversity(combined_df)
    mtr_gate_analysis, hinge_gate_analysis, tidal_analysis, temp_analysis = environmental_analysis.analyze_environmental_factors(combined_df)
    bird_summary_table = bird_tide_analysis.analyze_bird_tide_gate_behavior(combined_df)
    gate_combination_analysis.run_gate_combination_analysis(combined_df)

    # --- NEW ANALYSIS STEP ---
    # Run the tide cycle analysis, which returns multiple results
    tide_cycle_df, detection_by_tide, phase_detection = tide_cycle_analysis.analyze_tide_cycle_detections(combined_df)
    species_tide_table = tide_cycle_analysis.analyze_species_tide_preferences(tide_cycle_df)
    # --- END NEW ANALYSIS ---

    # 4. Generate Visualizations
    print("\n\n--- Generating All Visualizations ---")
    visualization.plot_species_analysis(species_summary)
    visualization.plot_environmental_factors(
        mtr_gate_analysis,
        hinge_gate_analysis,
        tidal_analysis,
        temp_analysis
    )
    
    if bird_summary_table is not None:
        visualization.plot_bird_analysis(bird_summary_table, combined_df)
    else:
        print("\nSkipping bird behavior plots: No summary data was generated.")

    visualization.create_safe_water_visualizations(combined_df)
    visualization.create_hypothesis_visualizations(combined_df)

    # --- NEW VISUALIZATION STEP ---
    # Bundle the tide analysis results and pass them to the new visualization function
    tide_viz_results = (detection_by_tide, phase_detection, species_tide_table)
    visualization.create_tide_cycle_visualizations(tide_cycle_df, tide_viz_results)
    # --- END NEW VISUALIZATION ---

    print("\n--- Analysis Pipeline Complete ---")


if __name__ == '__main__':
    main()