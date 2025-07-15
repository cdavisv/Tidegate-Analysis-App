# main.py

import pandas as pd
import data_loader
import data_combiner
import species_analysis
import environmental_analysis
import bird_tide_analysis
import visualization

# --- CONFIGURATION ---
# Use this variable to control the maximum time gap (in hours) to fill with interpolation.
MAX_INTERPOLATION_HOURS = .25  # <-- CHANGE THIS VALUE

# Define the file paths for your data files.
# TODO: Update these paths to point to your actual data files.
CAMERA_DATA_PATH = 'willanch_camera_final.csv'
WATER_DATA_PATH = 'willanch_sensor_final.csv'


def main():
    """
    Main function to orchestrate the entire data analysis pipeline,
    from data loading and combination to analysis and visualization.
    """
    print("--- Starting Analysis Pipeline ---")

    # 1. Load and Prepare Data
    # These functions load the raw camera and water data from CSVs and
    # standardize the formats and column names.
    camera_df = data_loader.load_and_prepare_camera_data(CAMERA_DATA_PATH)
    water_df = data_loader.load_and_prepare_water_data(WATER_DATA_PATH)

    # 2. Combine and Interpolate Data
    # This function merges the two datasets by their timestamps and fills
    # missing water data points using a time-aware interpolation method,
    # limited by the MAX_INTERPOLATION_HOURS variable.
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
    # Each analysis module processes the combined data to extract insights.
    species_summary, species_df = species_analysis.analyze_species_diversity(combined_df)
    mtr_gate_analysis, hinge_gate_analysis, tidal_analysis, temp_analysis = environmental_analysis.analyze_environmental_factors(combined_df)
    bird_summary_table = bird_tide_analysis.analyze_bird_tide_gate_behavior(combined_df)

    
    # 4. Generate Visualizations
    # The visualization module takes the results of the analyses
    # and generates plots and graphs.
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

    gate_analysis_df = pd.DataFrame({
    'Detection_Rate_Pct': [10, 25, 60]
    }, index=['Closed', 'Partially Open', 'Fully Open'])


    # --- ADD THIS LINE TO EXECUTE THE PLOT ---
    # This tells your program to actually run the function from your visualization file
    visualization.plot_gate_analysis(gate_analysis_df)


    print("\n--- Analysis Pipeline Complete ---")


if __name__ == '__main__':
    main()