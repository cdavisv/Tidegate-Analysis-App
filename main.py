# main.py

import pandas as pd
import warnings
import data_loader
import data_combiner
import visualization
import analysis

def main():
    """
    Main function to run the complete wildlife and water quality analysis pipeline.
    """
    # --- Setup ---
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns', 50)

    # --- 1. Data Loading ---
    # Use input() for interactive file paths or hardcode them
    # camera_data_filepath = input("Enter path to camera data CSV: ")
    # water_data_filepath = input("Enter path to water data CSV: ")
    camera_data_filepath = 'willanch_camera_final.csv'
    water_data_filepath = 'willanch_sensor_final.csv'

    print("--- Loading and Preparing Data ---")
    camera_df = data_loader.load_and_prepare_camera_data(camera_data_filepath)
    water_df = data_loader.load_and_prepare_water_data(water_data_filepath, file_id="Willanch")

    if camera_df.empty or water_df.empty:
        print("One or both data files failed to load. Exiting.")
        return

    # --- 2. Data Combination ---
    combined_df = data_combiner.combine_data(camera_df, water_df)
    
    # Create a DataFrame of just species detections for easier analysis
    species_df = combined_df[combined_df['has_camera_data']].copy()

    # --- 3. Exploratory Analysis & Visualization ---
    print("\n--- Running Initial Visualizations ---")
    visualization.create_safe_water_visualizations(combined_df, title_suffix="(Combined & Interpolated)")
    visualization.create_analysis_plots(combined_df, species_df)

    # --- 4. Core Analyses ---
    print("\n--- Running Core Analyses ---")
    gate_summary, gate_test_results = analysis.analyze_gate_impact(combined_df)
    if gate_summary is not None:
        visualization.plot_gate_analysis(gate_summary)

    hourly_stats, monthly_stats = analysis.analyze_temporal_patterns(combined_df)
    
    glm_model = analysis.run_glm_analysis(combined_df)

    # --- 5. Export Results ---
    print("\n--- Exporting Final Datasets ---")
    output_filename = 'combined_analysis_results.csv'
    combined_df.to_csv(output_filename, index=False)
    print(f"✅ Combined and processed data saved to: {output_filename}")

    if not species_df.empty:
        species_summary = species_df.groupby('Species')['Count'].sum().sort_values(ascending=False).reset_index()
        species_summary.to_csv('species_detection_summary.csv', index=False)
        print("✅ Species summary saved to: species_detection_summary.csv")

    print("\n🎉 Analysis Complete! 🎉")

if __name__ == '__main__':
    main()