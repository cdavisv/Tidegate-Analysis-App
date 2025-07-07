# main.py

import pandas as pd
import warnings
import data_loader
import data_combiner
import visualization
import analysis
import species_analysis
import environmental_analysis

def main():
    """
    Main function to run the complete wildlife and water quality analysis pipeline.
    """
    # --- Setup ---
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 120)

    # --- 1. Data Loading ---
    camera_data_filepath = 'willanch_camera_final.csv'
    water_data_filepath = 'willanch_sensor_final.csv'

    print("--- Loading and Preparing Data ---")
    camera_df = data_loader.load_and_prepare_camera_data(camera_data_filepath)
    water_df = data_loader.load_and_prepare_water_data(water_data_filepath, file_id="Willanch")

    if camera_df.empty or water_df.empty:
        print("One or both data files failed to load. Exiting.")
        return

    # --- 2. Initial Species Diversity Analysis ---
    species_summary, species_df_raw = species_analysis.analyze_species_diversity(camera_df)

    # --- 3. Data Combination ---
    combined_df = data_combiner.combine_data(camera_df, water_df)
    
    # --- 4. Environmental Analysis ---
    # Capture the four returned dataframes
    mtr_gate_df, hinge_gate_df, tidal_df, temp_df = environmental_analysis.analyze_environmental_factors(combined_df)

    # --- 5. Species-Specific Environmental Preferences ---
    species_df_combined = combined_df[combined_df['has_camera_data']].copy()
    species_analysis.analyze_species_preferences(species_df_combined)

    # --- 6. Visualization ---
    print("\n--- Generating All Visualizations ---")
    
    # Call the plotting function with the four captured dataframes
    visualization.plot_environmental_factors(mtr_gate_df, hinge_gate_df, tidal_df, temp_df)
    
    # Restore calls to other general visualization functions
    visualization.create_safe_water_visualizations(combined_df, title_suffix="(Combined & Interpolated)")
    visualization.create_analysis_plots(combined_df, species_df_raw)

    # --- 7. Modeling ---
    glm_model = analysis.run_glm_analysis(combined_df)

    # --- 8. Export Results ---
    print("\n--- Exporting Final Datasets ---")
    output_filename = 'combined_analysis_results.csv'
    combined_df.to_csv(output_filename, index=False)
    print(f"✅ Combined and processed data saved to: {output_filename}")

    if not species_summary.empty:
        species_summary.to_csv('species_detection_summary.csv')
        print("✅ Species summary saved to: species_detection_summary.csv")

    print("\n🎉 Analysis Complete! 🎉")


if __name__ == '__main__':
    main()