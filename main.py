
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


import pandas as pd
import data_loader
import data_combiner
import species_analysis  # Keep original for compatibility
import environmental_analysis  # Keep original for compatibility
import comprehensive_analysis  # Updated comprehensive analysis
import gate_combination_analysis
import bird_tide_analysis
import tide_cycle_analysis
import visualization

# --- Configuration ---
# Adjust CAMERA_DATA_PATH and WATER_DATA_PATH to your files

#Example files
#CAMERA_DATA_PATH = 'willanch_camera_final.csv'
#WATER_DATA_PATH = 'willanch_sensor_final.csv'
#CAMERA_DATA_PATH = 'Palouse_Combined_Camera_Full_Imageset.csv'  # Next camera data path
WATER_DATA_PATH = 'Palouse_Tide_Data_Combined_newest.csv'  # Next water data path
CAMERA_DATA_PATH = 'Larson_Camera_Final.csv' 
#WATER_DATA_PATH = 

def main():
    """
    Main function to run both Camera Activity Pattern and Wildlife Detection Efficiency analysis pipelines.
    """
    print("--- Starting Comprehensive Analysis Pipeline ---")

    # 1. Load Data
    camera_df = data_loader.load_and_prepare_camera_data(CAMERA_DATA_PATH)
    water_df = data_loader.load_and_prepare_water_data(WATER_DATA_PATH)

    # Check if the data loading failed before continuing
    if camera_df is None:
        print("\nCRITICAL: Camera data loading failed or returned no data. Cannot proceed with analysis. Exiting.")
        return
    if water_df is None:
        print("\nCRITICAL: Water data loading failed or returned no data. Cannot proceed with analysis. Exiting.")
        return

    # 2. Combine Data
    combined_df = data_combiner.combine_data(
        camera_df=camera_df,
        water_df=water_df
        #max_interp_hours=0.25
    )
    
    if combined_df.empty:
        print("Data combination resulted in an empty DataFrame. Exiting.")
        return

    combined_df.to_csv('combined_data_output.csv', index=False)
    print("\n -> Successfully saved the combined DataFrame to 'combined_data_output.csv'")

    # 3. Run Comprehensive Analysis (Both Camera Activity and Detection Efficiency)
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    comprehensive_results = comprehensive_analysis.run_comprehensive_analysis(combined_df)
    
    # 4. Run Additional Analyses (using original methods for compatibility)
    print("\n" + "="*80)
    print("RUNNING ADDITIONAL ANALYSES")
    print("="*80)
    
    # Use original species analysis return format for compatibility
    species_summary_df, species_df = species_analysis.analyze_species_diversity(combined_df)
    
    env_results = environmental_analysis.analyze_environmental_factors(combined_df)
    bird_tide_results = bird_tide_analysis.analyze_bird_tide_gate_behavior(combined_df)
    gate_combo_df = gate_combination_analysis.run_gate_combination_analysis(combined_df)
    tide_cycle_df, detection_by_tide, phase_detection = tide_cycle_analysis.analyze_tide_cycle_detections(gate_combo_df)
    species_tide_table = tide_cycle_analysis.analyze_species_tide_preferences(tide_cycle_df)

    # 5. Generate Visualizations
    print("\n\n--- Generating All Visualizations ---")
    
    # Original visualizations
    if species_summary_df is not None and not species_summary_df.empty:
        visualization.plot_species_analysis(species_summary_df)

    if env_results:
        visualization.plot_environmental_factors(*env_results)
    
    if bird_tide_results is not None:
         visualization.plot_bird_analysis(bird_tide_results, combined_df)

    visualization.create_safe_water_visualizations(combined_df)
    visualization.create_hypothesis_visualizations(gate_combo_df)
    
    tide_viz_results = (detection_by_tide, phase_detection, species_tide_table)
    visualization.create_tide_cycle_visualizations(tide_cycle_df, tide_viz_results)
    
    # 6. Generate Additional Comprehensive Visualizations
    import additional_visualizations
    additional_visualizations.create_all_additional_visualizations(comprehensive_results, combined_df)
    
    # 7. Generate Summary Report
    generate_summary_report(comprehensive_results)
    
    print("\n--- Comprehensive Analysis Pipeline Complete ---")


def generate_summary_report(comprehensive_results):
    """
    Generates a summary report comparing Camera Activity Pattern vs Wildlife Detection Efficiency methods.
    """
    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT")
    print("="*80)
    
    comparison = comprehensive_results['comparison']
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   • Total monitoring periods: {comparison['total_periods']:,}")
    print(f"   • Camera active periods: {comparison['camera_periods']:,}")
    print(f"   • Animal detection events: {comparison['animal_detections']:,}")
    print(f"   • Camera activity rate: {comparison['camera_activity_rate']:.3f}%")
    
    print(f"\n🔍 ANALYSIS RESULTS:")
    print(f"   • Camera activity rate (all time periods): {comparison['camera_activity_rate']:.3f}%")
    print(f"   • Wildlife detection success rate (camera active): {comparison['camera_detection_rate']:.1f}%")
    print(f"   • Overall detection rate (all time periods): {comparison['overall_detection_rate']:.3f}%")
    
    print(f"\n📈 KEY INSIGHTS:")
    
    # Get top species from both methods
    activity_top = comprehensive_results['camera_activity']['species_summary'].head(3)
    efficiency_top = comprehensive_results['detection_efficiency']['species_summary'].head(3)
    
    print(f"   • Top 3 species (both methods agree):")
    for i, (species, data) in enumerate(activity_top.iterrows(), 1):
        print(f"     {i}. {species}: {data['Total_Count']:.0f} individuals, {data['Detection_Events']} events")
    
    print(f"\n💡 ANALYSIS METHOD RECOMMENDATIONS:")
    print(f"   • Use CAMERA ACTIVITY PATTERN ANALYSIS for:")
    print(f"     - Understanding when cameras operate most effectively")
    print(f"     - Identifying monitoring bias and equipment performance")
    print(f"     - Operational planning and resource allocation")
    print(f"")
    print(f"   • Use WILDLIFE DETECTION EFFICIENCY ANALYSIS for:")
    print(f"     - Understanding wildlife behavior and optimal detection conditions")
    print(f"     - Conservation and habitat management decisions")
    print(f"     - Maximizing monitoring success rates")
    print(f"")
    print(f"   • Camera detection efficiency: {comparison['camera_detection_rate']:.1f}% - Good performance!")
    print(f"   • Consider monitoring during wide-open gate conditions for best results")
    
    # Save detailed results to CSV
    try:
        activity_summary = comprehensive_results['camera_activity']['species_summary']
        efficiency_summary = comprehensive_results['detection_efficiency']['species_summary']
        
        # Merge the summaries for comparison
        comparison_df = pd.merge(
            activity_summary.add_suffix('_Camera_Activity'),
            efficiency_summary.add_suffix('_Detection_Efficiency'),
            left_index=True,
            right_index=True,
            how='outer'
        ).fillna(0)
        
        comparison_df.to_csv('analysis_comparison.csv')
        print(f"\n💾 Detailed method comparison saved to 'analysis_comparison.csv'")
        
    except Exception as e:
        print(f"\n⚠️  Could not save comparison CSV: {e}")


if __name__ == '__main__':
    main()