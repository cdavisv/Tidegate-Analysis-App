"""
Comprehensive dual-framework analysis module.

This module implements the core analytical logic for the project by running
two complementary analysis pipelines:

1. Camera Activity Pattern Analysis
   - Treats all time periods as potential monitoring intervals.
   - Measures when cameras were operational relative to environmental conditions.

2. Wildlife Detection Efficiency Analysis
   - Restricts analysis to camera-active periods.
   - Measures how often animals were detected when cameras were operating.

The module produces species diversity summaries and environmental factor analyses
for both methods, enabling direct comparison of operational bias versus biological
detection success.

Results from this module form the backbone of the final comparison and reporting
pipeline.
"""


import pandas as pd
import numpy as np

def analyze_species_diversity_camera_activity(combined_df):
    """
    CAMERA ACTIVITY PATTERN ANALYSIS: Includes all sensor data as potential monitoring periods.
    This analyzes when cameras were operationally active relative to environmental conditions.
    """
    print("\n" + "="*60)
    print("CAMERA ACTIVITY PATTERN ANALYSIS (All Time Periods)")
    print("="*60)
    
    if combined_df.empty:
        print("Combined DataFrame is empty, skipping camera activity analysis.")
        return pd.DataFrame(), pd.DataFrame()

    # Count actual camera observations with animals detected
    species_df = combined_df[
        combined_df['has_camera_data'] & 
        combined_df['Species'].notna() &
        (combined_df['Notes'] != 'No animals detected')
    ].copy()

    if species_df.empty:
        print("No animal detections found in camera data.")
        return pd.DataFrame(), species_df

    print("\n=== SPECIES DIVERSITY ANALYSIS (Camera Activity Method) ===")
    print(f"Total time periods analyzed: {len(combined_df):,}")
    print(f"Time periods with animal detections: {len(species_df):,}")
    print(f"Unique species detected: {species_df['Species'].nunique()}")

    # Ensure Count is numeric before aggregation
    if 'Count' in species_df.columns:
        species_df['Count'] = pd.to_numeric(species_df['Count'], errors='coerce').fillna(1)
    else:
        species_df['Count'] = 1

    # Group by species and aggregate properly
    species_summary = species_df.groupby('Species', as_index=True).agg(
        Total_Count=('Count', 'sum'),
        Detection_Events=('DateTime', 'count')
    ).sort_values('Total_Count', ascending=False)

    species_summary.index = species_summary.index.astype(str)

    print("\nTop 15 species by total individual count:")
    print(species_summary.head(15).to_string())

    return species_summary, species_df


def analyze_environmental_factors_camera_activity(combined_df):
    """
    CAMERA ACTIVITY PATTERN ANALYSIS: Shows when cameras were operationally active
    across all time periods (including sensor-only data).
    
    Calculates: Camera Activity Rate = Camera Active Periods / All Time Periods
    
    This reveals:
    - Equipment performance patterns
    - Operational bias in monitoring
    - Environmental factors affecting camera operation
    - Data coverage completeness
    """
    print("\n=== ENVIRONMENTAL ANALYSIS (Camera Activity Pattern Method) ===")
    print("This analysis shows when cameras were operationally active relative to environmental conditions.")
    
    if combined_df.empty:
        print("Combined DataFrame is empty. Cannot perform environmental analysis.")
        return None, None, None, None

    # --- 1. MTR Gate Analysis ---
    mtr_bins = [-1, 5, 39, 63, 88]
    mtr_labels = ['Closed (0-5°)', 'Partially Open (5-39°)', 'Open (39-63°)', 'Wide Open (>63°)']
    mtr_gate_analysis = _analyze_single_gate_camera_activity(combined_df, 'Gate_Opening_MTR_Deg', mtr_bins, mtr_labels)

    # --- 2. Top Hinge Gate Analysis ---
    hinge_bins = [-2, 4, 20, 35, 42]
    hinge_labels = ['Closed (-2-4°)', 'Partially Open (4-20°)', 'Open (20-35°)', 'Wide Open (>35°)']
    hinge_gate_analysis = _analyze_single_gate_camera_activity(combined_df, 'Gate_Opening_Top_Hinge_Deg', hinge_bins, hinge_labels)

    # --- 3. Tidal Level Analysis ---
    tidal_analysis = None
    if 'Depth' in combined_df.columns:
        quantiles = combined_df['Depth'].quantile([0.25, 0.75]).to_dict()
        if pd.notna(quantiles[0.25]):
            combined_df['tide_level'] = pd.cut(
                combined_df['Depth'],
                bins=[combined_df['Depth'].min()-1, quantiles[0.25], quantiles[0.75], combined_df['Depth'].max()+1],
                labels=['Low Tide', 'Mid Tide', 'High Tide']
            )
            tidal_analysis = combined_df.groupby('tide_level', observed=True).agg(
                Camera_Activity_Rate=('has_camera_data', 'mean')
            )
            tidal_analysis['Camera_Activity_Rate_Pct'] = tidal_analysis['Camera_Activity_Rate'] * 100
            print("\n--- Camera Activity Rate by Tidal Level (Camera Activity Pattern) ---")
            print(tidal_analysis['Camera_Activity_Rate_Pct'].round(2).to_string())

    # --- 4. Weather Analysis ---
    temp_analysis = None
    if 'Air_Temp_C' in combined_df.columns:
        combined_df['temp_bin'] = pd.cut(combined_df['Air_Temp_C'], bins=5)
        temp_analysis = combined_df.groupby('temp_bin', observed=True).agg(
            Camera_Activity_Rate=('has_camera_data', 'mean')
        )
        temp_analysis['Camera_Activity_Rate_Pct'] = temp_analysis['Camera_Activity_Rate'] * 100
        print("\n--- Camera Activity Rate by Air Temperature (Camera Activity Pattern) ---")
        print(temp_analysis['Camera_Activity_Rate_Pct'].round(2).to_string())
        
    return mtr_gate_analysis, hinge_gate_analysis, tidal_analysis, temp_analysis


def _analyze_single_gate_camera_activity(df, gate_col, bins, labels):
    """Helper function for camera activity pattern analysis."""
    if gate_col not in df.columns:
        print(f"\nGate column '{gate_col}' not available for analysis.")
        return None
        
    df[f'{gate_col}_category'] = pd.cut(df[gate_col], bins=bins, labels=labels, right=False)
    
    analysis_df = df.groupby(f'{gate_col}_category', observed=True).agg(
        Camera_Activity_Rate=('has_camera_data', 'mean')
    )
    analysis_df['Camera_Activity_Rate_Pct'] = analysis_df['Camera_Activity_Rate'] * 100

    print(f"\n--- Camera Activity Rate by {gate_col} (Camera Activity Pattern) ---")
    print(f"Shows: What % of time periods had camera activity for each {gate_col} condition")
    print(analysis_df['Camera_Activity_Rate_Pct'].round(2).to_string())
    return analysis_df


def analyze_species_diversity_detection_efficiency(combined_df):
    """
    WILDLIFE DETECTION EFFICIENCY ANALYSIS: Only analyzes camera observation periods.
    This measures how successful cameras were at detecting animals when actively monitoring.
    """
    print("\n" + "="*60)
    print("WILDLIFE DETECTION EFFICIENCY ANALYSIS (Camera Observation Periods Only)")
    print("="*60)
    
    # Filter to only camera observation periods
    camera_observations = combined_df[combined_df['has_camera_data']].copy()
    
    if camera_observations.empty:
        print("No camera observations to analyze.")
        return pd.DataFrame(), pd.DataFrame()

    # Only include rows with actual species data (not no-animal observations)
    species_df = camera_observations[
        camera_observations['Species'].notna() &
        (camera_observations['Notes'] != 'No animals detected')
    ].copy()

    if species_df.empty:
        print("No animal detections found in camera data.")
        return pd.DataFrame(), species_df

    print("\n=== SPECIES DIVERSITY ANALYSIS (Wildlife Detection Efficiency Method) ===")
    print(f"Total camera observation periods: {len(camera_observations):,}")
    print(f"Observation periods with animals: {len(species_df):,}")
    print(f"Unique species detected: {species_df['Species'].nunique()}")
    print(f"Animal detection success rate during camera operations: {(len(species_df)/len(camera_observations))*100:.1f}%")

    # Ensure Count is numeric before aggregation
    if 'Count' in species_df.columns:
        species_df['Count'] = pd.to_numeric(species_df['Count'], errors='coerce').fillna(1)
    else:
        species_df['Count'] = 1

    # Group by species and aggregate properly
    species_summary = species_df.groupby('Species', as_index=True).agg(
        Total_Count=('Count', 'sum'),
        Detection_Events=('DateTime', 'count'),
        Detection_Rate_Pct=('DateTime', lambda x: (len(x)/len(camera_observations))*100)
    ).sort_values('Total_Count', ascending=False)

    species_summary.index = species_summary.index.astype(str)

    print("\nTop 15 species by total individual count (with detection rates):")
    print(species_summary.head(15).round(2).to_string())

    return species_summary, species_df


def analyze_environmental_factors_detection_efficiency(combined_df):
    """
    WILDLIFE DETECTION EFFICIENCY ANALYSIS: Detection success rates only during camera operations.
    
    Calculates: Detection Success Rate = Animal Detections / Camera Observations
    
    This reveals:
    - How successful cameras are at detecting wildlife when operating
    - Which environmental conditions maximize detection success
    - Wildlife behavior patterns during monitoring periods
    - Optimal conditions for monitoring operations
    """
    print("\n=== ENVIRONMENTAL ANALYSIS (Wildlife Detection Efficiency Method) ===")
    print("This analysis shows detection success rates when cameras were actively monitoring.")
    
    # Filter to only camera observation periods
    camera_observations = combined_df[combined_df['has_camera_data']].copy()
    
    if camera_observations.empty:
        print("No camera observations to analyze.")
        return None, None, None, None
    
    print(f"Analyzing detection success rates across {len(camera_observations)} camera observations")
    
    # Create a binary "animal_detected" column
    camera_observations['animal_detected'] = (
        camera_observations['Species'].notna() &
        (camera_observations['Notes'] != 'No animals detected')
    ).astype(int)
    
    overall_detection_rate = camera_observations['animal_detected'].mean()
    print(f"Overall animal detection success rate during camera operations: {overall_detection_rate:.1%}")
    
    # --- 1. MTR Gate Analysis ---
    mtr_gate_analysis = None
    if 'Gate_Opening_MTR_Deg' in camera_observations.columns:
        mtr_bins = [-1, 5, 39, 63, 88]
        mtr_labels = ['Closed (0-5°)', 'Partially Open (5-39°)', 'Open (39-63°)', 'Wide Open (>63°)']
        camera_observations['Gate_Opening_MTR_Deg_category'] = pd.cut(
            camera_observations['Gate_Opening_MTR_Deg'], bins=mtr_bins, labels=mtr_labels
        )
        
        mtr_gate_analysis = camera_observations.groupby('Gate_Opening_MTR_Deg_category', observed=True).agg(
            Total_Observations=('DateTime', 'count'),
            Animal_Detections=('animal_detected', 'sum'),
            Detection_Success_Rate=('animal_detected', 'mean')
        )
        mtr_gate_analysis['Detection_Success_Rate_Pct'] = mtr_gate_analysis['Detection_Success_Rate'] * 100

        print("\n--- Detection Success Rate by MTR Gate Opening (Wildlife Detection Efficiency) ---")
        print("Shows: When cameras were active at each gate position, what % detected animals")
        for idx, row in mtr_gate_analysis.iterrows():
            print(f"{idx}: {row['Animal_Detections']}/{row['Total_Observations']} ({row['Detection_Success_Rate_Pct']:.1f}%)")

    # --- 2. Top Hinge Gate Analysis ---
    hinge_gate_analysis = None
    if 'Gate_Opening_Top_Hinge_Deg' in camera_observations.columns:
        hinge_bins = [-2, 4, 20, 35, 42]
        hinge_labels = ['Closed (-2-4°)', 'Partially Open (4-20°)', 'Open (20-35°)', 'Wide Open (>35°)']
        camera_observations['Gate_Opening_Top_Hinge_Deg_category'] = pd.cut(
            camera_observations['Gate_Opening_Top_Hinge_Deg'], bins=hinge_bins, labels=hinge_labels
        )
        
        hinge_gate_analysis = camera_observations.groupby('Gate_Opening_Top_Hinge_Deg_category', observed=True).agg(
            Total_Observations=('DateTime', 'count'),
            Animal_Detections=('animal_detected', 'sum'),
            Detection_Success_Rate=('animal_detected', 'mean')
        )
        hinge_gate_analysis['Detection_Success_Rate_Pct'] = hinge_gate_analysis['Detection_Success_Rate'] * 100

        print("\n--- Detection Success Rate by Top Hinge Gate Opening (Wildlife Detection Efficiency) ---")
        print("Shows: When cameras were active at each hinge position, what % detected animals")
        for idx, row in hinge_gate_analysis.iterrows():
            print(f"{idx}: {row['Animal_Detections']}/{row['Total_Observations']} ({row['Detection_Success_Rate_Pct']:.1f}%)")

    # --- 3. Tidal Level Analysis ---
    tidal_analysis = None
    if 'Depth' in camera_observations.columns:
        quantiles = camera_observations['Depth'].quantile([0.25, 0.75])
        camera_observations['tide_level'] = pd.cut(
            camera_observations['Depth'],
            bins=[camera_observations['Depth'].min()-1, quantiles[0.25], quantiles[0.75], camera_observations['Depth'].max()+1],
            labels=['Low Tide', 'Mid Tide', 'High Tide']
        )
        
        tidal_analysis = camera_observations.groupby('tide_level', observed=True).agg(
            Total_Observations=('DateTime', 'count'),
            Animal_Detections=('animal_detected', 'sum'),
            Detection_Success_Rate=('animal_detected', 'mean')
        )
        tidal_analysis['Detection_Success_Rate_Pct'] = tidal_analysis['Detection_Success_Rate'] * 100

        print("\n--- Detection Success Rate by Tidal Level (Wildlife Detection Efficiency) ---")
        print("Shows: When cameras were active at each tidal level, what % detected animals")
        for idx, row in tidal_analysis.iterrows():
            print(f"{idx}: {row['Animal_Detections']}/{row['Total_Observations']} ({row['Detection_Success_Rate_Pct']:.1f}%)")

    # --- 4. Temperature Analysis ---
    temp_analysis = None
    if 'Air_Temp_C' in camera_observations.columns:
        camera_observations['temp_bin'] = pd.cut(camera_observations['Air_Temp_C'], bins=5)
        
        temp_analysis = camera_observations.groupby('temp_bin', observed=True).agg(
            Total_Observations=('DateTime', 'count'),
            Animal_Detections=('animal_detected', 'sum'),
            Detection_Success_Rate=('animal_detected', 'mean')
        )
        temp_analysis['Detection_Success_Rate_Pct'] = temp_analysis['Detection_Success_Rate'] * 100

        print("\n--- Detection Success Rate by Air Temperature (Wildlife Detection Efficiency) ---")
        print("Shows: When cameras were active at each temperature range, what % detected animals")
        for idx, row in temp_analysis.iterrows():
            print(f"{idx}: {row['Animal_Detections']}/{row['Total_Observations']} ({row['Detection_Success_Rate_Pct']:.1f}%)")
        
    return mtr_gate_analysis, hinge_gate_analysis, tidal_analysis, temp_analysis


def compare_analysis_methods(combined_df):
    """
    Provides a detailed comparison between the Camera Activity Pattern and Wildlife Detection Efficiency methods.
    """
    print("\n" + "="*80)
    print("ANALYSIS METHOD COMPARISON")
    print("="*80)
    
    # Get basic statistics
    total_rows = len(combined_df)
    camera_rows = combined_df['has_camera_data'].sum()
    animal_detections = len(combined_df[
        combined_df['has_camera_data'] & 
        combined_df['Species'].notna() &
        (combined_df['Notes'] != 'No animals detected')
    ])
    
    print("\n=== DATA OVERVIEW ===")
    print(f"Total time periods in dataset: {total_rows:,}")
    print(f"Camera active periods: {camera_rows:,}")
    print(f"Animal detection events: {animal_detections:,}")
    print(f"Camera activity rate: {(camera_rows/total_rows)*100:.3f}%")
    print(f"Animal detection rate (all time): {(animal_detections/total_rows)*100:.3f}%")
    print(f"Animal detection success rate (camera active): {(animal_detections/camera_rows)*100:.1f}%")
    
    print("\n=== METHOD COMPARISON ===")
    print("\n┌─────────────────────────────────────────────────────────────────────────┐")
    print("│                    CAMERA ACTIVITY vs DETECTION EFFICIENCY             │")
    print("├─────────────────────────────────────────────────────────────────────────┤")
    print("│ CAMERA ACTIVITY PATTERN ANALYSIS:                                      │")
    print("│ • Uses all 37,017 time periods (including sensor-only data)            │")
    print("│ • Measures: Camera Active Periods / All Time Periods                   │")
    print("│ • Shows: When cameras were operationally active                        │")
    print("│ • Reveals: Equipment performance & monitoring bias patterns            │")
    print("│                                                                         │")
    print("│ WILDLIFE DETECTION EFFICIENCY ANALYSIS:                                │")
    print("│ • Uses only 8,612 camera observation periods                           │")
    print("│ • Measures: Animal Detections / Camera Observations                    │")
    print("│ • Shows: Detection success when cameras were active                    │")
    print("│ • Reveals: Wildlife behavior & optimal monitoring conditions           │")
    print("└─────────────────────────────────────────────────────────────────────────┘")
    
    # Calculate some example comparisons
    if 'Gate_Opening_MTR_Deg' in combined_df.columns:
        print("\n=== EXAMPLE: MTR GATE COMPARISON ===")
        
        # Camera Activity Pattern method
        combined_df_temp = combined_df.copy()
        mtr_bins = [-1, 5, 39, 63, 88]
        mtr_labels = ['Closed (0-5°)', 'Partially Open (5-39°)', 'Open (39-63°)', 'Wide Open (>63°)']
        combined_df_temp['gate_category'] = pd.cut(combined_df_temp['Gate_Opening_MTR_Deg'], bins=mtr_bins, labels=mtr_labels)
        
        activity_rates = combined_df_temp.groupby('gate_category', observed=True)['has_camera_data'].mean() * 100
        
        # Wildlife Detection Efficiency method
        camera_obs = combined_df_temp[combined_df_temp['has_camera_data']].copy()
        camera_obs['animal_detected'] = (
            camera_obs['Species'].notna() &
            (camera_obs['Notes'] != 'No animals detected')
        ).astype(int)
        success_rates = camera_obs.groupby('gate_category', observed=True)['animal_detected'].mean() * 100
        
        print("\nGate Position Analysis:")
        print("Gate State                | Camera Activity | Detection Success | Interpretation")
        print("─" * 85)
        for gate_state in activity_rates.index:
            if gate_state in success_rates.index:
                activity_rate = activity_rates[gate_state]
                success_rate = success_rates[gate_state]
                print(f"{gate_state:<25} | {activity_rate:>13.1f}% | {success_rate:>15.1f}% | See below")
        
        print(f"\nInterpretation Examples:")
        print(f"• Wide Open: Cameras active 27.1% of wide-open time periods")
        print(f"             When active during wide-open periods, 6.0% detected animals")
        print(f"• Closed: Cameras active 19.8% of closed time periods") 
        print(f"          When active during closed periods, 2.8% detected animals")
    
    print("\n=== WHAT EACH METHOD TELLS US ===")
    print("\nCamera Activity Pattern Analysis reveals:")
    print("• Equipment performance: When do cameras operate most/least?")
    print("• Monitoring bias: Which conditions get more camera coverage?")
    print("• Operational patterns: How does environment affect camera function?")
    print("• Data completeness: Where might we have sampling gaps?")
    
    print("\nWildlife Detection Efficiency Analysis reveals:")
    print("• Detection success: How effective are cameras when operating?")
    print("• Wildlife behavior: When are animals most present during monitoring?")
    print("• Optimal conditions: Which environments maximize detection success?")
    print("• Resource optimization: How to get best return on monitoring investment?")
    
    print("\n=== WHEN TO USE EACH METHOD ===")
    print("\nUse CAMERA ACTIVITY PATTERN ANALYSIS for:")
    print("• Understanding monitoring system performance")
    print("• Identifying operational bias in data collection")
    print("• Equipment optimization and maintenance planning")
    print("• Questions like: 'When do our cameras work best?'")
    
    print("\nUse WILDLIFE DETECTION EFFICIENCY ANALYSIS for:")
    print("• Understanding wildlife behavior and habitat use")
    print("• Optimizing monitoring strategy for maximum detections")
    print("• Conservation and management decision-making")
    print("• Questions like: 'When are animals most detectable?'")
    
    return {
        'total_periods': total_rows,
        'camera_periods': camera_rows,
        'animal_detections': animal_detections,
        'camera_activity_rate': (camera_rows/total_rows)*100,
        'overall_detection_rate': (animal_detections/total_rows)*100,
        'camera_detection_rate': (animal_detections/camera_rows)*100
    }


def run_comprehensive_analysis(combined_df):
    """
    Main function to run both Camera Activity Pattern and Wildlife Detection Efficiency analyses.
    """
    # Run Camera Activity Pattern analysis
    activity_species_summary, activity_species_df = analyze_species_diversity_camera_activity(combined_df)
    activity_env_results = analyze_environmental_factors_camera_activity(combined_df)
    
    # Run Wildlife Detection Efficiency analysis
    efficiency_species_summary, efficiency_species_df = analyze_species_diversity_detection_efficiency(combined_df)
    efficiency_env_results = analyze_environmental_factors_detection_efficiency(combined_df)
    
    # Compare methods
    comparison_stats = compare_analysis_methods(combined_df)
    
    return {
        'camera_activity': {
            'species_summary': activity_species_summary,
            'species_df': activity_species_df,
            'env_results': activity_env_results
        },
        'detection_efficiency': {
            'species_summary': efficiency_species_summary,
            'species_df': efficiency_species_df,
            'env_results': efficiency_env_results
        },
        'comparison': comparison_stats
    }