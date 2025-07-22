# comprehensive_analysis.py

import pandas as pd
import numpy as np

def analyze_species_diversity_original(combined_df):
    """
    ORIGINAL ANALYSIS: Includes all sensor data as potential detection periods.
    This treats every sensor reading as a potential detection event.
    """
    print("\n" + "="*60)
    print("ORIGINAL ANALYSIS (All Time Periods)")
    print("="*60)
    
    if combined_df.empty:
        print("Combined DataFrame is empty, skipping species diversity analysis.")
        return pd.DataFrame(), pd.DataFrame()

    species_df = combined_df[combined_df['Species'] != 'No_Animals_Detected'].copy()

    if species_df.empty:
        print("No animal detections found in camera data.")
        return pd.DataFrame(), species_df

    print("\n=== SPECIES DIVERSITY ANALYSIS (Original Method) ===")
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


def analyze_environmental_factors_original(combined_df):
    """
    ORIGINAL ANALYSIS: Detection rates across all time periods (including sensor-only data).
    """
    print("\n=== ENVIRONMENTAL ANALYSIS (Original Method) ===")
    
    if combined_df.empty:
        print("Combined DataFrame is empty. Cannot perform environmental analysis.")
        return None, None, None, None

    # --- 1. MTR Gate Analysis ---
    mtr_bins = [-1, 5, 39, 63, 88]
    mtr_labels = ['Closed (0-5°)', 'Partially Open (5-39°)', 'Open (39-63°)', 'Wide Open (>63°)']
    mtr_gate_analysis = _analyze_single_gate_original(combined_df, 'Gate_Opening_MTR_Deg', mtr_bins, mtr_labels)

    # --- 2. Top Hinge Gate Analysis ---
    hinge_bins = [-2, 4, 20, 35, 42]
    hinge_labels = ['Closed (-2-4°)', 'Partially Open (4-20°)', 'Open (20-35°)', 'Wide Open (>35°)']
    hinge_gate_analysis = _analyze_single_gate_original(combined_df, 'Gate_Opening_Top_Hinge_Deg', hinge_bins, hinge_labels)

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
                Detection_Rate=('has_camera_data', 'mean')
            )
            tidal_analysis['Detection_Rate_Pct'] = tidal_analysis['Detection_Rate'] * 100
            print("\n--- Detection Rate by Tidal Level (Original) ---")
            print(tidal_analysis['Detection_Rate_Pct'].round(2).to_string())

    # --- 4. Weather Analysis ---
    temp_analysis = None
    if 'Air_Temp_C' in combined_df.columns:
        combined_df['temp_bin'] = pd.cut(combined_df['Air_Temp_C'], bins=5)
        temp_analysis = combined_df.groupby('temp_bin', observed=True).agg(
            Detection_Rate=('has_camera_data', 'mean')
        )
        temp_analysis['Detection_Rate_Pct'] = temp_analysis['Detection_Rate'] * 100
        print("\n--- Detection Rate by Air Temperature (Original) ---")
        print(temp_analysis['Detection_Rate_Pct'].round(2).to_string())
        
    return mtr_gate_analysis, hinge_gate_analysis, tidal_analysis, temp_analysis


def _analyze_single_gate_original(df, gate_col, bins, labels):
    """Helper function for original analysis."""
    if gate_col not in df.columns:
        print(f"\nGate column '{gate_col}' not available for analysis.")
        return None
        
    df[f'{gate_col}_category'] = pd.cut(df[gate_col], bins=bins, labels=labels, right=False)
    
    analysis_df = df.groupby(f'{gate_col}_category', observed=True).agg(
        Detection_Rate=('has_camera_data', 'mean')
    )
    analysis_df['Detection_Rate_Pct'] = analysis_df['Detection_Rate'] * 100

    print(f"\n--- Detection Rate by {gate_col} (Original) ---")
    print(analysis_df['Detection_Rate_Pct'].round(2).to_string())
    return analysis_df


def analyze_species_diversity_corrected(combined_df):
    """
    CORRECTED ANALYSIS: Only analyzes camera observation periods (406 rows).
    """
    print("\n" + "="*60)
    print("CORRECTED ANALYSIS (Camera Observation Periods Only)")
    print("="*60)
    
    # Filter to only camera observation periods
    camera_observations = combined_df[combined_df['has_camera_data']].copy()
    
    if camera_observations.empty:
        print("No camera observations to analyze.")
        return pd.DataFrame(), pd.DataFrame()

    # Remove "No_Animals_Detected" entries if they exist
    species_df = camera_observations[camera_observations['Species'] != 'No_Animals_Detected'].copy()

    if species_df.empty:
        print("No animal detections found in camera data.")
        return pd.DataFrame(), species_df

    print("\n=== SPECIES DIVERSITY ANALYSIS (Corrected Method) ===")
    print(f"Total camera observation periods: {len(camera_observations):,}")
    print(f"Observation periods with animals: {len(species_df):,}")
    print(f"Unique species detected: {species_df['Species'].nunique()}")
    print(f"Animal detection rate during camera observations: {(len(species_df)/len(camera_observations))*100:.1f}%")

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


def analyze_environmental_factors_corrected(combined_df):
    """
    CORRECTED ANALYSIS: Detection rates only across camera observation periods.
    """
    print("\n=== ENVIRONMENTAL ANALYSIS (Corrected Method) ===")
    
    # Filter to only camera observation periods
    camera_observations = combined_df[combined_df['has_camera_data']].copy()
    
    if camera_observations.empty:
        print("No camera observations to analyze.")
        return None, None, None, None
    
    print(f"Analyzing detection rates across {len(camera_observations)} camera observations")
    
    # Create a binary "animal_detected" column
    camera_observations['animal_detected'] = (
        camera_observations['Species'] != 'No_Animals_Detected'
    ).astype(int)
    
    overall_detection_rate = camera_observations['animal_detected'].mean()
    print(f"Overall animal detection rate during camera observations: {overall_detection_rate:.1%}")
    
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
            Detection_Rate=('animal_detected', 'mean')
        )
        mtr_gate_analysis['Detection_Rate_Pct'] = mtr_gate_analysis['Detection_Rate'] * 100

        print("\n--- Detection Rate by MTR Gate Opening (Corrected) ---")
        for idx, row in mtr_gate_analysis.iterrows():
            print(f"{idx}: {row['Animal_Detections']}/{row['Total_Observations']} ({row['Detection_Rate_Pct']:.1f}%)")

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
            Detection_Rate=('animal_detected', 'mean')
        )
        hinge_gate_analysis['Detection_Rate_Pct'] = hinge_gate_analysis['Detection_Rate'] * 100

        print("\n--- Detection Rate by Top Hinge Gate Opening (Corrected) ---")
        for idx, row in hinge_gate_analysis.iterrows():
            print(f"{idx}: {row['Animal_Detections']}/{row['Total_Observations']} ({row['Detection_Rate_Pct']:.1f}%)")

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
            Detection_Rate=('animal_detected', 'mean')
        )
        tidal_analysis['Detection_Rate_Pct'] = tidal_analysis['Detection_Rate'] * 100

        print("\n--- Detection Rate by Tidal Level (Corrected) ---")
        for idx, row in tidal_analysis.iterrows():
            print(f"{idx}: {row['Animal_Detections']}/{row['Total_Observations']} ({row['Detection_Rate_Pct']:.1f}%)")

    # --- 4. Temperature Analysis ---
    temp_analysis = None
    if 'Air_Temp_C' in camera_observations.columns:
        camera_observations['temp_bin'] = pd.cut(camera_observations['Air_Temp_C'], bins=5)
        
        temp_analysis = camera_observations.groupby('temp_bin', observed=True).agg(
            Total_Observations=('DateTime', 'count'),
            Animal_Detections=('animal_detected', 'sum'),
            Detection_Rate=('animal_detected', 'mean')
        )
        temp_analysis['Detection_Rate_Pct'] = temp_analysis['Detection_Rate'] * 100

        print("\n--- Detection Rate by Air Temperature (Corrected) ---")
        for idx, row in temp_analysis.iterrows():
            print(f"{idx}: {row['Animal_Detections']}/{row['Total_Observations']} ({row['Detection_Rate_Pct']:.1f}%)")
        
    return mtr_gate_analysis, hinge_gate_analysis, tidal_analysis, temp_analysis


def compare_analysis_methods(combined_df):
    """
    Provides a detailed comparison between the original and corrected analysis methods.
    """
    print("\n" + "="*80)
    print("ANALYSIS METHOD COMPARISON")
    print("="*80)
    
    # Get basic statistics
    total_rows = len(combined_df)
    camera_rows = combined_df['has_camera_data'].sum()
    animal_detections = len(combined_df[(combined_df['has_camera_data']) & (combined_df['Species'] != 'No_Animals_Detected')])
    
    print("\n=== DATA OVERVIEW ===")
    print(f"Total time periods in dataset: {total_rows:,}")
    print(f"Camera observation periods: {camera_rows:,}")
    print(f"Animal detection events: {animal_detections:,}")
    print(f"Camera activity rate: {(camera_rows/total_rows)*100:.3f}%")
    print(f"Animal detection rate (all time): {(animal_detections/total_rows)*100:.3f}%")
    print(f"Animal detection rate (camera active): {(animal_detections/camera_rows)*100:.1f}%")
    
    print("\n=== METHOD COMPARISON ===")
    print("\n┌─────────────────────────────────────────────────────────────────────────┐")
    print("│                         ORIGINAL vs CORRECTED ANALYSIS                 │")
    print("├─────────────────────────────────────────────────────────────────────────┤")
    print("│ ORIGINAL METHOD:                                                        │")
    print("│ • Uses all 36,123 time periods (including sensor-only data)            │")
    print("│ • Treats every sensor reading as a potential detection opportunity     │")
    print("│ • Detection rate = detections / all time periods                       │")
    print("│ • Shows how often animals appear across the entire monitoring period   │")
    print("│                                                                         │")
    print("│ CORRECTED METHOD:                                                       │")
    print("│ • Uses only 406 camera observation periods                             │")
    print("│ • Only analyzes times when cameras were actually recording             │")
    print("│ • Detection rate = detections / camera observation periods             │")
    print("│ • Shows how often animals appear when cameras are active               │")
    print("└─────────────────────────────────────────────────────────────────────────┘")
    
    # Calculate some example comparisons
    if 'Gate_Opening_MTR_Deg' in combined_df.columns:
        print("\n=== EXAMPLE: MTR GATE COMPARISON ===")
        
        # Original method
        combined_df_temp = combined_df.copy()
        mtr_bins = [-1, 5, 39, 63, 88]
        mtr_labels = ['Closed (0-5°)', 'Partially Open (5-39°)', 'Open (39-63°)', 'Wide Open (>63°)']
        combined_df_temp['gate_category'] = pd.cut(combined_df_temp['Gate_Opening_MTR_Deg'], bins=mtr_bins, labels=mtr_labels)
        
        original_rates = combined_df_temp.groupby('gate_category', observed=True)['has_camera_data'].mean() * 100
        
        # Corrected method
        camera_obs = combined_df_temp[combined_df_temp['has_camera_data']].copy()
        camera_obs['animal_detected'] = (camera_obs['Species'] != 'No_Animals_Detected').astype(int)
        corrected_rates = camera_obs.groupby('gate_category', observed=True)['animal_detected'].mean() * 100
        
        print("\nDetection Rates by Gate Position:")
        print("Gate State                | Original Method | Corrected Method | Difference")
        print("─" * 75)
        for gate_state in original_rates.index:
            if gate_state in corrected_rates.index:
                orig_rate = original_rates[gate_state]
                corr_rate = corrected_rates[gate_state]
                diff = corr_rate - orig_rate
                print(f"{gate_state:<25} | {orig_rate:>13.1f}% | {corr_rate:>14.1f}% | {diff:>+8.1f}%")
    
    print("\n=== INTERPRETATION DIFFERENCES ===")
    print("\nOriginal Method tells us:")
    print("• How often animals appear during the entire monitoring period")
    print("• Includes times when cameras weren't active (sensor-only data)")
    print("• Shows overall wildlife activity patterns across all conditions")
    print("• Detection rates appear low (< 2%) because most time periods have no camera data")
    
    print("\nCorrected Method tells us:")
    print("• How often animals appear when cameras are actually recording")
    print("• Only includes active camera observation periods")
    print("• Shows wildlife detection success rate during camera operations")
    print("• Detection rates appear high (likely 80-100%) because cameras detect animals most times they're active")
    
    print("\n=== WHICH METHOD TO USE WHEN ===")
    print("\nUse ORIGINAL METHOD for:")
    print("• Understanding overall ecosystem activity patterns")
    print("• Comparing environmental conditions across the full time series")
    print("• Questions like: 'How does gate position affect overall wildlife presence?'")
    
    print("\nUse CORRECTED METHOD for:")
    print("• Understanding camera detection efficiency")
    print("• Optimizing camera placement and operation")
    print("• Questions like: 'When cameras are active, which conditions lead to more detections?'")
    
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
    Main function to run both original and corrected analyses, then compare them.
    """
    # Run original analysis
    orig_species_summary, orig_species_df = analyze_species_diversity_original(combined_df)
    orig_env_results = analyze_environmental_factors_original(combined_df)
    
    # Run corrected analysis
    corr_species_summary, corr_species_df = analyze_species_diversity_corrected(combined_df)
    corr_env_results = analyze_environmental_factors_corrected(combined_df)
    
    # Compare methods
    comparison_stats = compare_analysis_methods(combined_df)
    
    return {
        'original': {
            'species_summary': orig_species_summary,
            'species_df': orig_species_df,
            'env_results': orig_env_results
        },
        'corrected': {
            'species_summary': corr_species_summary,
            'species_df': corr_species_df,
            'env_results': corr_env_results
        },
        'comparison': comparison_stats
    }