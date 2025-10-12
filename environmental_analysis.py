# environmental_analysis.py

import pandas as pd
from scipy import stats

def _analyze_single_gate(df, gate_col, bins, labels):
    """
    Helper function to analyze detection rates for a single gate column.
    FIXED: Now uses proper animal detection logic and avoids SettingWithCopyWarning.
    """
    if gate_col not in df.columns:
        print(f"\nGate column '{gate_col}' not available for analysis.")
        return None
    
    # Handle string boolean values
    df_temp = df.copy()
    if df_temp['has_camera_data'].dtype == 'object':
        df_temp['has_camera_data'] = df_temp['has_camera_data'] == 'True'
    
    # Filter to camera observations only
    camera_obs = df_temp[df_temp['has_camera_data']].copy()
    
    if camera_obs.empty:
        print(f"\nNo camera observations available for {gate_col} analysis.")
        return None
        
    # Create gate categories
    camera_obs[f'{gate_col}_category'] = pd.cut(camera_obs[gate_col], bins=bins, labels=labels, right=False)
    
    # Create proper animal detection flag
    camera_obs['animal_detected'] = (
        camera_obs['Species'].notna() &
        (camera_obs['Notes'] != 'No animals detected')
    ).astype(int)
    
    # Group by gate category and calculate detection rates
    analysis_df = camera_obs.groupby(f'{gate_col}_category', observed=True).agg(
        Total_Observations=('DateTime', 'count'),
        Animal_Detections=('animal_detected', 'sum'),
        Detection_Rate=('animal_detected', 'mean')
    )
    analysis_df['Detection_Rate_Pct'] = analysis_df['Detection_Rate'] * 100

    print(f"\n--- Detection Rate by {gate_col} ---")
    for idx, row in analysis_df.iterrows():
        print(f"{idx}: {row['Animal_Detections']}/{row['Total_Observations']} ({row['Detection_Rate_Pct']:.1f}%)")
    
    return analysis_df

def analyze_environmental_factors(combined_df):
    """
    Analyzes and prints detection rates based on environmental factors.
    FIXED: Now handles two separate gate analyses with correct animal detection logic and no warnings.

    Returns:
        tuple: Contains DataFrames for MTR gate, Hinge gate, tidal, and temp analysis.
    """
    if combined_df.empty:
        print("Combined DataFrame is empty. Cannot perform environmental analysis.")
        return None, None, None, None

    print("\n\n=== ENVIRONMENTAL ANALYSIS ===")
    
    # --- 1. MTR Gate Analysis ---
    mtr_bins = [-1, 5, 39, 63, 88]
    mtr_labels = ['Closed (0-5°)', 'Partially Open (5-39°)', 'Open (39-63°)', 'Wide Open (>63°)']
    mtr_gate_analysis = _analyze_single_gate(combined_df, 'Gate_Opening_MTR_Deg', mtr_bins, mtr_labels)

    # --- 2. Top Hinge Gate Analysis ---
    hinge_bins = [-2, 4, 20, 35, 42]
    hinge_labels = ['Closed (-2-4°)', 'Partially Open (4-20°)', 'Open (20-35°)', 'Wide Open (>35°)']
    hinge_gate_analysis = _analyze_single_gate(combined_df, 'Gate_Opening_Top_Hinge_Deg', hinge_bins, hinge_labels)

    # --- 3. Tidal Level Analysis ---
    tidal_analysis = None
    if 'Depth' in combined_df.columns:
        # Handle string boolean values
        df_temp = combined_df.copy()
        if df_temp['has_camera_data'].dtype == 'object':
            df_temp['has_camera_data'] = df_temp['has_camera_data'] == 'True'
        
        # Filter to camera observations only
        camera_obs = df_temp[df_temp['has_camera_data']].copy()
        
        if not camera_obs.empty:
            # Calculate quantiles from camera observations with depth data
            depth_data = camera_obs[camera_obs['Depth'].notna()].copy()
            
            if not depth_data.empty:
                quantiles = depth_data['Depth'].quantile([0.25, 0.75])
                
                # FIXED: Explicitly cast to avoid FutureWarning
                tide_level_categories = pd.cut(
                    depth_data['Depth'],
                    bins=[depth_data['Depth'].min()-0.01, quantiles[0.25], quantiles[0.75], depth_data['Depth'].max()+0.01],
                    labels=['Low Tide', 'Mid Tide', 'High Tide']
                )
                depth_data = depth_data.assign(tide_level=tide_level_categories)
                
                # FIXED: Explicitly cast to int to avoid FutureWarning
                animal_detected_values = (
                    depth_data['Species'].notna() &
                    (depth_data['Notes'] != 'No animals detected')
                ).astype(int)
                depth_data = depth_data.assign(animal_detected=animal_detected_values)
                
                tidal_analysis = depth_data.groupby('tide_level', observed=True).agg(
                    Total_Observations=('DateTime', 'count'),
                    Animal_Detections=('animal_detected', 'sum'),
                    Detection_Rate=('animal_detected', 'mean')
                )
                tidal_analysis['Detection_Rate_Pct'] = tidal_analysis['Detection_Rate'] * 100
                
                print("\n--- Detection Rate by Tidal Level ---")
                for idx, row in tidal_analysis.iterrows():
                    print(f"{idx}: {row['Animal_Detections']}/{row['Total_Observations']} ({row['Detection_Rate_Pct']:.1f}%)")
    else:
        print("\nTidal level (Depth) data not available for analysis.")

    # --- 4. Weather Analysis ---
    temp_analysis = None
    if 'Air_Temp_C' in combined_df.columns:
        # Handle string boolean values
        df_temp = combined_df.copy()
        if df_temp['has_camera_data'].dtype == 'object':
            df_temp['has_camera_data'] = df_temp['has_camera_data'] == 'True'
        
        # Filter to camera observations only
        camera_obs = df_temp[df_temp['has_camera_data']].copy()
        
        if not camera_obs.empty:
            temp_data = camera_obs[camera_obs['Air_Temp_C'].notna()].copy()
            
            if not temp_data.empty:
                # FIXED: Explicitly create temp_bin categories to avoid FutureWarning
                temp_bin_categories = pd.cut(temp_data['Air_Temp_C'], bins=5)
                temp_data = temp_data.assign(temp_bin=temp_bin_categories)
                
                # FIXED: Explicitly cast to int to avoid FutureWarning
                animal_detected_values = (
                    temp_data['Species'].notna() &
                    (temp_data['Notes'] != 'No animals detected')
                ).astype(int)
                temp_data = temp_data.assign(animal_detected=animal_detected_values)
                
                temp_analysis = temp_data.groupby('temp_bin', observed=True).agg(
                    Total_Observations=('DateTime', 'count'),
                    Animal_Detections=('animal_detected', 'sum'),
                    Detection_Rate=('animal_detected', 'mean')
                )
                temp_analysis['Detection_Rate_Pct'] = temp_analysis['Detection_Rate'] * 100
                
                print("\n--- Detection Rate by Air Temperature ---")
                for idx, row in temp_analysis.iterrows():
                    print(f"{idx}: {row['Animal_Detections']}/{row['Total_Observations']} ({row['Detection_Rate_Pct']:.1f}%)")
    else:
        print("\nWeather (Air Temp) data not available for analysis.")
        
    return mtr_gate_analysis, hinge_gate_analysis, tidal_analysis, temp_analysis