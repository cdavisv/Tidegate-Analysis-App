# environmental_analysis.py

import pandas as pd
from scipy import stats

def _analyze_single_gate(df, gate_col, bins, labels):
    """Helper function to analyze detection rates for a single gate column."""
    if gate_col not in df.columns:
        print(f"\nGate column '{gate_col}' not available for analysis.")
        return None
        
    df[f'{gate_col}_category'] = pd.cut(df[gate_col], bins=bins, labels=labels, right=False)
    
    analysis_df = df.groupby(f'{gate_col}_category', observed=True).agg(
        Detection_Rate=('has_camera_data', 'mean')
    )
    analysis_df['Detection_Rate_Pct'] = analysis_df['Detection_Rate'] * 100

    print(f"\n--- Detection Rate by {gate_col} ---")
    print(analysis_df['Detection_Rate_Pct'].round(2).to_string())
    return analysis_df

def analyze_environmental_factors(combined_df):
    """
    Analyzes and prints detection rates based on environmental factors.
    Now handles two separate gate analyses with data-driven bins.

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
    
    # FIX: Use the new, standardized column name for the hinge gate analysis.
    hinge_col_name = 'Gate_Opening_Top_Hinge_Deg' 
    hinge_gate_analysis = _analyze_single_gate(combined_df, hinge_col_name, hinge_bins, hinge_labels)

    # Initialize other variables
    tidal_analysis, temp_analysis = None, None

    # --- 3. Tidal Level Analysis ---
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
            print("\n--- Detection Rate by Tidal Level ---")
            print(tidal_analysis['Detection_Rate_Pct'].round(2).to_string())
    else:
        print("\nTidal level (Depth) data not available for analysis.")

    # --- 4. Weather Analysis ---
    if 'Air_Temp_C' in combined_df.columns:
        combined_df['temp_bin'] = pd.cut(combined_df['Air_Temp_C'], bins=5)
        temp_analysis = combined_df.groupby('temp_bin', observed=True).agg(
            Detection_Rate=('has_camera_data', 'mean')
        )
        temp_analysis['Detection_Rate_Pct'] = temp_analysis['Detection_Rate'] * 100
        print("\n--- Detection Rate by Air Temperature ---")
        print(temp_analysis['Detection_Rate_Pct'].round(2).to_string())
    else:
        print("\nWeather (Air Temp) data not available for analysis.")
        
    return mtr_gate_analysis, hinge_gate_analysis, tidal_analysis, temp_analysis