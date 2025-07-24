# bird_tide_analysis.py

import pandas as pd
import numpy as np


def analyze_bird_tide_gate_behavior(combined_df):
    """
    Performs a detailed analysis of ALL animal detection patterns for multiple gate configurations.
    Now includes birds, mammals, and unknown species.
    """
    print("\n\n=== WILDLIFE & TIDE GATE BEHAVIOR ANALYSIS ===")
    
    # UPDATED: Include ALL species detected by cameras (not just birds)
    # Get all unique species except 'No_Animals_Detected'
    all_detected_species = combined_df[combined_df['has_camera_data'] & (combined_df['Species'] != 'No_Animals_Detected')]['Species'].unique()
    
    print(f"Including all detected species: {list(all_detected_species)}")
    
    # Create detection flag for ALL animals (not just birds)
    combined_df['is_animal_detection'] = combined_df['Species'].isin(all_detected_species)
    
    total_animal_detections = combined_df['is_animal_detection'].sum()
    if total_animal_detections == 0:
        print("No animal detections found in the data. Skipping analysis.")
        return pd.DataFrame()

    print(f"Found {total_animal_detections} total animal detections to analyze.")

    if 'Depth' in combined_df.columns:
        combined_df['tidal_change_m_hr'] = combined_df['Depth'].diff() * 2
        
        slack_tide_threshold = 0.05
        median_depth = combined_df['Depth'].median()

        conditions = [
            combined_df['tidal_change_m_hr'] > slack_tide_threshold,
            combined_df['tidal_change_m_hr'] < -slack_tide_threshold,
            (combined_df['tidal_change_m_hr'].abs() <= slack_tide_threshold) & (combined_df['Depth'] >= median_depth),
            (combined_df['tidal_change_m_hr'].abs() <= slack_tide_threshold) & (combined_df['Depth'] < median_depth)
        ]
        choices = ['Rising', 'Falling', 'High Slack', 'Low Slack']
        
        combined_df['detailed_tidal_flow'] = np.select(conditions, choices, default=np.nan)
        
        print(f"Rows with indeterminate tidal flow: {combined_df['detailed_tidal_flow'].isna().sum()}")
        
    else:
        print("Cannot calculate tidal change: 'Depth' column not found.")
        return pd.DataFrame()

    mtr_gate_col = 'Gate_Opening_MTR_Deg_category'
    # Update the helper function to use the new detection column
    mtr_summary_table = _create_and_print_summary_all_species(combined_df, mtr_gate_col, "MTR Gate")

    hinge_gate_col = 'Gate_Opening_Top_Hinge_Deg'
    if hinge_gate_col in combined_df.columns:
        hinge_bins = [-2, 4, 20, 35, 42]
        hinge_labels = ['Closed (-2-4°)', 'Partially Open (4-20°)', 'Open (20-35°)', 'Wide Open (>35°)']
        combined_df['Gate_Opening_Top_Hinge_Deg_category'] = pd.cut(combined_df[hinge_gate_col], bins=hinge_bins, labels=hinge_labels, right=False)
        _create_and_print_summary_all_species(combined_df, 'Gate_Opening_Top_Hinge_Deg_category', "Top Hinge Gate")
    
    return mtr_summary_table


def _create_and_print_summary_all_species(df, gate_category_col, analysis_title):
    """
    Helper function to create, print, and interpret ALL animal detection summary table.
    Updated to use 'is_animal_detection' instead of 'is_bird_detection'.
    """
    if gate_category_col not in df.columns or 'detailed_tidal_flow' not in df.columns:
        print(f"\nSkipping '{analysis_title}' analysis: Required columns not found.")
        return pd.DataFrame()

    # Enhanced filtering to remove NaN, 'Unknown', and null values
    analysis_df = df[
        df['detailed_tidal_flow'].notna() & 
        (df['detailed_tidal_flow'] != 'Unknown') &
        (df['detailed_tidal_flow'] != 'nan') &  # Remove string 'nan'
        (~df['detailed_tidal_flow'].isna())     # Remove actual NaN
    ].copy()
    analysis_df = analysis_df.dropna(subset=[gate_category_col])

    if analysis_df.empty:
        print(f"No data available for '{analysis_title}' analysis after filtering.")
        return pd.DataFrame()

    # UPDATED: Use 'is_animal_detection' instead of 'is_bird_detection'
    summary_table = (
        analysis_df.groupby([gate_category_col, 'detailed_tidal_flow'], observed=True)['is_animal_detection']
        .mean()
        .unstack()
        .fillna(0)
        * 100
    )

    if summary_table.empty:
        print(f"\nNo animal activity detected for {analysis_title} conditions.")
        return summary_table

    print(f"\n\n--- Animal Detection Rate (%) by {analysis_title} Status and DETAILED Tidal Flow ---")
    print(summary_table.round(2))

    if summary_table.values.max() > 0:
        best_rate = summary_table.values.max()
        pos = np.where(summary_table.values == best_rate)
        gate_state = summary_table.index[pos[0][0]]
        tidal_state = summary_table.columns[pos[1][0]]
        
        print(f"\nHYPOTHESIS TEST ({analysis_title}): Peak animal activity ({best_rate:.2f}%) occurs when the gate is '{gate_state}' and the tide is '{tidal_state}'.")
    
    return summary_table