# bird_tide_analysis.py

import pandas as pd
import numpy as np

def _create_and_print_summary(df, gate_category_col, analysis_title):
    """
    Helper function to create, print, and interpret a bird detection summary table.
    """
    if gate_category_col not in df.columns or 'detailed_tidal_flow' not in df.columns:
        print(f"\nSkipping '{analysis_title}' analysis: Required columns not found.")
        return pd.DataFrame()

    # --- FIX: More robust filtering. Removes both NaN and string 'Unknown' ---
    analysis_df = df.copy()
    analysis_df = analysis_df[analysis_df['detailed_tidal_flow'].notna() & (analysis_df['detailed_tidal_flow'] != 'Unknown')]
    analysis_df = analysis_df.dropna(subset=[gate_category_col])
    # --- FIX END ---

    if analysis_df.empty:
        print(f"No data available for '{analysis_title}' analysis after filtering.")
        return pd.DataFrame()

    summary_table = (
        analysis_df.groupby([gate_category_col, 'detailed_tidal_flow'], observed=True)['is_bird_detection']
        .mean()
        .unstack()
        .fillna(0)
        * 100
    )

    if summary_table.empty:
        print(f"\nNo bird activity detected for {analysis_title} conditions.")
        return summary_table

    print(f"\n\n--- Bird Detection Rate (%) by {analysis_title} Status and DETAILED Tidal Flow ---")
    print(summary_table.round(2))

    if summary_table.values.max() > 0:
        best_rate = summary_table.values.max()
        pos = np.where(summary_table.values == best_rate)
        gate_state = summary_table.index[pos[0][0]]
        tidal_state = summary_table.columns[pos[1][0]]
        
        print(f"\nHYPOTHESIS TEST ({analysis_title}): Peak bird activity ({best_rate:.2f}%) occurs when the gate is '{gate_state}' and the tide is '{tidal_state}'.")
    
    return summary_table


def analyze_bird_tide_gate_behavior(combined_df):
    """
    Performs a detailed analysis of bird detection patterns for multiple gate configurations.
    """
    print("\n\n=== BIRD & TIDE GATE BEHAVIOR ANALYSIS ===")
    
    bird_species = [
        'Anatidae', 'Branta Canadensis', 'Megaceryle Alcyon', 'Podiceps Grisegena',
        'Ardea', 'Ardea Alba', 'Ardea Herodias', 'Mergus Merganser', 
        'Corvus Brachyrhynchos', 'Phalacrocoracidae', 'Anas Platyrhynchos', 'Cathartes Aura'
    ]
    
    combined_df['is_bird_detection'] = combined_df['Species'].isin(bird_species)
    
    total_bird_detections = combined_df['is_bird_detection'].sum()
    if total_bird_detections == 0:
        print("No bird detections found in the data. Skipping analysis.")
        return pd.DataFrame()

    print(f"Found {total_bird_detections} total bird detections to analyze.")

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
    else:
        print("Cannot calculate tidal change: 'Depth' column not found.")
        return pd.DataFrame()

    mtr_gate_col = 'Gate_Opening_MTR_Deg_category'
    mtr_summary_table = _create_and_print_summary(combined_df, mtr_gate_col, "MTR Gate")

    hinge_gate_col = 'Gate_Opening_Top_Hinge_Deg'
    if hinge_gate_col in combined_df.columns:
        hinge_bins = [-2, 4, 20, 35, 42]
        hinge_labels = ['Closed (-2-4°)', 'Partially Open (4-20°)', 'Open (20-35°)', 'Wide Open (>35°)']
        combined_df['Gate_Opening_Top_Hinge_Deg_category'] = pd.cut(combined_df[hinge_gate_col], bins=hinge_bins, labels=hinge_labels, right=False)
        _create_and_print_summary(combined_df, 'Gate_Opening_Top_Hinge_Deg_category', "Top Hinge Gate")
    
    return mtr_summary_table