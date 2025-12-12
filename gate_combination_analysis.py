"""
Combined tide gate interaction analysis module.

This module analyzes wildlife detection patterns across combined gate states,
considering both MTR and top-hinge gate configurations simultaneously.

Key features:
- Unified animal detection flag
- Detailed tidal flow classification
- Multiple gate combination hypotheses
- Pivot-table-based detection rate summaries

The analysis supports complex interaction hypotheses about how multiple gate
mechanisms jointly influence wildlife movement and detection probability.
"""


import pandas as pd
import numpy as np


def run_gate_combination_analysis(combined_df):
    """
    Main function to run a series of complex gate combination analyses.
    Updated to include ALL animals (birds, mammals, unknown species).
    """
    print("\n\n--- GATE COMBINATION ANALYSIS (All Animals) ---")
    
    # FIXED: Create detection flag correctly - only actual animals, not no-animal observations
    if 'Species' in combined_df.columns:
        combined_df['is_animal_detection'] = (
            combined_df['has_camera_data'] &
            combined_df['Species'].notna() &
            (combined_df['Notes'] != 'No animals detected')
        )
        
        # Get all unique species except null/nan values
        all_detected_species = combined_df[combined_df['is_animal_detection']]['Species'].dropna().unique()
        print(f"Including all detected species: {list(all_detected_species)}")
    else:
        print("Note: 'Species' column not found. Skipping animal detection analysis.")
        combined_df['is_animal_detection'] = False

    # Define tidal flow state
    if 'Depth' in combined_df.columns:
        combined_df['tidal_change_m_hr'] = combined_df['Depth'].diff() * 2
        conditions = [
            combined_df['tidal_change_m_hr'] > 0.05,
            combined_df['tidal_change_m_hr'] < -0.05,
            (combined_df['tidal_change_m_hr'].abs() <= 0.05) & (combined_df['Depth'] >= combined_df['Depth'].median()),
            (combined_df['tidal_change_m_hr'].abs() <= 0.05) & (combined_df['Depth'] < combined_df['Depth'].median())
        ]
        choices_obj = [np.array(choice, dtype=object) for choice in ['Rising', 'Falling', 'High Slack', 'Low Slack']]
        combined_df['detailed_tidal_flow'] = np.select(conditions, choices_obj, default=np.nan)
    else:
        print("Note: 'Depth' column not found. Skipping tidal flow analysis.")
        combined_df['detailed_tidal_flow'] = np.nan

    # Define gate categories if columns exist
    if 'Gate_Opening_MTR_Deg' in combined_df.columns:
        combined_df['MTR_category'] = pd.cut(combined_df['Gate_Opening_MTR_Deg'], bins=[-1, 5, 39, 63, 88], labels=['Closed', 'Partially Open', 'Open', 'Wide Open'])
    if 'Gate_Opening_Top_Hinge_Deg' in combined_df.columns:
        combined_df['Hinge_category'] = pd.cut(combined_df['Gate_Opening_Top_Hinge_Deg'], bins=[-2, 4, 20, 35, 42], labels=['Closed', 'Partially Open', 'Open', 'Wide Open'])

    # Analysis 1: Simple combined gate state
    if 'MTR_category' in combined_df.columns and 'Hinge_category' in combined_df.columns:
        conditions_a = [
            (combined_df['MTR_category'] == 'Wide Open') | (combined_df['Hinge_category'] == 'Wide Open'),
            (combined_df['MTR_category'] == 'Open') | (combined_df['Hinge_category'] == 'Open'),
            (combined_df['MTR_category'] == 'Partially Open') | (combined_df['Hinge_category'] == 'Partially Open'),
            (combined_df['MTR_category'] == 'Closed') & (combined_df['Hinge_category'] == 'Closed')
        ]
        choices_a = ['Wide Open', 'Open', 'Partially Open', 'Closed']
        combined_df['simple_gate_category'] = np.select(conditions_a, choices_a, default='Other')
        _create_and_print_pivot_summary_all_species(combined_df, 'simple_gate_category', "Combined Gate State (Simple)")
    else:
        print("\nSkipping Analysis A: Required gate category columns not available.")

    # Analysis 2: Specific gate combinations
    if 'MTR_category' in combined_df.columns and 'Hinge_category' in combined_df.columns:
        conditions_b = [
            (combined_df['MTR_category'] == 'Closed') & (combined_df['Hinge_category'] == 'Closed'),
            (combined_df['MTR_category'] == 'Partially Open') & (combined_df['Hinge_category'] == 'Closed'),
            (combined_df['MTR_category'] == 'Partially Open') & (combined_df['Hinge_category'].isin(['Partially Open', 'Open'])),
            (combined_df['MTR_category'] == 'Open') & (combined_df['Hinge_category'] == 'Wide Open'),
            (combined_df['MTR_category'] == 'Wide Open')
        ]
        choices_b = [
            'MTR Closed & Hinge Closed',
            'MTR Partially Open & Hinge Closed',
            'MTR Partially Open & Hinge Partially/Open',
            'MTR Open & Hinge Wide Open',
            'MTR Wide Open'
        ]
        combined_df['specific_gate_combo'] = np.select(conditions_b, choices_b, default='Other')
        _create_and_print_pivot_summary_all_species(combined_df, 'specific_gate_combo', "Combined Gate State (Specific Combos)")
    else:
        print("\nSkipping Analysis B: Both 'MTR_category' and 'Hinge_category' are required.")

    return combined_df


def _create_and_print_pivot_summary_all_species(df, category_col, title):
    """
    Creates and prints a pivot table summary showing ALL animal detection rates (%)
    and identifies the peak activity hypothesis.
    Updated to use 'is_animal_detection' instead of 'is_bird_detection'.
    """
    print(f"\n--- Animal Detection Rate (%) by {title} ---")

    if 'is_animal_detection' not in df.columns or not df['is_animal_detection'].any():
        print(f"Skipping pivot summary for {title}: No animal detections found.")
        return

    if 'detailed_tidal_flow' not in df.columns or category_col not in df.columns:
        print(f"Skipping pivot summary for {title}: Missing required columns.")
        return

    # Enhanced filtering to remove NaN, 'Unknown', and null values
    analysis_df = df[
        df['detailed_tidal_flow'].notna() & 
        (df['detailed_tidal_flow'] != 'Unknown') &
        (df['detailed_tidal_flow'] != 'nan') &  # Remove string 'nan'
        (~df['detailed_tidal_flow'].isna())     # Remove actual NaN
    ].copy()
    analysis_df = analysis_df.dropna(subset=[category_col])

    if analysis_df.empty:
        print(f"Skipping pivot summary for {title}: No data after filtering.")
        return

    try:
        # UPDATED: Use 'is_animal_detection' instead of 'is_bird_detection'
        pivot = pd.pivot_table(
            analysis_df,
            values='is_animal_detection',
            index=category_col,
            columns='detailed_tidal_flow',
            aggfunc='mean',
            fill_value=0
        )
        pivot = (pivot * 100).round(2)
        
        # Remove any remaining 'nan' columns from the pivot table
        if 'nan' in pivot.columns:
            pivot = pivot.drop(columns=['nan'])
        
        print(pivot)

        if pivot.empty or pivot.values.max() == 0:
            print(f"\nHYPOTHESIS TEST ({title}): No significant animal activity detected for these conditions.")
        else:
            # Find the position of the highest rate
            peak_rate = pivot.values.max()
            # Get the row (gate state) and column (tide state) names for the peak
            peak_gate_state = pivot.stack().idxmax()[0]
            peak_tidal_state = pivot.stack().idxmax()[1]
            
            print(f"\nHYPOTHESIS TEST ({title}): Peak animal activity ({peak_rate:.2f}%) occurs when the gate is '{peak_gate_state}' and the tide is '{peak_tidal_state}'.")

    except Exception as e:
        print(f"Could not generate pivot table for '{title}': {e}")