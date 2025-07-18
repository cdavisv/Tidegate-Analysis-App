# gate_combination_analysis.py

import pandas as pd
import numpy as np

def _create_and_print_pivot_summary(df, category_col, title):
    """
    Placeholder helper function to create and print a pivot table summary.
    This function would typically summarize bird detections against the specified category.
    """
    if 'is_bird_detection' in df.columns and df['is_bird_detection'].any():
        print(f"\n--- Pivot Summary: {title} ---")
        try:
            pivot = pd.pivot_table(
                df, 
                values='is_bird_detection', 
                index=category_col, 
                columns='detailed_tidal_flow', 
                aggfunc='sum',
                fill_value=0
            )
            print(pivot)
        except Exception as e:
            print(f"Could not generate pivot table for '{title}': {e}")
    else:
        print(f"\n--- Skipping Pivot Summary for {title} (no bird detections or required columns) ---")

def _analyze_top_detection_times(df):
    """
    Placeholder helper function to analyze top detection times.
    """
    print("\n--- Analysis C: Top Detection Times (Placeholder) ---")
    if 'is_bird_detection' in df.columns and df['is_bird_detection'].any():
        print("Top detection time analysis would be performed here.")
    else:
        print("Skipping analysis due to no bird detections.")


def run_gate_combination_analysis(combined_df):
    """
    Main function to run a series of complex gate combination analyses.
    This version is robust and handles missing columns gracefully.
    """
    print("--- Running Full Gate Combination Analysis ---")
    
    # --- 1. Initial Data Preparation (with robust checks) ---
    bird_species = [
        'Anatidae', 'Branta Canadensis', 'Bucephala albeola' 'Megaceryle Alcyon', 
        'Podiceps Grisegena', 'Gavia Immer', 'Nannopterum auritus', 'Urile pelagicus',
        'Ardea', 'Ardea Alba', 'Ardea Herodias', 'Mergus Merganser', 
        'Corvus Brachyrhynchos', 'Phalacrocoracidae', 'Anas Platyrhynchos', 'Cathartes Aura'
    ]
    if 'Species' in combined_df.columns:
        combined_df['is_bird_detection'] = combined_df['Species'].isin(bird_species)
    else:
        print("Note: 'Species' column not found. Skipping bird detection analysis.")
        # Create a dummy column to prevent downstream errors
        combined_df['is_bird_detection'] = False

    if 'Depth' in combined_df.columns:
        combined_df['tidal_change_m_hr'] = combined_df['Depth'].diff() * 2
        conditions = [
            combined_df['tidal_change_m_hr'] > 0.05,
            combined_df['tidal_change_m_hr'] < -0.05,
            (combined_df['tidal_change_m_hr'].abs() <= 0.05) & (combined_df['Depth'] >= combined_df['Depth'].median()),
            (combined_df['tidal_change_m_hr'].abs() <= 0.05) & (combined_df['Depth'] < combined_df['Depth'].median())
        ]
        combined_df['detailed_tidal_flow'] = np.select(conditions, ['Rising', 'Falling', 'High Slack', 'Low Slack'], default=np.nan)
    else:
        print("Note: 'Depth' column not found. Skipping tidal flow analysis.")
        combined_df['detailed_tidal_flow'] = 'Unknown'


    # --- Safely create gate category columns ---
    # If the source column doesn't exist, the category column won't be created.
    if 'Gate_Opening_MTR_Deg' in combined_df.columns:
        combined_df['MTR_category'] = pd.cut(combined_df['Gate_Opening_MTR_Deg'], bins=[-1, 5, 39, 63, 88], labels=['Closed', 'Partially Open', 'Open', 'Wide Open'])
    else:
        print("Note: 'Gate_Opening_MTR_Deg' not found. 'MTR_category' will not be available.")

    if 'Gate_Opening_Top_Hinge_Deg' in combined_df.columns:
        combined_df['Hinge_category'] = pd.cut(combined_df['Gate_Opening_Top_Hinge_Deg'], bins=[-2, 4, 20, 35, 42], labels=['Closed', 'Partially Open', 'Open', 'Wide Open'])
    else:
        print("Note: 'Gate_Opening_Top_Hinge_Deg' not found. 'Hinge_category' will not be available.")

    # --- 2. Run All Analyses (with robust condition building) ---

    # For safety, create dummy series for categories if they don't exist.
    # This prevents KeyErrors in the logic below.
    mtr_cat = combined_df.get('MTR_category', pd.Series(dtype='str'))
    hinge_cat = combined_df.get('Hinge_category', pd.Series(dtype='str'))

    # --- Analysis A: Simple Categories, Ignoring Degrees ---
    if not mtr_cat.empty or not hinge_cat.empty:
        conditions_a = [
            (mtr_cat == 'Wide Open') | (hinge_cat == 'Wide Open'),
            (mtr_cat == 'Open') | (hinge_cat == 'Open'),
            (mtr_cat == 'Partially Open') | (hinge_cat == 'Partially Open'),
            (mtr_cat == 'Closed') & (hinge_cat == 'Closed')
        ]
        choices_a = ['Wide Open', 'Open', 'Partially Open', 'Closed']
        combined_df['simple_gate_category'] = np.select(conditions_a, choices_a, default='Other')
        _create_and_print_pivot_summary(combined_df, 'simple_gate_category', "Combined Gate State (Simple)")
    else:
        print("Skipping Analysis A: No gate category columns available.")
        combined_df['simple_gate_category'] = 'Not Applicable'

    # --- Analysis B: Specific Degree-Based Combinations ---
    if not mtr_cat.empty and not hinge_cat.empty:
        conditions_b = [
            (mtr_cat == 'Closed') & (hinge_cat == 'Closed'),
            (mtr_cat == 'Partially Open') & (hinge_cat == 'Closed'),
            (mtr_cat == 'Partially Open') & (hinge_cat.isin(['Partially Open', 'Open'])),
            (mtr_cat == 'Open') & (hinge_cat == 'Wide Open'),
            (mtr_cat == 'Wide Open')
        ]
        choices_b = [
            'MTR Closed & Hinge Closed',
            'MTR Partially Open & Hinge Closed',
            'MTR Partially Open & Hinge Partially/Open',
            'MTR Open & Hinge Wide Open',
            'MTR Wide Open'
        ]
        combined_df['specific_gate_combo'] = np.select(conditions_b, choices_b, default='Other')
        _create_and_print_pivot_summary(combined_df, 'specific_gate_combo', "Combined Gate State (Specific Combos)")
    else:
        print("Skipping Analysis B: Both 'MTR_category' and 'Hinge_category' are required.")
        combined_df['specific_gate_combo'] = 'Not Applicable'

    # --- Analysis C: Top Detection Times Analysis ---
    _analyze_top_detection_times(combined_df)
    
    print("\n--- Full Analysis Complete ---")
    return combined_df


"""
# --- Example Usage ---
if __name__ == '__main__':
    # Test Case: A full dataset with all columns
    print("\n" + "="*50 + "\n--- Running Test Case 1: Full Data ---")
    full_data = {
        'Species': ['Anas Platyrhynchos', 'Ardea Herodias', 'Gavia Immer', 'Anas Platyrhynchos'],
        'Depth': [3.1, 3.2, 3.15, 3.0],
        'Gate_Opening_MTR_Deg': [80, 10, 45, 5],
        'Gate_Opening_Top_Hinge_Deg': [40, 5, 25, 3]
    }
    df_full = pd.DataFrame(full_data)
    result_full = run_gate_combination_analysis(df_full.copy())
    print("\nFinal DataFrame (Full Data):")
    print(result_full[['simple_gate_category', 'specific_gate_combo', 'detailed_tidal_flow']])

    # Test Case: A partial dataset missing gate info
    print("\n" + "="*50 + "\n--- Running Test Case 2: Missing Gate Data ---")
    partial_data = {
        'Species': ['Ardea Alba', 'Corvus Brachyrhynchos'],
        'Depth': [2.5, 2.6]
    }
    df_partial = pd.DataFrame(partial_data)
    result_partial = run_gate_combination_analysis(df_partial.copy())
    print("\nFinal DataFrame (Partial Data):")
    print(result_partial[['simple_gate_category', 'specific_gate_combo', 'detailed_tidal_flow']])
"""