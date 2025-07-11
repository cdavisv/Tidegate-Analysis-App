# bird_tide_analysis.py

import pandas as pd
import numpy as np

def analyze_bird_tide_gate_behavior(combined_df):
    """
    Performs a detailed analysis of bird detection patterns, with a definitive
    fix to ensure the final summary table is clean.
    """
    print("\n\n=== BIRD & TIDE GATE BEHAVIOR ANALYSIS (DEFINITIVE FIX) ===")
    
    # --- 1. Identify Bird Species ---
    # !!! ACTION REQUIRED: Please edit this list to include ALL bird species from your dataset !!!
    bird_species = [
        'Anatidae', 'Branta Canadensis', 'Megaceryle Alcyon', 'Podiceps Grisegena',
        'Ardea', 'Ardea Alba', 'Ardea Herodias', 'Mergus Merganser', 
        'Corvus Brachyrhynchos', 'Phalacrocoracidae', 'Anas Platyrhynchos', 'Cathartes Aura'
        # Add your other bird species here, for example: ', 'Your_Other_Bird_1', 'Your_Other_Bird_2'
    ]
    
    combined_df['is_bird_detection'] = combined_df['Species'].isin(bird_species)
    
    total_bird_detections = combined_df['is_bird_detection'].sum()
    if total_bird_detections == 0:
        print("No bird detections found in the data. Skipping analysis.")
        return pd.DataFrame()

    print(f"Found {total_bird_detections} total bird detections to analyze.")

    # --- 2. Calculate Tidal Change & Detailed Tidal State ---
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

    # --- 3. Detailed Analysis with Definitive Cleanup ---
    gate_col_category = 'Gate_Opening_MTR_Deg_category'
    if gate_col_category in combined_df.columns:

        analysis_df = combined_df.dropna(subset=['detailed_tidal_flow', gate_col_category]).copy()
        
        if analysis_df.empty:
            print("No data available after filtering for valid tidal and gate states.")
            return pd.DataFrame()

        # --- FINAL FIX FOR TABLE CREATION & NaN DISPLAY ---
        summary_table = (
            analysis_df.groupby([gate_col_category, 'detailed_tidal_flow'], observed=True)['is_bird_detection']
            .mean()
            .unstack() # Create the table
            .fillna(0) # Replace any NaN from empty groups with 0
            * 100
        )
        # --- END OF FIX ---

        print("\n--- Bird Detection Rate (%) by Gate Status and DETAILED Tidal Flow ---")
        print(summary_table.round(2))

        # Interpretation
        if not summary_table.empty and summary_table.values.max() > 0:
            
            best_rate = summary_table.values.max()
            pos = np.where(summary_table.values == best_rate)
            
            gate_state = summary_table.index[pos[0][0]]
            tidal_state = summary_table.columns[pos[1][0]]

            condition_text = f"the gate is '{gate_state}' and the tide is '{tidal_state}'"
            print(f"\nHYPOTHESIS TEST: Peak bird activity ({best_rate:.2f}%) occurs when {condition_text}.")
        else:
            print("\nNo bird activity detected in the summarized conditions.")
        
        return summary_table
    else:
        print("Cannot perform detailed analysis: Gate categories not found.")
        return pd.DataFrame()