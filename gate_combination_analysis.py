# gate_combination_analysis.py

import pandas as pd
import numpy as np

def _create_and_print_pivot_summary(df, gate_category_col, analysis_title):
    """
    Helper function to create, print, and interpret a bird detection summary pivot table.
    """
    print(f"\n\n--- Bird Detection Rate (%) by {analysis_title} ---")

    # --- 1. Data Validation and Preparation ---
    # Added MTR_category and Hinge_category to required columns for the new logic
    required_cols = [gate_category_col, 'detailed_tidal_flow', 'is_bird_detection', 'MTR_category', 'Hinge_category']
    if not all(col in df.columns for col in required_cols):
        print(" -> Skipping analysis: Required columns not found.")
        return

    analysis_df = df.dropna(subset=required_cols).copy()
    if analysis_df.empty:
        print(" -> No data available for this analysis after filtering.")
        return

    # --- 2. Create and Clean Summary Table ---
    summary_table = (
        analysis_df.groupby([gate_category_col, 'detailed_tidal_flow'], observed=True)['is_bird_detection']
        .mean()
        .unstack()
        .fillna(0) * 100
    )
    summary_table.columns = summary_table.columns.astype(str)
    summary_table = summary_table.loc[:, summary_table.columns != 'nan']

    if summary_table.empty:
        print(" -> No bird activity detected for these conditions.")
        return

    print(summary_table.round(2))

    # --- 3. Hypothesis Test with 'Other' category handling ---
    if summary_table.values.max() > 0:
        best_rate = summary_table.values.max()
        # Find the position of the max value
        pos = np.where(summary_table.values == best_rate)
        # Get the corresponding index (gate state) and column (tidal state) names
        gate_state = summary_table.index[pos[0][0]]
        tidal_state = summary_table.columns[pos[1][0]]

        # If the top category is 'Other', find the specific sub-category driving the trend
        if gate_state == 'Other':
            # Filter the dataframe to the specific 'Other' condition that had the peak rate
            other_conditions_df = analysis_df[
                (analysis_df[gate_category_col] == 'Other') &
                (analysis_df['detailed_tidal_flow'] == tidal_state)
            ]

            if not other_conditions_df.empty and other_conditions_df['is_bird_detection'].any():
                # Find the specific sub-combination within 'Other' that is most attractive to birds
                top_combo_in_other = (
                    other_conditions_df.groupby(['MTR_category', 'Hinge_category'], observed=True)['is_bird_detection']
                    .mean()
                    .idxmax()
                )
                
                mtr_cat = top_combo_in_other[0]
                hinge_cat = top_combo_in_other[1]

                print(f"\nHYPOTHESIS TEST ({analysis_title}): Peak bird activity ({best_rate:.2f}%) occurs during '{gate_state}' conditions and a '{tidal_state}' tide.")
                print(f"  -> The specific combination driving this peak is MTR Gate: '{mtr_cat}' and Hinge Gate: '{hinge_cat}'.")
            else:
                # Fallback message in case filtering results in an empty or non-detection dataframe
                print(f"\nHYPOTHESIS TEST ({analysis_title}): Peak bird activity ({best_rate:.2f}%) occurs when the gate is '{gate_state}' and the tide is '{tidal_state}'.")

        else:
            # Original message for all non-'Other' categories
            print(f"\nHYPOTHESIS TEST ({analysis_title}): Peak bird activity ({best_rate:.2f}%) occurs when the gate is '{gate_state}' and the tide is '{tidal_state}'.")


def _analyze_top_detection_times(df):
    """
    Finds and provides a detailed summary of the time of day and gate state
    with the highest number of bird detections.
    """
    print("\n\n--- Top Bird Detection Time Analysis ---")
    
    # --- 1. Data Validation and Preparation ---
    required_cols = ['is_bird_detection', 'MTR_category', 'Hinge_category', 'DateTime', 'Gate_Opening_MTR_Deg', 'Gate_Opening_Top_Hinge_Deg', 'detailed_tidal_flow']
    if not all(col in df.columns for col in required_cols):
        print(" -> Skipping analysis: Required columns not found.")
        return

    # Create a detailed state description and extract the hour
    df['full_gate_state'] = 'MTR: ' + df['MTR_category'].astype(str) + ' | Hinge: ' + df['Hinge_category'].astype(str)
    df['hour'] = df['DateTime'].dt.hour
    
    bird_events = df[df['is_bird_detection']].copy()
    if bird_events.empty:
        print(" -> No bird detections found to analyze.")
        return

    # --- 2. Group and Find Top Times ---
    time_summary = bird_events.groupby(['full_gate_state', 'hour']).size().reset_index(name='detection_count')
    
    if time_summary.empty:
        print(" -> Could not summarize detection times.")
        return

    top_count = time_summary['detection_count'].max()
    top_times = time_summary[time_summary['detection_count'] == top_count]

    print(f"Peak activity involves {top_count} detection(s) in a single hour under these conditions:")

    # --- 3. Report Results with Improved Logic ---
    for _, row in top_times.iterrows():
        # Get all events that match the peak criteria
        all_peak_events = bird_events[
            (bird_events['full_gate_state'] == row['full_gate_state']) &
            (bird_events['hour'] == row['hour'])
        ]

        print(f"\n- At hour: {int(row['hour'])}:00")
        print(f"  Gate State: {row['full_gate_state']}")
        
        # --- Summarize average conditions across all detections ---
        print(f"\n  Summary across all {len(all_peak_events)} detections:")
        
        mtr_angle_avg = all_peak_events['Gate_Opening_MTR_Deg'].mean()
        print(f"    Avg MTR Gate Angle: {mtr_angle_avg:.2f}°")
        
        hinge_angle_avg = all_peak_events['Gate_Opening_Top_Hinge_Deg'].mean()
        if pd.notna(hinge_angle_avg):
            print(f"    Avg Hinge Gate Angle: {hinge_angle_avg:.2f}°")
        else:
            print("    Avg Hinge Gate Angle: N/A")

        tidal_flow_mode = all_peak_events['detailed_tidal_flow'].mode()
        dominant_tidal_flow = tidal_flow_mode[0] if not tidal_flow_mode.empty and pd.notna(tidal_flow_mode[0]) else "N/A"
        print(f"    Dominant Tidal Flow: {dominant_tidal_flow}")


        # --- Find and display a more complete example record ---
        print("\n  Example of a complete data record from this period:")
        
        complete_examples = all_peak_events.dropna(subset=['Gate_Opening_Top_Hinge_Deg', 'detailed_tidal_flow'])
        
        if not complete_examples.empty:
            best_example = complete_examples.iloc[0]
        else:
            best_example = all_peak_events.iloc[0] # Fallback to the first event
        
        print(f"    MTR Gate Angle: {best_example['Gate_Opening_MTR_Deg']:.2f}°")
        
        if pd.notna(best_example['Gate_Opening_Top_Hinge_Deg']):
            print(f"    Hinge Gate Angle: {best_example['Gate_Opening_Top_Hinge_Deg']:.2f}°")
        else:
            print("    Hinge Gate Angle: N/A")

        if pd.notna(best_example['detailed_tidal_flow']):
            print(f"    Tidal Flow: {best_example['detailed_tidal_flow']}")
        else:
            print("    Tidal Flow: N/A")


def run_gate_combination_analysis(combined_df):
    """
    Main function to run a series of complex gate combination analyses for bird detections.
    """
    # --- 1. Initial Data Preparation ---
    bird_species = [
        'Anatidae', 'Branta Canadensis', 'Megaceryle Alcyon', 'Podiceps Grisegena',
        'Ardea', 'Ardea Alba', 'Ardea Herodias', 'Mergus Merganser', 
        'Corvus Brachyrhynchos', 'Phalacrocoracidae', 'Anas Platyrhynchos', 'Cathartes Aura'
    ]
    combined_df['is_bird_detection'] = combined_df['Species'].isin(bird_species)

    if 'Depth' in combined_df.columns:
        combined_df['tidal_change_m_hr'] = combined_df['Depth'].diff() * 2
        conditions = [
            combined_df['tidal_change_m_hr'] > 0.05,
            combined_df['tidal_change_m_hr'] < -0.05,
            (combined_df['tidal_change_m_hr'].abs() <= 0.05) & (combined_df['Depth'] >= combined_df['Depth'].median()),
            (combined_df['tidal_change_m_hr'].abs() <= 0.05) & (combined_df['Depth'] < combined_df['Depth'].median())
        ]
        combined_df['detailed_tidal_flow'] = np.select(conditions, ['Rising', 'Falling', 'High Slack', 'Low Slack'], default=np.nan)

    if 'Gate_Opening_MTR_Deg' in combined_df.columns:
        combined_df['MTR_category'] = pd.cut(combined_df['Gate_Opening_MTR_Deg'], bins=[-1, 5, 39, 63, 88], labels=['Closed', 'Partially Open', 'Open', 'Wide Open'])
    if 'Gate_Opening_Top_Hinge_Deg' in combined_df.columns:
        combined_df['Hinge_category'] = pd.cut(combined_df['Gate_Opening_Top_Hinge_Deg'], bins=[-2, 4, 20, 35, 42], labels=['Closed', 'Partially Open', 'Open', 'Wide Open'])

    # --- 2. Run All Analyses ---

    # Analysis A: Simple Categories, Ignoring Degrees
    conditions_a = [
        (combined_df['MTR_category'] == 'Wide Open') | (combined_df['Hinge_category'] == 'Wide Open'),
        (combined_df['MTR_category'] == 'Open') | (combined_df['Hinge_category'] == 'Open'),
        (combined_df['MTR_category'] == 'Partially Open') | (combined_df['Hinge_category'] == 'Partially Open'),
        (combined_df['MTR_category'] == 'Closed') & (combined_df['Hinge_category'] == 'Closed')
    ]
    choices_a = ['Wide Open', 'Open', 'Partially Open', 'Closed']
    combined_df['simple_gate_category'] = np.select(conditions_a, choices_a, default='Other')
    _create_and_print_pivot_summary(combined_df, 'simple_gate_category', "Combined Gate State (Simple)")

    # Analysis B: Specific Degree-Based Combinations
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
    _create_and_print_pivot_summary(combined_df, 'specific_gate_combo', "Combined Gate State (Specific Combos)")

    # Analysis C: Top Detection Times Analysis
    _analyze_top_detection_times(combined_df)