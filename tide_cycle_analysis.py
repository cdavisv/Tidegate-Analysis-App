"""
Tidal cycle and phase-based wildlife detection analysis module.

This module analyzes wildlife detection patterns across the full tidal cycle,
moving beyond static tidal categories to include continuous tidal phase modeling.
It explicitly excludes unknown or indeterminate tidal states to ensure valid
ecological interpretation.

Key capabilities:
- Classification of tidal states (rising, falling, high slack, low slack)
- Detection rate analysis using actual animal detections (not camera activity)
- Statistical testing of detection differences across tidal states
- Phase-based tidal analysis using normalized water depth
- Species-specific tidal preference analysis

This module supports fine-grained behavioral hypotheses about when wildlife is
most likely to be detected relative to tidal movement and phase.
"""


import pandas as pd
import numpy as np
from scipy import stats

def analyze_tide_cycle_detections(combined_df):
    """
    Analyzes animal detection patterns across the tidal cycle.
    This version filters out 'Unknown' states before analysis.
    FIXED: Now uses actual animal detections instead of camera activity.
    """
    print("\n\n=== TIDE CYCLE DETECTION ANALYSIS ===")
    
    if 'Depth' not in combined_df.columns:
        print("No tidal depth data available. Skipping tide cycle analysis.")
        return combined_df, pd.DataFrame(), pd.DataFrame()
    
    df = combined_df.copy()
    
    # FIXED: Create proper animal detection flag
    df['is_animal_detection'] = (
        df['has_camera_data'] &
        df['Species'].notna() &
        (df['Notes'] != 'No animals detected')
    )
    
    df['tidal_change_m_hr'] = df['Depth'].diff() * 2
    
    slack_threshold = 0.05
    median_depth = df['Depth'].median()
    
    conditions = [
        df['tidal_change_m_hr'] > slack_threshold,
        df['tidal_change_m_hr'] < -slack_threshold,
        (df['tidal_change_m_hr'].abs() <= slack_threshold) & (df['Depth'] >= median_depth),
        (df['tidal_change_m_hr'].abs() <= slack_threshold) & (df['Depth'] < median_depth)
    ]
    choices = ['Rising', 'Falling', 'High Slack', 'Low Slack']
    df['tidal_state'] = np.select(conditions, choices, default='Unknown')
    
    # --- PRIMARY FIX: Filter the dataframe BEFORE any analysis ---
    analysis_df = df[df['tidal_state'] != 'Unknown'].copy()
    print(f"Filtered out {(len(df) - len(analysis_df)):,} rows with 'Unknown' tidal state.")
    # --- END FIX ---
    
    if analysis_df.empty:
        print("No data remains after filtering for known tidal states.")
        return df, pd.DataFrame(), pd.DataFrame()

    # FIXED: Use animal detection flag instead of camera activity flag
    detection_by_tide = analysis_df.groupby('tidal_state').agg(
        total_observations=('DateTime', 'count'),
        detections=('is_animal_detection', 'sum'),        # FIXED: actual animal detections
        detection_rate=('is_animal_detection', 'mean')    # FIXED: animal detection rate
    ).round(4)
    
    print("\n--- Detection Rates by Tidal State (Excluding Unknowns) ---")
    print(detection_by_tide)
    
    if detection_by_tide['detections'].sum() > 20:
        # FIXED: Use animal detection flag for chi-square test
        contingency = pd.crosstab(analysis_df['is_animal_detection'], analysis_df['tidal_state'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        print(f"\nChi-square test for independence: p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("Result: Significant relationship between tidal state and detection rate!")
        else:
            print("Result: No significant relationship found.")
    
    # The rest of the function for phase analysis
    df['tidal_height_normalized'] = (df['Depth'] - df['Depth'].min()) / (df['Depth'].max() - df['Depth'].min())
    
    from scipy.signal import find_peaks
    window = 25
    df['depth_smoothed'] = df['Depth'].rolling(window=window, center=True).mean()
    
    peaks, _ = find_peaks(df['depth_smoothed'].fillna(0), distance=20)
    troughs, _ = find_peaks(-df['depth_smoothed'].fillna(0), distance=20)
    
    print(f"\nIdentified {len(peaks)} high tides and {len(troughs)} low tides in the dataset")
    
    df['tidal_phase'] = np.arcsin(2 * df['tidal_height_normalized'] - 1) / np.pi + 0.5
    
    phase_bins = np.linspace(0, 1, 13)
    phase_labels = [f"{i/12:.2f}-{(i+1)/12:.2f}" for i in range(12)]
    df['phase_bin'] = pd.cut(df['tidal_phase'], bins=phase_bins, labels=phase_labels, include_lowest=True)
    
    # FIXED: Use animal detection flag instead of camera activity flag
    phase_detection = df.groupby('phase_bin', observed=True).agg(
        observations=('DateTime', 'count'),
        detections=('is_animal_detection', 'sum'),        # FIXED: actual animal detections
        detection_rate=('is_animal_detection', 'mean')    # FIXED: animal detection rate
    )
    
    print("\n--- Detection Rates by Tidal Phase ---")
    print("(0 = Low tide, 0.5 = High tide, 1 = Next low tide)")
    print(phase_detection[['detections', 'detection_rate']].round(4))
    
    if not phase_detection.empty and phase_detection['detections'].sum() > 0:
        peak_phase = phase_detection['detection_rate'].idxmax()
        peak_rate = phase_detection['detection_rate'].max()
        print(f"\nPeak detection rate ({peak_rate:.1%}) occurs at tidal phase {peak_phase}")
    
    return df, detection_by_tide, phase_detection

def analyze_species_tide_preferences(combined_df, top_n=10):
    """
    Analyzes tidal preferences, excluding 'Unknown' and NaN states.
    FIXED: Now uses proper animal detection logic.
    """
    if 'tidal_state' not in combined_df.columns:
        print("\nCannot analyze species tide preferences without tidal state data.")
        return None
    
    # FIXED: Use proper animal detection logic
    detections_df = combined_df[
        combined_df['has_camera_data'] & 
        combined_df['Species'].notna() &
        (combined_df['Notes'] != 'No animals detected') &
        (combined_df['tidal_state'] != 'Unknown') &
        (combined_df['tidal_state'] != 'nan') &  # Remove string 'nan'
        (combined_df['tidal_state'].notna()) &   # Remove actual NaN
        (~combined_df['tidal_state'].isna())     # Additional NaN check
    ].copy()
    
    if detections_df.empty:
        print("\nNo animal detections with known tidal states to analyze.")
        return None
    
    print(f"\n--- Top {top_n} Species Tidal Preferences (Excluding Unknowns and NaN) ---")
    
    top_species = detections_df['Species'].value_counts().head(top_n).index
    
    species_tide_table = pd.crosstab(
        detections_df[detections_df['Species'].isin(top_species)]['Species'],
        detections_df[detections_df['Species'].isin(top_species)]['tidal_state'],
        normalize='index'
    ) * 100
    
    # --- ADDITIONAL FIX: Remove any remaining 'nan' or 'Unknown' columns ---
    cols_to_remove = ['nan', 'Unknown']
    for col in cols_to_remove:
        if col in species_tide_table.columns:
            species_tide_table = species_tide_table.drop(columns=[col])
    # --- END ADDITIONAL FIX ---
    
    print("\nPercentage of detections by tidal state for each species:")
    print(species_tide_table.round(1))
    
    return species_tide_table