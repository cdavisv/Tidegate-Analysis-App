# tide_cycle_analysis.py

import pandas as pd
import numpy as np
from scipy import stats

def analyze_tide_cycle_detections(combined_df):
    """
    Analyzes animal detection patterns across the tidal cycle.
    Works with or without gate data.
    """
    print("\n\n=== TIDE CYCLE DETECTION ANALYSIS ===")
    
    if 'Depth' not in combined_df.columns:
        print("No tidal depth data available. Skipping tide cycle analysis.")
        return None
    
    # Create a copy for analysis
    df = combined_df.copy()
    
    # Calculate tidal metrics
    df['tidal_change_m_hr'] = df['Depth'].diff() * 2  # Assuming 30-minute intervals
    
    # Define tidal states with more granularity
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
    
    # Calculate detection rates by tidal state
    detection_by_tide = df.groupby('tidal_state').agg(
        total_observations=('DateTime', 'count'),
        detections=('has_camera_data', 'sum'),
        detection_rate=('has_camera_data', 'mean')
    ).round(4)
    
    print("\n--- Detection Rates by Tidal State ---")
    print(detection_by_tide)
    
    # Statistical test
    if detection_by_tide['detections'].sum() > 20:  # Need enough data for chi-square
        # Create contingency table
        contingency = pd.crosstab(df['has_camera_data'], df['tidal_state'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        print(f"\nChi-square test for independence: p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("Result: Significant relationship between tidal state and detection rate!")
        else:
            print("Result: No significant relationship found.")
    
    # Analyze detection timing within tidal cycles
    # Create normalized tidal position (0 = low tide, 0.5 = high tide, 1 = next low tide)
    df['tidal_height_normalized'] = (df['Depth'] - df['Depth'].min()) / (df['Depth'].max() - df['Depth'].min())
    
    # Find local minima and maxima to identify tidal cycles
    from scipy.signal import find_peaks
    
    # Smooth the data for better peak detection
    window = 25  # ~12.5 hours if data is every 30 minutes
    df['depth_smoothed'] = df['Depth'].rolling(window=window, center=True).mean()
    
    # Find peaks (high tides) and troughs (low tides)
    peaks, _ = find_peaks(df['depth_smoothed'].fillna(0), distance=20)
    troughs, _ = find_peaks(-df['depth_smoothed'].fillna(0), distance=20)
    
    print(f"\nIdentified {len(peaks)} high tides and {len(troughs)} low tides in the dataset")
    
    # Create tidal phase (0-1 scale where 0 and 1 are low tide, 0.5 is high tide)
    df['tidal_phase'] = np.nan
    
    # Simple approach: use sine approximation based on normalized height
    # This assumes a sinusoidal tide pattern
    df['tidal_phase'] = np.arcsin(2 * df['tidal_height_normalized'] - 1) / np.pi + 0.5
    
    # Bin the tidal phases for analysis
    phase_bins = np.linspace(0, 1, 13)  # 12 bins across the tidal cycle
    phase_labels = [f"{i/12:.2f}-{(i+1)/12:.2f}" for i in range(12)]
    df['phase_bin'] = pd.cut(df['tidal_phase'], bins=phase_bins, labels=phase_labels, include_lowest=True)
    
    # Calculate detection rates by tidal phase
    phase_detection = df.groupby('phase_bin', observed=True).agg(
        observations=('DateTime', 'count'),
        detections=('has_camera_data', 'sum'),
        detection_rate=('has_camera_data', 'mean')
    )
    
    print("\n--- Detection Rates by Tidal Phase ---")
    print("(0 = Low tide, 0.5 = High tide, 1 = Next low tide)")
    print(phase_detection[['detections', 'detection_rate']].round(4))
    
    # Find peak detection phases
    if not phase_detection.empty and phase_detection['detections'].sum() > 0:
        peak_phase = phase_detection['detection_rate'].idxmax()
        peak_rate = phase_detection['detection_rate'].max()
        print(f"\nPeak detection rate ({peak_rate:.1%}) occurs at tidal phase {peak_phase}")
    
    return df, detection_by_tide, phase_detection


def analyze_species_tide_preferences(combined_df, top_n=10):
    """
    Analyzes tidal preferences for the most common species.
    """
    if 'tidal_state' not in combined_df.columns:
        print("\nCannot analyze species tide preferences without tidal state data.")
        return None
    
    # Filter to only animal detections
    detections_df = combined_df[combined_df['has_camera_data'] & (combined_df['Species'] != 'No_Animals_Detected')].copy()
    
    if detections_df.empty:
        print("\nNo animal detections to analyze.")
        return None
    
    print(f"\n--- Top {top_n} Species Tidal Preferences ---")
    
    # Get top species
    top_species = detections_df['Species'].value_counts().head(top_n).index
    
    # Create crosstab
    species_tide_table = pd.crosstab(
        detections_df[detections_df['Species'].isin(top_species)]['Species'],
        detections_df[detections_df['Species'].isin(top_species)]['tidal_state'],
        normalize='index'
    ) * 100
    
    print("\nPercentage of detections by tidal state for each species:")
    print(species_tide_table.round(1))
    
    # Find species with strongest tidal preferences
    # Calculate coefficient of variation for each species
    species_preferences = {}
    for species in species_tide_table.index:
        cv = species_tide_table.loc[species].std() / species_tide_table.loc[species].mean()
        species_preferences[species] = cv
    
    sorted_prefs = sorted(species_preferences.items(), key=lambda x: x[1], reverse=True)
    
    print("\n--- Species Tidal Preference Strength ---")
    print("(Higher values indicate stronger tidal preferences)")
    for species, cv in sorted_prefs[:5]:
        print(f"{species}: {cv:.3f}")
        preferred_state = species_tide_table.loc[species].idxmax()
        print(f"  -> Prefers: {preferred_state} ({species_tide_table.loc[species, preferred_state]:.1f}% of detections)")
    
    return species_tide_table