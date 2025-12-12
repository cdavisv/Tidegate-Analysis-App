"""
Species diversity and preference analysis module.

This module focuses on analyzing species diversity using standardized species
names and properly aggregated detection counts.

Key outputs include:
- Total individual counts per species
- Detection event counts
- Validation against duplicate species naming
- Optional environmental preference summaries

All analyses explicitly exclude no-animal observations to ensure biological
interpretability.
"""

import pandas as pd

def analyze_species_diversity(camera_df):
    """
    Analyzes the camera dataframe to calculate species diversity metrics.
    This version ensures proper aggregation of standardized species names.

    Args:
        camera_df (pd.DataFrame): The pre-processed camera data.

    Returns:
        tuple: A tuple containing the species_summary DataFrame and the filtered species_df.
    """
    if camera_df.empty:
        print("Camera data is empty, skipping species diversity analysis.")
        return pd.DataFrame(), pd.DataFrame()

    # FIXED: Only include actual animal detections, exclude 'No_Animals_Detected' records
    species_df = camera_df[
        (camera_df['Species'].notna()) & 
        (camera_df['Species'] != 'No_Animals_Detected') &
        (camera_df['Notes'] != 'No animals detected')
    ].copy()

    if species_df.empty:
        print("No animal detections found in camera data.")
        return pd.DataFrame(), species_df

    print("\n=== SPECIES DIVERSITY ANALYSIS ===")
    # FIXED: Report the correct number - actual animal detections, not total records
    print(f"Total animal detections: {len(species_df):,}")
    print(f"Unique species detected: {species_df['Species'].nunique()}")

    # IMPORTANT: Ensure Count is numeric before aggregation
    if 'Count' in species_df.columns:
        species_df['Count'] = pd.to_numeric(species_df['Count'], errors='coerce').fillna(1)
    else:
        species_df['Count'] = 1

    # Group by species and aggregate properly
    species_summary = species_df.groupby('Species', as_index=True).agg(
        Total_Count=('Count', 'sum'),
        Detection_Events=('DateTime', 'count')
    ).sort_values('Total_Count', ascending=False)

    # Convert the index to string to ensure consistent display
    species_summary.index = species_summary.index.astype(str)

    print("\nTop 15 species by total individual count:")
    print(species_summary.head(15).to_string())

    # Additional check for any remaining duplicates
    print("\n--- Checking for potential remaining duplicates ---")
    species_list = species_summary.index.tolist()
    found_duplicates = False
    
    for i, sp1 in enumerate(species_list[:20]):  # Check top 20 species
        for j, sp2 in enumerate(species_list[:20]):
            if i < j and sp1.lower() == sp2.lower():
                print(f"WARNING: Found similar species names: '{sp1}' and '{sp2}'")
                found_duplicates = True
    
    if not found_duplicates:
        print("No duplicate species names found in top 20 species.")

    return species_summary, species_df


def analyze_species_preferences(species_df):
    """
    Analyzes the environmental preferences for the most detected species.

    Args:
        species_df (pd.DataFrame): DataFrame containing only animal detections,
                                   merged with environmental data.
    """
    if species_df.empty:
        return

    print("\n\n=== SPECIES ENVIRONMENTAL PREFERENCES ===")

    top_species = species_df['Species'].value_counts().nlargest(8).index.tolist()
    
    # Filter the dataframe to only include top species for efficiency
    top_species_df = species_df[species_df['Species'].isin(top_species)].copy()

    # --- Gate Preference ---
    if 'gate_category' in top_species_df.columns:
        print("\n--- Detections by Gate State ---")
        gate_preference = pd.crosstab(top_species_df['Species'], top_species_df['gate_category'])
        print(gate_preference)

    # --- Tidal Preference ---
    if 'tide_level' in top_species_df.columns:
        print("\n--- Detections by Tidal Level ---")
        tidal_preference = pd.crosstab(top_species_df['Species'], top_species_df['tide_level'])
        print(tidal_preference)