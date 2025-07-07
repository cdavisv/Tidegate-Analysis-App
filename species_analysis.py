# species_analysis.py

import pandas as pd

def analyze_species_diversity(camera_df):
    """
    Analyzes the camera dataframe to calculate species diversity metrics.

    Args:
        camera_df (pd.DataFrame): The pre-processed camera data.

    Returns:
        tuple: A tuple containing the species_summary DataFrame and the filtered species_df.
    """
    if camera_df.empty:
        print("Camera data is empty, skipping species diversity analysis.")
        return pd.DataFrame(), pd.DataFrame()

    species_df = camera_df[camera_df['Species'] != 'No_Animals_Detected'].copy()

    if species_df.empty:
        print("No animal detections found in camera data.")
        return pd.DataFrame(), species_df

    print("\n=== SPECIES DIVERSITY ANALYSIS ===")
    print(f"Total animal detections: {len(species_df):,}")
    print(f"Unique species detected: {species_df['Species'].nunique()}")

    species_summary = species_df.groupby('Species').agg(
        Total_Count=('Count', 'sum'),
        Detection_Events=('DateTime', 'count')
    ).sort_values('Total_Count', ascending=False)

    print("\nTop 15 species by total individual count:")
    print(species_summary.head(15))

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