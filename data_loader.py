# data_loader.py

import pandas as pd
import numpy as np

def load_and_prepare_camera_data(file_id):
    """
    Loads and processes camera trap data, handling multiple species entries
    per row and ensuring all related columns are standardized.
    This version properly expands rows with multiple species into separate rows.
    """
    print("--- Loading and Preparing Camera Data ---")
    df = pd.read_csv(file_id)
    
    print(f"Raw camera data shape: {df.shape}")

    # Helper function to correctly process multi-species columns
    def process_and_combine_species(df_to_process):
        all_species_dfs = []
        
        # Identify base columns (not species-specific)
        species_pattern_cols = []
        for i in range(1, 10):  # Check up to Species 9 to be safe
            species_pattern_cols.extend([
                f'Species {i}', f'Species {i} Count', 
                f'Species {i} Distance', f'Species {i} Activity'
            ])
        
        base_cols = [col for col in df_to_process.columns if col not in species_pattern_cols]
        
        print(f"Base columns (non-species-specific): {len(base_cols)}")
        
        # Process each species column
        species_found = 0
        for i in range(1, 10):  # Check up to Species 9
            species_col = f'Species {i}'
            if species_col in df_to_process.columns:
                species_found += 1
                # Get all related columns for this species number
                current_species_cols = [
                    species_col, 
                    f'Species {i} Count', 
                    f'Species {i} Distance', 
                    f'Species {i} Activity'
                ]
                
                # Select base columns plus species-specific columns that exist
                cols_to_select = base_cols + [col for col in current_species_cols if col in df_to_process.columns]
                
                # Create a temporary dataframe for this species
                temp_df = df_to_process[cols_to_select].copy()
                
                # Only keep rows where this species is not null
                temp_df = temp_df[temp_df[species_col].notna()].copy()
                
                if not temp_df.empty:
                    # Rename the species-specific columns to generic names
                    rename_dict = {
                        f'Species {i}': 'Species',
                        f'Species {i} Count': 'Count',
                        f'Species {i} Distance': 'Distance',
                        f'Species {i} Activity': 'Activity'
                    }
                    temp_df.rename(columns=rename_dict, inplace=True)
                    
                    # Add to our list
                    all_species_dfs.append(temp_df)
                    print(f"Species {i}: {len(temp_df)} non-null entries")
        
        print(f"Found {species_found} species columns in the data")
        
        if not all_species_dfs:
            print("WARNING: No species data found!")
            return pd.DataFrame()

        # Concatenate all species dataframes
        final_df = pd.concat(all_species_dfs, ignore_index=True)
        
        # Ensure Count column exists and is numeric
        if 'Count' not in final_df.columns:
            final_df['Count'] = 1
        else:
            final_df['Count'] = pd.to_numeric(final_df['Count'], errors='coerce').fillna(1).astype(int)
        
        print(f"Total rows after expanding multi-species entries: {len(final_df)}")
        
        return final_df

    # Process the dataframe to expand multi-species rows
    processed_df = process_and_combine_species(df)
    
    if processed_df.empty:
        print("ERROR: Processed dataframe is empty!")
        return processed_df
    
    # Standardize species names to fix capitalization inconsistencies
    print("\n--- Standardizing Species Names ---")
    if 'Species' in processed_df.columns:
        # First, let's see what we're working with
        original_unique = processed_df['Species'].nunique()
        
        # Standardize the species names:
        # 1. Strip leading/trailing whitespace
        # 2. Replace multiple spaces with single space
        # 3. Handle case normalization carefully
        processed_df['Species'] = processed_df['Species'].str.strip()
        processed_df['Species'] = processed_df['Species'].str.replace(r'\s+', ' ', regex=True)
        
        # Create a standardized mapping based on lowercase comparison
        # This will help us identify and merge duplicates
        species_lower_map = {}
        for species in processed_df['Species'].dropna().unique():
            species_lower = species.lower()
            if species_lower not in species_lower_map:
                # First occurrence sets the standard
                species_lower_map[species_lower] = species
            else:
                # If we already have this species (in lowercase), use the existing standard
                print(f"  Found duplicate: '{species}' will be merged with '{species_lower_map[species_lower]}'")
        
        # Apply the standardization
        processed_df['Species'] = processed_df['Species'].str.lower().map(species_lower_map).fillna(processed_df['Species'])
        
        # Now apply proper capitalization for scientific names
        # Most scientific names should have first letter capitalized for genus, lowercase for species
        def standardize_scientific_name(name):
            if pd.isna(name):
                return name
            
            # Special cases that should remain as-is
            special_cases = {
                'unknown': 'Unknown',
                'no_animals_detected': 'No_Animals_Detected',
                'anatidae': 'Anatidae',  # Family names stay capitalized
                'phalacrocoracidae': 'Phalacrocoracidae',
                'corvidae': 'Corvidae'
            }
            
            name_lower = name.lower()
            if name_lower in special_cases:
                return special_cases[name_lower]
            
            # For binomial names (genus + species)
            parts = name.split()
            if len(parts) == 2:
                # Capitalize genus, lowercase species epithet
                return f"{parts[0].capitalize()} {parts[1].lower()}"
            elif len(parts) == 1:
                # Single word - likely genus or common name
                return parts[0].capitalize()
            else:
                # For anything else, just title case
                return name.title()
        
        processed_df['Species'] = processed_df['Species'].apply(standardize_scientific_name)
        
        new_unique = processed_df['Species'].nunique()
        print(f"Species names before standardization: {original_unique}")
        print(f"Species names after standardization: {new_unique}")
        print(f"Merged {original_unique - new_unique} duplicate species names")
        
        # Show the standardized species list
        species_counts = processed_df['Species'].value_counts()
        print("\nTop species after standardization:")
        print(species_counts.head(10))
    
    # Convert DateTime to proper format
    processed_df['DateTime'] = pd.to_datetime(processed_df['DateTime'])
    
    # Show some statistics about the expansion
    print("\n--- Multi-Species Expansion Statistics ---")
    dup_times = processed_df[processed_df.duplicated(subset=['DateTime'], keep=False)]
    unique_dup_times = dup_times['DateTime'].nunique()
    print(f"Timestamps with multiple species: {unique_dup_times}")
    print(f"Total rows with shared timestamps: {len(dup_times)}")
    
    if not dup_times.empty:
        # Show an example
        sample_time = dup_times['DateTime'].iloc[0]
        sample_data = processed_df[processed_df['DateTime'] == sample_time][['DateTime', 'Species', 'Count']]
        print(f"\nExample - Multiple species at {sample_time}:")
        print(sample_data)
    
    print("--- Camera Data Loaded and Processed ---")
    return processed_df


def load_and_prepare_water_data(file_id):
    """
    Loads water sensor data and standardizes column names.
    This updated version ensures all key measurement columns are converted
    to a numeric type for successful interpolation and treats zeros as missing values.
    """
    print("\n--- Loading Water Quality Data ---")
    df = pd.read_csv(file_id)

    print(f"Original water data columns: {list(df.columns)}")

    # Combine date and time columns into a single DateTime object
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.drop(columns=['Date', 'Time'], inplace=True)

    # --- MODIFICATION START ---

    # Standardize column names. ADDED 'Tidal Level Inside Tidegate [m]'
    rename_map = {
        'Tidal Level Outside Tidegate [m]': 'Depth',
        'Tidal Level Inside Tidegate [m]': 'Depth_Inside',
        'Air Temp [C]': 'Air_Temp_C',
        'Gate Opening MTR [Degrees]': 'Gate_Opening_MTR_Deg',
        'Gate Opening Top Hinge [Degrees]': 'Gate_Opening_Top_Hinge_Deg',
        'Wind Speed [km/h]': 'Wind_Speed_km_h',
    }
    
    # Rename only the columns that are present in the dataframe
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Define all columns that should be numeric for analysis and interpolation
    numeric_cols = [
        'Depth', 'Depth_Inside', 'Air_Temp_C', 'Gate_Opening_MTR_Deg',
        'Gate_Opening_Top_Hinge_Deg', 'Wind_Speed_km_h'
    ]

    # Forcefully convert these columns to numeric type.
    # Any values that cannot be converted (e.g., '---', 'Error') will become NaN.
    print("\nEnsuring data columns are numeric for interpolation...")
    for col in numeric_cols:
        if col in df.columns:
            original_nulls = df[col].isnull().sum()
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # CRITICAL: Replace zeros with NaN for tidal level columns
            if col in ['Depth', 'Depth_Inside']:
                zero_count = (df[col] == 0).sum()
                df.loc[df[col] == 0, col] = np.nan
                print(f"  -> Replaced {zero_count} zeros with NaN in '{col}'")
            
            new_nulls = df[col].isnull().sum()
            if (new_nulls > original_nulls):
                print(f"  -> Converted {new_nulls - original_nulls} non-numeric values to NaN in '{col}'")

    # --- MODIFICATION END ---
    
    # Set DateTime as the index, which is required for the merging logic
    df.set_index('DateTime', inplace=True)
    
    # Print summary statistics for the depth columns to verify
    print("\n--- Depth Column Statistics After Processing ---")
    if 'Depth' in df.columns:
        print(f"Depth (Outside): Non-null values: {df['Depth'].notna().sum()}, "
              f"Min: {df['Depth'].min():.2f}, Max: {df['Depth'].max():.2f}")
    if 'Depth_Inside' in df.columns:
        print(f"Depth_Inside: Non-null values: {df['Depth_Inside'].notna().sum()}, "
              f"Min: {df['Depth_Inside'].min():.2f}, Max: {df['Depth_Inside'].max():.2f}")
    
    print(f"\nProcessed water data shape: {df.shape}")
    print("--- Water Quality Data Loaded and Standardized ---")
    return df