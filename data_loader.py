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
    """
    print("\n--- Loading Water Quality Data ---")
    df = pd.read_csv(file_id)

    # Print columns for debugging
    print(f"Water data columns: {list(df.columns)}")

    # Combine date and time columns
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.drop(columns=['Date', 'Time'], inplace=True)

    # Rename columns to standardized names
    rename_map = {
        'Tidal Level Outside Tidegate [m]': 'Depth',
        'Air Temp [C]': 'Air_Temp_C',
        'Gate Opening MTR [Degrees]': 'Gate_Opening_MTR_Deg',
        'Gate Opening Top Hinge [Degrees]': 'Gate_Opening_Top_Hinge_Deg',
        'Wind Speed [km/h]': 'Wind_Speed_km_h',
        # Add any other column mappings here as needed
    }
    
    # Only rename columns that actually exist
    rename_map_filtered = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=rename_map_filtered, inplace=True)
    
    # Set DateTime as index for merging
    df.set_index('DateTime', inplace=True)
    
    print(f"Water data shape: {df.shape}")
    print("--- Water Quality Data Loaded and Standardized ---")
    return df