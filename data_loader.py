# data_loader.py

import pandas as pd
import numpy as np

def load_and_prepare_camera_data(file_id):
    """
    Loads and processes camera trap data, handling multiple species entries
    per row and ensuring all related columns are standardized.
    """
    print("--- Loading and Preparing Camera Data ---")
    df = pd.read_csv(file_id)

    # Helper function to correctly process multi-species columns
    def process_and_combine_species(df_to_process):
        all_species_dfs = []
        base_cols = [col for col in df_to_process.columns if not any(s in col for s in ['Species 1', 'Species 2', 'Species 3'])]

        for i in range(1, 4):
            species_col = f'Species {i}'
            if species_col in df_to_process.columns:
                current_species_cols = [species_col, f'Species {i} Count', f'Species {i} Distance', f'Species {i} Activity']
                cols_to_select = base_cols + [col for col in current_species_cols if col in df_to_process.columns]
                
                temp_df = df_to_process[cols_to_select].copy()
                temp_df.dropna(subset=[species_col], inplace=True)

                if not temp_df.empty:
                    rename_dict = {
                        f'Species {i}': 'Species',
                        f'Species {i} Count': 'Count',
                        f'Species {i} Distance': 'Distance',
                        f'Species {i} Activity': 'Activity'
                    }
                    temp_df.rename(columns=rename_dict, inplace=True)
                    all_species_dfs.append(temp_df)
        
        if not all_species_dfs:
            return pd.DataFrame()

        final_df = pd.concat(all_species_dfs, ignore_index=True)
        if 'Count' not in final_df.columns:
            final_df['Count'] = 1
        else:
            final_df['Count'] = pd.to_numeric(final_df['Count'], errors='coerce').fillna(1).astype(int)
        
        return final_df

    processed_df = process_and_combine_species(df)
    processed_df['DateTime'] = pd.to_datetime(processed_df['DateTime'])
    print("--- Camera Data Loaded and Processed ---")
    return processed_df


# In data_loader.py, replace the water data function with this DEBUGGING version:

def load_and_prepare_water_data(file_id):
    """
    Loads water sensor data in DEBUG MODE to print its column names.
    """
    print("--- Loading Water Quality Data (DEBUG MODE) ---")
    df = pd.read_csv(file_id)

    # --- DEBUGGING STEP: Print the exact column names from the file ---
    print("\n[DEBUG] Columns loaded from the sensor data file:")
    print(list(df.columns))
    print("-" * 30)
    # --- END DEBUGGING ---

    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.drop(columns=['Date', 'Time'], inplace=True)

    rename_map = {
        'Tidal Level Outside Tidegate [m]': 'Depth',
        'Air Temp [C]': 'Air_Temp_C',
        'Gate Opening MTR [Degrees]': 'Gate_Opening_MTR_Deg',
        'Gate Opening Top Hinge [Degrees]': 'Gate_Opening_Top_Hinge_Deg'
    }
    df.rename(columns=rename_map, inplace=True)
            
    df.set_index('DateTime', inplace=True)
    print("--- Water Quality Data Loaded and Standardized ---")
    return df