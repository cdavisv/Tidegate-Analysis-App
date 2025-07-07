# data_loader.py

import pandas as pd
import numpy as np

def load_and_prepare_camera_data(filepath):
    """
    Loads the camera trap data from a CSV file and processes it to handle
    multiple species entries per row, transforming it into a long format.
    """
    df = pd.read_csv(filepath)

    def process_and_combine_species(df_to_process):
        species_related_cols = [col for col in df_to_process.columns if 'Species' in col]
        base_cols = [col for col in df_to_process.columns if col not in species_related_cols]
        all_species_dfs = []
        for i in range(1, 4):
            species_col = f'Species {i}'
            if species_col in df_to_process.columns:
                current_species_cols = [
                    species_col, f'Species {i} Count', f'Species {i} Distance', f'Species {i} Activity'
                ]
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
        if all_species_dfs:
            final_df = pd.concat(all_species_dfs, ignore_index=True)
            final_df['Count'] = pd.to_numeric(final_df.get('Count'), errors='coerce').fillna(1).astype(int)
            final_df.fillna({'Distance': 'Unknown', 'Activity': 'Unknown'}, inplace=True)
            return final_df
        return pd.DataFrame(columns=base_cols + ['Species', 'Count', 'Distance', 'Activity'])

    processed_df = process_and_combine_species(df)
    if 'Date' in processed_df.columns and 'Time' in processed_df.columns:
        processed_df['DateTime'] = pd.to_datetime(processed_df['Date'] + ' ' + processed_df['Time'], errors='coerce')
        processed_df.drop(columns=['Date', 'Time'], inplace=True)
        processed_df.dropna(subset=['DateTime'], inplace=True)
        processed_df.sort_values('DateTime', inplace=True, ignore_index=True)
    return processed_df

def load_and_prepare_water_data(file_path, file_id="Unknown", max_interpolation_gap='30 minutes'):
    """
    Loads water quality data, standardizes column names, cleans data, merges
    duplicate timestamps, and performs gap-aware interpolation.
    """
    print(f"\n--- Loading and Preparing Water Data from: {file_id} ---")
    try:
        water_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")
        return pd.DataFrame()
    if water_df.empty:
        return water_df

    rename_map = {
        'Temp C': 'Water_Temp_C', 'Temp_C': 'Water_Temp_C',
        'Water Temperature Inside TG [C]': 'Water_Temp_In_C', 'Water_Temperature_Inside_TG_C': 'Water_Temp_In_C',
        'Water Temperature Outside Tidegate [C]': 'Water_Temp_Out_C', 'Water_Temperature_Outside_Tidegate_C': 'Water_Temp_Out_C',
        'Tidal Level Inside Tidegate [m]': 'Tide_Level_In_m', 'Tidal_Level_Inside_Tidegate_m': 'Tide_Level_In_m',
        'Tidal Level Outside Tidegate [m]': 'Tide_Level_Out_m', 'Tidal_Level_Outside_Tidegate_m': 'Tide_Level_Out_m',
        'Water Level': 'Tide_Level_Verified_m', 'Verified Water Level': 'Tide_Level_Verified_m',
        'pH': 'pH', 'pH (units)': 'pH',
        'Dissolved Oxygen mgL': 'DO_mgL', 'DO (mg/L)': 'DO_mgL',
        'Salinity': 'Salinity_psu', 'Sal psu': 'Salinity_psu', 'Sal (ppt)': 'Salinity_ppt',
        'Turbidity FNU': 'Turbidity_FNU', 'Turb FNU/NTU': 'Turbidity_FNU', 'Turbidity NTU': 'Turbidity_NTU',
        'Depth m': 'Depth_m', 'cDepth m': 'cDepth_m', # Keep specific depth names for fallback logic
        'Gate Opening MTR [Degrees]': 'Gate_Opening_MTR_Deg',
        'Date Time': 'DateTime_Source', 'PST': 'DateTime_Source', 'DateTime': 'DateTime_Source'
    }
    water_df.rename(columns=rename_map, inplace=True)

    if 'DateTime_Source' in water_df.columns:
        water_df['DateTime'] = pd.to_datetime(water_df['DateTime_Source'], errors='coerce')
    elif 'Date' in water_df.columns and 'Time' in water_df.columns:
        water_df['DateTime'] = pd.to_datetime(water_df['Date'] + ' ' + water_df['Time'], errors='coerce')
    water_df.dropna(subset=['DateTime'], inplace=True)
    water_df.drop(columns=[col for col in ['Date', 'Time', 'DateTime_Source'] if col in water_df.columns], inplace=True)

    for col in water_df.columns:
        if col != 'DateTime':
            water_df[col] = pd.to_numeric(water_df[col], errors='coerce')

    # Fallback to create a generic Water_Temp_C column for consistent analysis
    if 'Water_Temp_C' not in water_df.columns:
        if 'Water_Temp_In_C' in water_df.columns:
            water_df['Water_Temp_C'] = water_df['Water_Temp_In_C']
            print("  --> Created 'Water_Temp_C' from 'Water_Temp_In_C' for standardization.")
        elif 'Water_Temp_Out_C' in water_df.columns:
            water_df['Water_Temp_C'] = water_df['Water_Temp_Out_C']
            print("  --> Created 'Water_Temp_C' from 'Water_Temp_Out_C' for standardization.")

    # --- ADDED THIS BLOCK TO FIX THE DEPTH ISSUE ---
    # Standardize depth column for consistent analysis
    if 'Depth' not in water_df.columns:
        if 'Tide_Level_Out_m' in water_df.columns:
            water_df['Depth'] = water_df['Tide_Level_Out_m']
            print("  --> Created 'Depth' column from 'Tide_Level_Out_m' for standardization.")
        elif 'Depth_m' in water_df.columns:
            water_df['Depth'] = water_df['Depth_m']
            print("  --> Created 'Depth' column from 'Depth_m' for standardization.")
        elif 'Tide_Level_Verified_m' in water_df.columns:
            water_df['Depth'] = water_df['Tide_Level_Verified_m']
            print("  --> Created 'Depth' column from 'Tide_Level_Verified_m' for standardization.")
    # --- END OF ADDED BLOCK ---

    water_df = water_df.set_index('DateTime').sort_index()
    if water_df.index.duplicated(keep=False).any():
        water_df = water_df.groupby(water_df.index).mean()

    water_df = water_df.interpolate(method='time', limit_direction='both', limit_area='inside', limit=int(pd.Timedelta(max_interpolation_gap).total_seconds() // 1800))

    return water_df