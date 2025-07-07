# data_combiner.py

import pandas as pd

def combine_data(camera_df, water_df, method='linear', limit_direction='both', water_interpolation_tolerance='30 minutes'):
    """
    Combines camera and water quality data using a robust merge and interpolation process.

    Args:
        camera_df (pd.DataFrame): DataFrame with camera data and 'DateTime' column.
        water_df (pd.DataFrame): DataFrame with water data and a 'DateTime' index.
        method (str): Interpolation method ('linear', 'time', etc.).
        limit_direction (str): Direction for filling NaNs.
        water_interpolation_tolerance (str): Max time gap for interpolation.

    Returns:
        pd.DataFrame: A unified time series DataFrame.
    """
    print("\n--- Combining Data using Robust Merge and Smart Interpolation ---")

    cam = camera_df.copy()
    wat = water_df.copy()

    # Ensure DateTime types are consistent
    cam['DateTime'] = pd.to_datetime(cam['DateTime'])
    if not isinstance(wat.index, pd.DatetimeIndex):
        if 'DateTime' in wat.columns:
            wat['DateTime'] = pd.to_datetime(wat['DateTime'])
            wat.set_index('DateTime', inplace=True)
        else:
            raise ValueError("Water DataFrame must have a 'DateTime' index or column.")

    # --- CORRECTED LINE ---
    # Create a pandas Index from the camera's DateTime Series before using .union()
    camera_index = pd.Index(cam['DateTime'])
    combined_index = camera_index.union(wat.index).unique().sort_values()
    # --- END CORRECTION ---

    # Reindex water data to the full timeline to create gaps, then interpolate
    water_timeline_df = wat.reindex(combined_index).interpolate(
        method=method,
        limit_direction=limit_direction,
        limit_area='inside'
    ).reset_index().rename(columns={'index': 'DateTime'})

    # Merge the interpolated water timeline with the camera data
    final_df = pd.merge(water_timeline_df, cam, on='DateTime', how='left')
    final_df['has_camera_data'] = final_df['Species'].notna()

    print(f"--- Combination Complete. Final dataset has {len(final_df)} rows. ---")
    return final_df