# data_combiner.py

import pandas as pd
import numpy as np

def combine_data(camera_df, water_df, max_interp_hours=3):
    """
    Combines camera and water data by creating a unified timeline,
    interpolating the water data, and then merging it with the camera observations.
    This version handles duplicate timestamps and preserves the index name.
    """
    print("\n--- Combining Data (Robust Method) ---")

    # --- 1. Preparation ---
    if 'DateTime' not in water_df.columns:
        water_df = water_df.reset_index()

    camera_df['DateTime'] = pd.to_datetime(camera_df['DateTime'])
    water_df['DateTime'] = pd.to_datetime(water_df['DateTime'])
    
    print(f"Camera data shape: {camera_df.shape}")
    print(f"Initial water data shape: {water_df.shape}")
    print(f"Water data columns available: {list(water_df.columns)}")

    # --- 2. Create Unified Timeline & Interpolate Water Data ---
    print("Handling potential duplicate timestamps in water data by averaging...")
    water_indexed = water_df.groupby('DateTime').mean(numeric_only=True)
    print(f"Water data shape after removing duplicates: {water_indexed.shape}")

    numeric_water_cols = water_indexed.select_dtypes(include=np.number).columns.tolist()
    
    print(f"Found the following numeric columns for interpolation: {numeric_water_cols}")
    
    if 'Depth' in numeric_water_cols:
        print("  ✅ 'Depth' correctly identified as numeric.")
    if 'Depth_Inside' in numeric_water_cols:
        print("  ✅ 'Depth_Inside' correctly identified as numeric.")

    all_timestamps = pd.concat([
        camera_df['DateTime'],
        water_indexed.index.to_series()
    ]).unique()
    
    timeline_df = pd.DataFrame(index=pd.to_datetime(all_timestamps)).sort_index()

    # --- FIX START ---
    # The KeyError occurs because the reindex operation can cause the index to lose its name.
    # We explicitly set the index name here to ensure that when we call reset_index() later,
    # it creates a column named 'DateTime' instead of 'index'.
    timeline_df.index.name = 'DateTime'
    # --- FIX END ---

    interpolated_water = water_indexed.reindex(timeline_df.index)

    if numeric_water_cols and max_interp_hours > 0:
        limit_minutes = int(max_interp_hours * 60)
        interpolated_water[numeric_water_cols] = interpolated_water[numeric_water_cols].interpolate(
            method='time', limit=limit_minutes, limit_direction='both'
        )
        print(f"Interpolation performed with a time limit of {limit_minutes} minutes.")
    
    interpolated_water = interpolated_water.reset_index()

    # --- 3. Final Merge ---
    # This merge will now succeed because interpolated_water is guaranteed to have a 'DateTime' column.
    combined_df = pd.merge(
        camera_df,
        interpolated_water,
        on='DateTime',
        how='outer'
    ).sort_values('DateTime').reset_index(drop=True)

    # --- 4. Final Processing ---
    combined_df['has_camera_data'] = combined_df['Species'].notna()

    print(f"\n--- Combination and Interpolation Complete ---")
    print(f"Final dataset has {len(combined_df)} rows.")
    print(f"Unique timestamps: {combined_df['DateTime'].nunique()}")
    print(f"Rows with camera data: {combined_df['has_camera_data'].sum()}")
    print(f"Null values in 'Depth' after process: {combined_df['Depth'].isnull().sum()}")
    print(f"Null values in 'Depth_Inside' after process: {combined_df['Depth_Inside'].isnull().sum()}")


    return combined_df