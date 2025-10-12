# data_combiner.py

import pandas as pd
import numpy as np

def combine_data(camera_df, water_df, max_interp_hours=0.25):  # CHANGED: Default from 1 to 0.25 hours (15 minutes)
    """
    Combines camera and water data by creating a unified timeline,
    interpolating the water data, and then merging it with the camera observations.
    FIXED: Properly identifies camera observations including 'No_Animals_Detected' entries.
    UPDATED: Default interpolation limit changed to 15 minutes (0.25 hours).
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

    # IMPROVEMENT: Show camera detection breakdown  
    if 'Species' in camera_df.columns:
        # Calculate from the original camera_df before merging
        total_camera_obs = len(camera_df)
        animals_detected = len(camera_df[camera_df['Species'].notna()])
        no_animals_detected = total_camera_obs - animals_detected  # Calculate by subtraction
        
        print(f"\nCamera detection breakdown:")
        print(f"   • Total camera observations: {total_camera_obs:,}")
        print(f"   • No animals detected: {no_animals_detected:,}")
        print(f"   • Animals detected: {animals_detected:,}")
        print(f"   • Actual detection rate: {(animals_detected / total_camera_obs) * 100:.1f}%")

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
    timeline_df.index.name = 'DateTime'

    interpolated_water = water_indexed.reindex(timeline_df.index)

    if numeric_water_cols and max_interp_hours > 0:
        limit_minutes = int(max_interp_hours * 60)  # Convert hours to minutes
        interpolated_water[numeric_water_cols] = interpolated_water[numeric_water_cols].interpolate(
            method='time', limit=limit_minutes, limit_direction='both'
        )
        print(f"Interpolation performed with a time limit of {limit_minutes} minutes.")
    
    interpolated_water = interpolated_water.reset_index()

    # --- 3. Final Merge ---
    combined_df = pd.merge(
        camera_df,
        interpolated_water,
        on='DateTime',
        how='outer'
    ).sort_values('DateTime').reset_index(drop=True)

    # --- 4. Final Processing ---
    # FIXED: 'has_camera_data' should be True for ALL camera observations
    # This includes both animal detections AND no-animal observations
    # We identify camera observations by checking if they came from the camera_df
    
    # Method 1: Check if any camera-specific columns have data
    camera_indicator_cols = ['Count', 'Notes']  # These exist in all camera observations
    available_camera_cols = [col for col in camera_indicator_cols if col in combined_df.columns]
    
    if available_camera_cols:
        combined_df['has_camera_data'] = combined_df[available_camera_cols].notna().any(axis=1)
    else:
        # Fallback: if camera indicator columns aren't available, use Species
        combined_df['has_camera_data'] = (
            combined_df['Species'].notna() | 
            (combined_df['Notes'] == 'No animals detected')
        )
    
    # IMPROVEMENT: Add explicit animal detection flag
    combined_df['animal_detected'] = (
        combined_df['has_camera_data'] & 
        (combined_df['Species'] != 'No_Animals_Detected') &
        combined_df['Species'].notna()
    )

    print(f"\n--- Combination and Interpolation Complete ---")
    print(f"Final dataset has {len(combined_df)} rows.")
    print(f"Unique timestamps: {combined_df['DateTime'].nunique()}")
    print(f"Rows with camera data: {combined_df['has_camera_data'].sum()}")
    print(f"Camera observations with animals: {combined_df['animal_detected'].sum()}")
    print(f"Camera observations with no animals: {(combined_df['has_camera_data'] & ~combined_df['animal_detected']).sum()}")
    
    if combined_df['has_camera_data'].sum() > 0:
        true_detection_rate = combined_df['animal_detected'].sum() / combined_df['has_camera_data'].sum()
        print(f"True detection rate: {true_detection_rate:.1%}")
    
    print(f"Null values in 'Depth' after process: {combined_df['Depth'].isnull().sum()}")
    print(f"Null values in 'Depth_Inside' after process: {combined_df['Depth_Inside'].isnull().sum()}")

    return combined_df