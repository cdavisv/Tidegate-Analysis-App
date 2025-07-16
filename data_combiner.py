# data_combiner.py

import pandas as pd
import numpy as np

def combine_data(camera_df, water_df, max_interp_hours=3):
    """
    Combines, aggregates, and interpolates camera and water data.
    Modified to preserve multiple species entries at the same timestamp.
    """
    print("\n--- Combining Data ---")

    # First, let's see what we're working with
    print(f"Camera data shape before merge: {camera_df.shape}")
    print(f"Water data shape before merge: {water_df.shape}")
    
    # Check for duplicate timestamps with different species
    if 'DateTime' in camera_df.columns:
        dup_times = camera_df[camera_df.duplicated(subset=['DateTime'], keep=False)]
        if not dup_times.empty:
            print(f"Found {len(dup_times)} camera records with duplicate timestamps (multiple species)")

    # Merge the dataframes
    combined_df = pd.merge(
        water_df,
        camera_df,
        left_index=True,
        right_on='DateTime',
        how='outer'
    ).sort_values(by='DateTime')

    if combined_df.empty:
        print("Combined DataFrame is empty after merge.")
        return combined_df

    # Create the 'has_camera_data' helper column BEFORE aggregation
    combined_df['has_camera_data'] = combined_df['Species'].notna()

    # --- MODIFIED AGGREGATION APPROACH ---
    # Instead of aggregating all duplicate timestamps, we need to be more careful
    # to preserve multiple species at the same timestamp
    
    # First, identify which columns should be aggregated vs preserved
    water_columns = [col for col in water_df.columns if col in combined_df.columns]
    camera_specific_columns = ['Species', 'Count', 'Distance', 'Activity']
    
    # Create a unique identifier for each camera observation
    # This ensures that multiple species at the same time remain separate
    combined_df['obs_id'] = combined_df.groupby(['DateTime', 'Species']).ngroup()
    
    # For water data columns, we want to forward-fill within each timestamp group
    # This ensures all species at a given timestamp get the same water data
    for col in water_columns:
        if col in combined_df.columns:
            combined_df[col] = combined_df.groupby('DateTime')[col].transform('first')
    
    # Now we can keep all rows without aggressive aggregation
    # Just remove true duplicates (same time, same species)
    combined_df = combined_df.drop_duplicates(subset=['DateTime', 'Species'])
    
    # Drop the temporary obs_id column
    combined_df = combined_df.drop(columns=['obs_id'])

    # --- Time-Aware Interpolation ---
    # We need a different approach since we have duplicate timestamps
    # First, save the complete data with duplicates
    complete_df = combined_df.copy()
    
    # For interpolation, we'll work with unique timestamps only
    # Get water data columns to interpolate
    water_columns = [col for col in water_df.columns if col in combined_df.columns]
    numeric_water_cols = [col for col in water_columns if col in combined_df.select_dtypes(include=np.number).columns]
    
    # Create a temporary dataframe with unique timestamps for interpolation
    unique_time_df = combined_df[['DateTime'] + numeric_water_cols].drop_duplicates(subset=['DateTime'])
    unique_time_df.set_index('DateTime', inplace=True)
    
    # Perform interpolation on the unique timestamp data
    if len(numeric_water_cols) > 0 and max_interp_hours > 0:
        # Upsample to minute frequency
        upsampled_df = unique_time_df.resample('1min').asfreq()
        limit_in_minutes = int(max_interp_hours * 60)
        
        # Interpolate the water data
        upsampled_df[numeric_water_cols] = upsampled_df[numeric_water_cols].interpolate(
            method='time',
            limit=limit_in_minutes,
            limit_direction='both'
        )
        
        # Return to original timestamps
        interpolated_df = upsampled_df.reindex(unique_time_df.index).reset_index()
        
        # Merge the interpolated water data back with the complete data
        # First, drop the old water columns from complete_df
        complete_df = complete_df.drop(columns=numeric_water_cols, errors='ignore')
        
        # Then merge with the interpolated data
        combined_df = pd.merge(
            complete_df,
            interpolated_df,
            on='DateTime',
            how='left'
        )
    else:
        combined_df = complete_df

    # Update has_camera_data after interpolation
    combined_df['has_camera_data'] = combined_df['Species'].notna()
    
    print(f"--- Combination and Interpolation Complete ---")
    print(f"Final dataset has {len(combined_df)} rows")
    print(f"Unique timestamps: {combined_df['DateTime'].nunique()}")
    print(f"Rows with camera data: {combined_df['has_camera_data'].sum()}")
    
    # Show a sample of rows with multiple species at same timestamp
    multi_species_times = combined_df[combined_df.duplicated(subset=['DateTime'], keep=False)]
    if not multi_species_times.empty:
        print(f"\nExample of multiple species at same timestamp:")
        sample_time = multi_species_times['DateTime'].iloc[0]
        print(combined_df[combined_df['DateTime'] == sample_time][['DateTime', 'Species', 'Count']].head())
    
    return combined_df