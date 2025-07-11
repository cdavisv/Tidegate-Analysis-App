# data_combiner.py

import pandas as pd
import numpy as np

def combine_data(camera_df, water_df, max_interp_hours=3):
    """
    Combines, aggregates, and interpolates camera and water data.
    """
    print("\n--- Combining Data ---")

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

    # FIX: Create the 'has_camera_data' helper column BEFORE aggregation.
    combined_df['has_camera_data'] = combined_df['Species'].notna()

    # --- Aggregate Duplicate Timestamps ---
    agg_rules = {col: 'first' for col in combined_df.columns if col != 'DateTime'}
    agg_rules['Species'] = lambda s: ', '.join(s.dropna().unique())
    agg_rules['Count'] = 'sum'
    agg_rules['Activity'] = lambda s: ', '.join(s.dropna().unique())
    # Add a rule to correctly aggregate the new helper column.
    agg_rules['has_camera_data'] = 'max'
    
    combined_df = combined_df.groupby('DateTime').agg(agg_rules).reset_index()

    # --- Time-Aware Interpolation ---
    combined_df.set_index('DateTime', inplace=True)
    original_index = combined_df.index

    upsampled_df = combined_df.resample('1min').asfreq()
    limit_in_minutes = max_interp_hours * 60

    numeric_cols = upsampled_df.select_dtypes(include=np.number).columns
    upsampled_df[numeric_cols] = upsampled_df[numeric_cols].interpolate(
        method='time',
        limit=limit_in_minutes,
        limit_direction='both'
    )

    combined_df = upsampled_df.reindex(original_index).reset_index()

    # The 'has_camera_data' column is now created before aggregation, so this line is no longer needed.
    # combined_df['has_camera_data'] = combined_df['Species'].notna() 
    
    print(f"--- Combination and Interpolation Complete. Final dataset has {len(combined_df)} rows. ---")
    
    return combined_df