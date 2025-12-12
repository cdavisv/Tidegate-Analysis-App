"""
Camera data loading, normalization, and species expansion module.

This module loads raw camera CSV data and transforms it into a standardized,
analysis-ready format. It handles datasets where multiple species may be recorded
in a single observation row.

Major responsibilities:
- Dynamic detection of species and count columns
- Expansion of wide-format species data into long format
- Standardization of species names
- Preservation of no-animal camera observations
- Validation of timestamps and structural integrity

The output is suitable for downstream merging, environmental analysis, and
species-level modeling.
"""


import pandas as pd

def find_species_columns(df):
    """
    Dynamically and accurately finds all 'Species X' columns in the DataFrame.
    """
    species_cols = []
    for col in df.columns:
        if col.startswith('Species '):
            parts = col.split(' ')
            if len(parts) == 2 and parts[1].isdigit():
                species_cols.append(col)
    print(f"Found {len(species_cols)} species columns in the data: {species_cols}")
    return species_cols

def process_and_combine_species(df, species_cols, base_cols):
    """
    Processes each species column using the provided base_cols, combines them 
    into a single long-format DataFrame. Only processes rows that actually have species data.
    """
    all_species_dfs = []
    
    print(f"Processing {len(species_cols)} species columns for DETECTION rows only...")
    print(f"Base columns: {base_cols}")

    # Handle blank species names with counts > 0 (only for rows we're processing)
    for i in range(1, len(species_cols) + 1):
        species_col = f'Species {i}'
        count_col = f'Species {i} Count'
        alt_count_col = f'Species Count {i}'
        
        # Use whichever format exists in the DataFrame
        if count_col in df.columns:
            actual_count_col = count_col
        elif alt_count_col in df.columns:
            actual_count_col = alt_count_col
        else:
            actual_count_col = None

        if species_col in df.columns and actual_count_col is not None:
            # Convert count to numeric and fill NaN with 0
            df[actual_count_col] = pd.to_numeric(df[actual_count_col], errors='coerce').fillna(0)
            
            # Check for blank/empty species names with counts > 0
            blank_with_count = (df[species_col].isna() | (df[species_col].str.strip() == '')) & (df[actual_count_col] > 0)
            if blank_with_count.any():
                df.loc[blank_with_count, species_col] = 'Unknown'
                print(f"   -> Filled {blank_with_count.sum()} blank species names in {species_col} with 'Unknown'")

    # Process each species column
    for i in range(1, len(species_cols) + 1):
        species_col = f'Species {i}'
        count_col = f'Species {i} Count'
        alt_count_col = f'Species Count {i}'
        notes_col = f'Notes {i}'
        
        # Use whichever count format exists in the DataFrame
        if count_col in df.columns:
            actual_count_col = count_col
        elif alt_count_col in df.columns:
            actual_count_col = alt_count_col
        else:
            actual_count_col = None
        
        print(f"   -> Processing {species_col}...")
        
        if species_col not in df.columns:
            print(f"      Warning: {species_col} not found in DataFrame")
            continue
        
        # Build list of columns for this species
        current_species_cols = [species_col]
        if actual_count_col is not None:
            current_species_cols.append(actual_count_col)
        if notes_col in df.columns:
            current_species_cols.append(notes_col)

        # Create temporary dataframe with base columns + species columns
        temp_df = df[base_cols + current_species_cols].copy()
        
        # IMPORTANT: Only filter out rows where species is explicitly null/empty
        # Don't filter out rows where species exists but might be blank
        before_filter = len(temp_df)
        temp_df = temp_df[
            temp_df[species_col].notna() & 
            (temp_df[species_col].astype(str).str.strip() != '') &
            (temp_df[species_col].astype(str).str.strip() != 'nan')
        ]
        after_filter = len(temp_df)
        
        print(f"      Rows before filtering: {before_filter}")
        print(f"      Rows after filtering: {after_filter}")
        print(f"      Valid species entries: {after_filter}")

        if temp_df.empty:
            print(f"      No valid entries found in {species_col}")
            continue

        # Rename columns to standard names
        rename_dict = {species_col: 'Species'}
        if actual_count_col is not None:
            rename_dict[actual_count_col] = 'Count'
        if notes_col in df.columns:
            rename_dict[notes_col] = 'Notes'
        
        temp_df = temp_df.rename(columns=rename_dict)

        # Add Notes column if it doesn't exist
        if 'Notes' not in temp_df.columns:
            temp_df['Notes'] = ''

        # Add Count column with default value of 1 if missing
        if 'Count' not in temp_df.columns:
            temp_df['Count'] = 1
        else:
            # Ensure Count is numeric and fill NaN with 1
            temp_df['Count'] = pd.to_numeric(temp_df['Count'], errors='coerce').fillna(1)

        print(f"      Sample species values: {temp_df['Species'].head(3).tolist()}")
        all_species_dfs.append(temp_df)

    if not all_species_dfs:
        print("\nNo valid species entries found in detection rows.")
        return pd.DataFrame()

    print(f"\nCombining {len(all_species_dfs)} DataFrames...")
    combined = pd.concat(all_species_dfs, ignore_index=True)
    print(f"Combined detections DataFrame shape: {combined.shape}")
    print(f"Unique species found in detections: {combined['Species'].nunique()}")
    
    return combined


def standardize_species_names(df):
    """Standardizes species names to correct for typos and variations."""
    print("\n--- Standardizing Species Names ---")
    
    if df.empty:
        print("DataFrame is empty, skipping standardization.")
        return df
    
    special_cases = {
        'unknown': 'Unknown', 'brant': 'Branta canadensis', 'canada goose': 'Branta canadensis',
        'canada geese': 'Branta canadensis', 'cackling goose': 'Branta hutchinsii', 'great egret': 'Ardea alba',
        'great blue heron': 'Ardea herodias', 'belted kingfisher': 'Megaceryle alcyon',
        'double-crested cormorant': 'Nannopterum auritus', 'pelagic cormorant': 'Urile pelagicus',
        'river otter': 'Lontra canadensis', 'columbian black-tailed deer': 'Odocoileus Hemionus Columbianus',
        'black-tailed deer': 'Odocoileus Hemionus Columbianus', 'turkey vulture': 'Cathartes aura',
        'red-necked grebe': 'Podiceps grisegena', 'common loon': 'Gavia immer',
        'common merganser': 'Mergus merganser', 'bufflehead': 'Bucephala albeola',
        'mallard': 'Anas platyrhynchos', 'american crow': 'Corvus brachyrhynchos',
        'cormorant': 'Phalacrocoracidae'
    }

    original_species = df['Species'].unique()
    print(f"Species names before standardization: {len(original_species)}")
    
    mapping = {k.lower().strip(): v for k, v in special_cases.items()}
    df['Species'] = df['Species'].str.lower().str.strip().map(mapping).fillna(df['Species'].str.strip().str.title())
    
    standardized_species = df['Species'].unique()
    print(f"Species names after standardization: {len(standardized_species)}")
    
    merged_count = len(original_species) - len(standardized_species)
    if merged_count > 0:
        print(f"Merged {merged_count} duplicate species names")
        
    return df

def analyze_multi_species_rows(original_df, expanded_df):
    """Provides statistics on rows that contained multiple species."""
    print("\n--- Multi-Species Expansion Statistics ---")
    
    if expanded_df.empty:
        print("Expanded DataFrame is empty, skipping multi-species analysis.")
        return
    
    multi_species_timestamps = expanded_df['DateTime'].value_counts()
    multi_species_timestamps = multi_species_timestamps[multi_species_timestamps > 1].index
    
    if len(multi_species_timestamps) > 0:
        print(f"Timestamps with multiple species: {len(multi_species_timestamps)}")
        shared_timestamp_rows = expanded_df[expanded_df['DateTime'].isin(multi_species_timestamps)]
        print(f"Total rows with shared timestamps: {len(shared_timestamp_rows)}")
        
        example_timestamp = multi_species_timestamps[0]
        print(f"\nExample - Multiple species at {example_timestamp}:")
        print(expanded_df[expanded_df['DateTime'] == example_timestamp][['DateTime', 'Species', 'Count']])
    else:
        print("No instances of multiple species detected in a single timestamp.")

def load_and_prepare_camera_data(file_id):
    """
    Main function to load, process, and standardize camera data.
    FIXED: Now includes rows with no species data as 'No_Animals_Detected'.
    """
    print("\n--- Loading and Preparing Camera Data ---")
    try:
        df = pd.read_csv(file_id, low_memory=False)
        print(f"Raw camera data shape: {df.shape}")
        print(f"Column names: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading camera data: {e}")
        return None

    # Handle DateTime column creation
    if 'DateTime' in df.columns:
        print("\nFound existing 'DateTime' column.")
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    elif 'Date' in df.columns and 'Time' in df.columns:
        print("\nFound 'Date' and 'Time' columns. Combining them into 'DateTime'.")
        try:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        except Exception as e:
            print(f"Error combining 'Date' and 'Time' columns: {e}")
            return None
    else:
        print("\nCRITICAL ERROR: Could not find 'DateTime' column, nor 'Date' and 'Time' columns to create it.")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Remove rows with invalid DateTime
    before_datetime_filter = len(df)
    df = df.dropna(subset=['DateTime'])
    after_datetime_filter = len(df)
    print(f"Removed {before_datetime_filter - after_datetime_filter} rows with invalid DateTime")
    
    # Define base columns (columns that are not species-specific)
    base_cols = [col for col in df.columns if not any(sc in col for sc in ['Species', 'Count', 'Notes'])]
    print(f"\nBase columns (non-species-specific): {len(base_cols)}")
    print(f"Base columns: {base_cols}")

    # CRITICAL FIX: Check for camera observations vs detections
    species_1_col = 'Species 1'
    if species_1_col not in df.columns:
        print(f"ERROR: '{species_1_col}' column not found!")
        return None
    
    # Count total camera observations (all rows with valid DateTime)
    total_camera_observations = len(df)
    
    # Count actual detections (rows with Species 1 data)
    actual_detections = df[species_1_col].notna() & (df[species_1_col].str.strip() != '')
    detection_count = actual_detections.sum()
    no_detection_count = total_camera_observations - detection_count
    
    print(f"\n🔍 CAMERA OBSERVATION ANALYSIS:")
    print(f"   • Total camera observations: {total_camera_observations:,}")
    print(f"   • Observations with animals detected: {detection_count:,}")
    print(f"   • Observations with no animals detected: {no_detection_count:,}")
    print(f"   • True detection rate: {(detection_count/total_camera_observations)*100:.1f}%")

    # Find species columns
    species_cols = find_species_columns(df)
    
    if not species_cols:
        print("No species columns found.")
        species_cols = []
    
    # FIXED APPROACH: Process ALL camera observations
    all_camera_observations = []
    
    # 1. First, add all rows with NO detections
    no_detection_rows = df[~actual_detections].copy()
    if not no_detection_rows.empty:
        # Create standardized format for no-detection rows
        no_detection_standardized = no_detection_rows[base_cols].copy()
        #no_detection_standardized['Species'] = 'No_Animals_Detected'
        no_detection_standardized['Count'] = 0
        no_detection_standardized['Notes'] = 'No animals detected'
        all_camera_observations.append(no_detection_standardized)
        print(f"   • Added {len(no_detection_standardized)} 'No_Animals_Detected' records")
    
    # 2. Then, process rows WITH detections using existing logic
    detection_rows = df[actual_detections].copy()
    if not detection_rows.empty and species_cols:
        processed_detections = process_and_combine_species(detection_rows, species_cols, base_cols)
        if not processed_detections.empty:
            # Standardize species names for detections
            processed_detections = standardize_species_names(processed_detections)
            all_camera_observations.append(processed_detections)
            print(f"   • Added {len(processed_detections)} animal detection records")
    
    # 3. Combine all observations
    if all_camera_observations:
        final_df = pd.concat(all_camera_observations, ignore_index=True)
        
        # FIXED: Calculate statistics correctly based on actual data structure
        total_records = len(final_df)
        detection_records = len(final_df[final_df['Species'].notna()])  # Rows with actual species
        no_detection_records = len(final_df[final_df['Notes'] == 'No animals detected'])  # Rows with no animals
        
        print(f"\n✅ FINAL CAMERA DATA:")
        print(f"   • Total records: {total_records:,}")
        print(f"   • Detection records: {detection_records:,}")
        print(f"   • No-detection records: {no_detection_records:,}")
        print(f"   • Unique species: {final_df['Species'].nunique()}")
        
        # Show species distribution
        print("\nSpecies distribution:")
        # Only show actual species (exclude NaN)
        species_counts = final_df[final_df['Species'].notna()]['Species'].value_counts()
        print(species_counts.head(10))
        
        return final_df
    else:
        print("\nERROR: No camera observations could be processed.")
        return None


def load_and_prepare_water_data(file_id):
    """Loads and prepares water quality and environmental data."""
    print("\n--- Loading Water Quality Data ---")
    try:
        df = pd.read_csv(file_id, low_memory=False)
        print(f"Raw water data shape: {df.shape}")
    except Exception as e:
        print(f"Error loading water data: {e}")
        return None

    # Rename columns to standardized names
    rename_map = {
        'Air Temp [C]': 'Air_Temp_C', 
        'Gate Opening MTR [Degrees]': 'Gate_Opening_MTR_Deg',
        'Gate Opening Top Hinge [Degrees]': 'Gate_Opening_Top_Hinge_Deg',
        'Tidal Level Outside Tidegate [m]': 'Depth', 
        'Tidal Level Inside Tidegate [m]': 'Depth_Inside',
        'Wind Speed [km/h]': 'Wind_Speed_km_h'
    }
    df = df.rename(columns=rename_map)

    # Handle DateTime column creation
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    elif 'Date' in df.columns and 'Time' in df.columns:
        try:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        except Exception as e:
            print(f"Error combining 'Date' and 'Time' columns: {e}")
            return None
    else:
        print("\nCRITICAL ERROR: Could not find 'DateTime' column, nor 'Date' and 'Time' columns to create it.")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Remove rows with invalid DateTime
    before_datetime_filter = len(df)
    df = df.dropna(subset=['DateTime'])
    after_datetime_filter = len(df)
    print(f"Removed {before_datetime_filter - after_datetime_filter} rows with invalid DateTime")

    # Process numeric columns
    numeric_cols = [
        'Air_Temp_C', 'Gate_Opening_MTR_Deg', 'Gate_Opening_Top_Hinge_Deg',
        'Depth', 'Depth_Inside', 'Wind_Speed_km_h'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            # Handle depth columns specially - replace 0 with NaN
            if 'Depth' in col:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    print(f"Replacing {zero_count} zero values in {col} with NaN")
                    df[col] = df[col].replace(0, pd.NA)

            # Convert to numeric
            original_non_numeric = df[col].notna().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            final_non_numeric = df[col].notna().sum()
            converted = original_non_numeric - final_non_numeric
            if converted > 0:
                print(f"Converted {converted} non-numeric values to NaN in {col}")

    # Select final columns
    final_cols = ['DateTime'] + [col for col in df.columns if col in list(rename_map.values())]
    final_cols = [col for col in final_cols if col in df.columns]
    
    result_df = df[final_cols]
    print(f"Final water data shape: {result_df.shape}")
    print(f"Final columns: {list(result_df.columns)}")
    
    return result_df