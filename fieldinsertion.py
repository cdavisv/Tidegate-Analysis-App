"""
CSV field correction utility.

This script updates CSV files by inserting 'Unknown' species labels when
species fields are empty but corresponding count fields indicate animal presence.

It is designed as a command-line utility for cleaning legacy or inconsistently
formatted datasets prior to ingestion into the main analysis pipeline.
"""

import pandas as pd
import sys

def update_csv(file, output_file=None):
    try:

        df = pd.read_csv(file)

        if len(df.column) < 9:
            print("Error: Not enough columns")
            return

        col_f = df.columns[5]
        col_i = df.columns[8]

        # Apply the rule: if column F is empty and column I > 0, set column F to "Unknown"
        mask = (df[col_f].isna() | (df[col_f] == '')) & (pd.to_numeric(df[col_i], errors='coerce') > 0)
        df.loc[mask, col_f] = 'Unknown'

        # set output
        if output_file is None:
            output_file = file

        df.to_csv(output_file, index=False)
        print("Success!")
        # Print summary of changes
        changes_made = mask.sum()
        print(f"Changes made: {changes_made} rows updated")

    except FileNotFoundError:
        print(f"Error: File '{file}' not found.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")


def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_csv_file> [output_csv_file]")
        print("Example: python script.py data.csv")
        print("Example: python script.py data.csv updated_data.csv")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    update_csv(input_file, output_file)


if __name__ == "__main__":
    main()