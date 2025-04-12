#!/usr/bin/env python3
"""
Script to extract the first row from a parquet file and save it as JSONL and Parquet.
"""
import os
import sys
import pandas as pd
import json

def extract_row_i(input_file, parquet_output, i):
    """
    Extract the first row from a parquet file and save it in JSONL and Parquet formats.
    
    Args:
        input_file (str): Path to the input parquet file
        jsonl_output (str): Path to the output JSONL file
        parquet_output (str): Path to the output parquet file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' does not exist.")
            return False
        
        # Read the parquet file
        print(f"Reading '{input_file}'...")
        df = pd.read_parquet(input_file)
        
        if df.empty:
            print("Error: The input parquet file is empty.")
            return False
        
        # Extract the first row
        a_row = df.iloc[[i]]
        print(f"Extracted first row with columns: {', '.join(a_row.columns)}")
        print(f"prompt: {a_row['prompt']}")

        # for index, r in df.reset_index().iterrows():
        #     print(f"index: {index} prompt: {r['prompt']}")
        
        # Save as Parquet
        try:
            a_row.to_parquet(parquet_output)
            print(f"Successfully saved Parquet file to '{parquet_output}'")
        except Exception as e:
            print(f"Error saving Parquet file: {e}")
            return False
        
        return True
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def main():
    """Main function to run the script."""
    input_file = "aime.parquet"
    index = 16
    parquet_output = "aime_single_row.parquet"
    
    print("Starting extraction process...")
    success = extract_row_i(input_file, parquet_output, index)
    
    if success:
        print("Extraction and saving completed successfully!")
    else:
        print("Extraction process failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()

