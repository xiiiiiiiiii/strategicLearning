#!/usr/bin/env python3
import os
import sys
import pandas as pd
from pathlib import Path

def view_parquet_file(file_path, num_rows=5, verbose=False):
    """
    Display the content of a single parquet file with detailed row view option.
    
    Args:
        file_path: Path to the parquet file
        num_rows: Number of rows to display (default: 5)
        verbose: Whether to display the full content of each field (default: False)
    """
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: File '{file_path}' does not exist")
        return
    
    if not path.is_file() or path.suffix != '.parquet':
        print(f"Error: '{file_path}' is not a valid parquet file")
        return
    
    try:
        print(f"\n{'='*80}")
        print(f"File: {path}")
        print(f"{'='*80}")
        
        df = pd.read_parquet(path)
        print(f"\nShape: {df.shape} (rows × columns)")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst {num_rows} rows:")
        
        if verbose:
            # Display detailed view for each row
            for i, row in df.head(num_rows).iterrows():
                print(f"\nRow {i}:")
                print("-" * 40)
                for col, value in row.items():
                    print(f"{col}:")
                    
                    # Try to format the output in a readable way
                    import json
                    
                    def format_value(value, indent=0):
                        indent_str = "  " * indent
                        
                        # Format lists and arrays
                        if isinstance(value, (list, tuple)) or hasattr(value, 'tolist'):
                            # Convert numpy arrays to list
                            if hasattr(value, 'tolist'):
                                items = value.tolist()
                            else:
                                items = value
                                
                            # For single-item arrays
                            if len(items) == 1:
                                print(indent_str, end="")
                                format_value(items[0], indent)
                                return
                            
                            # For multi-item arrays
                            for item in items:
                                format_value(item, indent)
                                
                        # Format dictionaries nicely
                        elif isinstance(value, dict):
                            for k, v in value.items():
                                print(f"{indent_str}{k}:")
                                if isinstance(v, (dict, list, tuple)) or hasattr(v, 'tolist'):
                                    format_value(v, indent + 1)
                                else:
                                    print(f"{indent_str}  {v}")
                        
                        # Format strings - special handling for Markdown/LaTeX
                        elif isinstance(value, str):
                            if value.startswith("<think>") or "\\boxed" in value:
                                # For Markdown with LaTeX, print with spacing
                                lines = value.split("\n")
                                for line in lines:
                                    print(f"{indent_str}{line}")
                            else:
                                print(f"{indent_str}{value}")
                        
                        # Default formatting for other types
                        else:
                            print(f"{indent_str}{value}")
                    
                    # Format the value using our custom formatter
                    format_value(value)
                    
                    print("-" * 40)
        else:
            # Display standard dataframe view
            print(df.head(num_rows))
        
    except Exception as e:
        print(f"Error reading {path}: {e}")

def view_parquet_files(directory_path, num_rows=5, verbose=False):
    """
    Display the content of all parquet files in the specified directory and its subdirectories.
    
    Args:
        directory_path: Path to the directory containing parquet files
        num_rows: Number of rows to display from each file (default: 5)
        verbose: Whether to display the full content of each field (default: False)
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory '{directory_path}' does not exist")
        return
    
    if directory.is_file() and directory.suffix == '.parquet':
        # If a single parquet file is provided, view it
        view_parquet_file(directory_path, num_rows, verbose)
        return
    
    parquet_files = list(directory.glob('**/*.parquet'))
    
    if not parquet_files:
        print(f"No parquet files found in '{directory_path}'")
        return
    
    for file_path in parquet_files:
        view_parquet_file(file_path, num_rows, verbose)

if __name__ == "__main__":
    path = "/Users/auguste.pribula/Downloads/deepscaler_eval_logs"
    num_rows = 5
    verbose = False
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            num_rows = int(sys.argv[2])
        except ValueError:
            print(f"Error: '{sys.argv[2]}' is not a valid number of rows")
            sys.exit(1)
    
    if len(sys.argv) > 3 and sys.argv[3].lower() in ['true', 't', 'yes', 'y', '1']:
        verbose = True
    
    # Check if the path is a file or directory and call the appropriate function
    if Path(path).is_file() and Path(path).suffix == '.parquet':
        view_parquet_file(path, num_rows, verbose)
    else:
        view_parquet_files(path, num_rows, verbose)