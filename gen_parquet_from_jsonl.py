"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any
import json

import pandas as pd



def load_dataset(dataset_file_path: str) -> List[Dict[str, Any]]:
    """Load a dataset from a JSONL file line by line."""
    if not os.path.exists(dataset_file_path):
        raise ValueError(f"Dataset file not found: {dataset_file_path}")

    data = []
    try:
        with open(dataset_file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {dataset_file_path}: {e}")
    except Exception as exc:
        raise ValueError(f"Error loading dataset: {exc}") from exc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--dataset_file_path', default='../train.jsonl',
                       help='Path to the dataset file')
    args = parser.parse_args()

    dataset_file_path = args.dataset_file_path

    # Initialize datasets
    train_dataset = load_dataset(dataset_file_path)

    # Save training dataset
    print("train data size:", len(train_dataset))
    train_df = pd.DataFrame(train_dataset)
    train_df.to_parquet('train.parquet')
    train_df.to_json('train.jsonl', orient='records', lines=True)
