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
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str: str) -> str:
    """Extract the final boxed solution from a solution string.

    Args:
        solution_str: Raw solution string that may contain multiple boxed answers

    Returns:
        The final boxed answer with box notation removed
    """
    return remove_boxed(last_boxed_only_string(solution_str))


def make_map_fn(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example.pop('problem')
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        question = f"{question} {instruction}"
        answer = example.pop('answer')

        data = {
            "data_source": "",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn

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
    parser.add_argument('--dataset_file_path', default='./grpo_tweak_dataset.jsonl',
                       help='Path to the dataset file')
    parser.add_argument('--local_dir', default='./data',
                       help='Local directory to save processed datasets')
    args = parser.parse_args()

    local_dir = args.local_dir
    dataset_file_path = args.dataset_file_path
    
    # Make local directory if it doesn't exist
    makedirs(local_dir)

    # Initialize datasets
    train_dataset = load_dataset(dataset_file_path)

    # Process training data
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')
    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            train_data.append(processed_example)

    # Save training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))
    train_df.to_json(os.path.join(local_dir, 'train.jsonl'), orient='records', lines=True)
