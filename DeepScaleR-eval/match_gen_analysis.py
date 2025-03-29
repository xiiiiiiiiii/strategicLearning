#!/usr/bin/env python3
# match_gen_analysis.py - Script to analyze model performance and create tweak dataset

import os
import pandas as pd
import glob
import json
import numpy as np

# Output file name
OUTPUT_FILE = "grpo_tweak_dataset.jsonl"

def load_all_jsonl_to_dataframe(folder_path):
    """Load all JSONL files from a folder into a pandas DataFrame."""
    # Find all .jsonl files
    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    
    # Create list to hold dataframes
    dfs = []
    
    for file_path in jsonl_files:
        try:
            # Read file into dataframe
            df = pd.read_json(file_path, lines=True)
            df['source_file'] = os.path.basename(file_path)  # Optional: track source file
            dfs.append(df)
            print(f"Loaded {len(df)} rows from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Combine all dataframes
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Created DataFrame with {len(combined_df)} rows from {len(jsonl_files)} files")
        return combined_df
    else:
        print("No valid files found")
        return pd.DataFrame()

def remove_boxed(s):
    if s is None:
        return None

    left = "\\boxed{"

    if s[:len(left)] != left:
        return None
    if s[-1] != "}":
        return None

    return s[len(left):-1]

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def extract_solution(text):
    return remove_boxed(last_boxed_only_string(text))

def correctness_reward_func(response: str, actual_answers: str) -> float:
    extracted_answer = extract_solution(response)
    return 1.0 if extracted_answer == actual_answers else 0.0

def pass_at_k(n, c, k):
    """
    :param n: total number of samples 
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k: 
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze model performance and create tweak dataset')
    parser.add_argument('--input_folder', type=str, default="../data/DeepScaleR_1_5_B_results/",
                        help='Path to folder containing JSONL result files')
    args = parser.parse_args()
    
    # Load data
    folder_path = args.input_folder
    combined_data = load_all_jsonl_to_dataframe(folder_path)
    
    # Check unique questions and answers
    check = combined_data.copy()[['prompt', 'answer']].drop_duplicates()
    
    # Transform for analysis
    df = combined_data.copy()
    df['question_text'] = df['prompt']
    df['ground_truth'] = df['answer']
    df.drop(['source_file'], axis=1, inplace=True)
    
    df = df.explode('outputs')
    df['response'] = df['outputs'].apply(lambda x: x['output'])
    df['extracted_solution'] = df['response'].apply(lambda x: extract_solution(x))
    df['response_word_count'] = df['response'].astype(str).str.split().str.len()
    df['reward'] = (df['extracted_solution'] == df['ground_truth'].astype(str)).astype(float)
    
    # Calculate pass@k metrics
    k = 1
    perf = (
        df[['question_text', 'ground_truth', 'reward']]
        .groupby(['question_text', 'ground_truth'])
        .agg(
            count=('reward', 'count'),
            sum_reward=('reward', 'sum'),
            accuracy=('reward', 'mean')
        )
    )
    
    print(f"len(perf): {len(perf)}")
    
    perf[f'pass@{k}'] = perf.apply(
        lambda x: pass_at_k(x['count'], x['sum_reward'], k), 
        axis=1
    )
    
    # Categorize problems
    total = len(perf)
    print(f"total: {total}")
    
    # Too hard
    too_hard = perf[(perf['accuracy'] == 0.0)]
    total_too_hard = len(too_hard)
    print(f"total_too_hard: {total_too_hard}")
    
    # Ripe for improvement
    ripe = perf[(perf['accuracy'] <= 0.8) & (perf['accuracy'] > 0.0)]
    total_ripe = len(ripe)
    print(f"total_ripe: {total_ripe}")
    
    # Not worth improving
    already_done = perf[(perf['accuracy'] >= 0.8)]
    total_already_done = len(already_done)
    print(f"total_already_done: {total_already_done}")
    
    # Verification
    check_total = total_too_hard + total_ripe + total_already_done
    assert(total == check_total)
    
    # Create tweak dataset
    tweak_dataset = [
        {
            'problem': r['question_text'],
            'answer': str(r['ground_truth']),
        }
        for index, r in ripe.reset_index().iterrows()
    ]
    
    # Create directory if it doesn't exist
    dirname = os.path.dirname(OUTPUT_FILE)
    if len(dirname.strip()) > 0:
        os.makedirs(dirname, exist_ok=True)
    
    # Save to JSONL file
    with open(OUTPUT_FILE, 'w') as f:
        for item in tweak_dataset:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {len(tweak_dataset)} records to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()