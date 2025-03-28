import os
import pandas as pd
import glob
import argparse
from pathlib import Path
import numpy as np

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
    :param n: total number of samples :param c: number of correct
    Samples
    :param k: k in pass@$k$
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

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

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate DeepScaleR output on mathematical problems')
    parser.add_argument('--folder_path', type=str, default="../data/deepscaler_eval_check/",
                        help='Path to the folder containing JSONL files with model outputs')
    args = parser.parse_args()
    
    # Load data
    folder_path = args.folder_path
    combined_data = load_all_jsonl_to_dataframe(folder_path)
    
    # Transform for analysis
    check = combined_data.copy()[['prompt', 'answer']].drop_duplicates()
    print(f"len(check): {len(check)}")
    
    df = combined_data.copy()
    df['question_text'] = df['prompt']
    df['ground_truth'] = df['answer']
    df.drop(['source_file'], axis=1, inplace=True)
    df = df.explode('outputs')
    df['response'] = df['outputs'].apply(lambda x: x['output'])
    df['extracted_solution'] = df['response'].apply(lambda x: extract_solution(x))
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
    
    perf[f'pass@{k}'] = perf.apply(
        lambda x: pass_at_k(x['count'], x['sum_reward'], k), 
        axis=1
    )
    
    # Create final dataset
    tweak_dataset = [
        {
            'problem': r['question_text'],
            'answer': str(r['ground_truth']),
            f'pass@{k}': str(r[f'pass@{k}']),
        }
        for index, r in perf.reset_index().iterrows()
    ]
    
    # Print final output (same as last cell of notebook)
    pass_at_k_str = f'pass@{k}'
    for p in tweak_dataset:
        print(f"problem {p['problem']}")
        print(f"answer {p['answer']}")
        print(f"{pass_at_k_str} {p[pass_at_k_str]}")
        print()

if __name__ == "__main__":
    main()