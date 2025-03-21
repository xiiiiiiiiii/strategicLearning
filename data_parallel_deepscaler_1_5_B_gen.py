# SPDX-License-Identifier: Apache-2.0
# usage:
# VLLM_USE_V1=1 uv run data_parallel_deepscaler_1_5_B_gen.py
# VLLM_LOG_LEVEL=DEBUG VLLM_VERBOSE=TRUE VLLM_USE_V1=1 uv run data_parallel_deepscaler_1_5_B_gen.py
# from https://docs.vllm.ai/en/latest/getting_started/examples/data_parallel.html
# we need to have a launcher to create multiple data parallel
# ranks. And each rank will create a vLLM instance to process its own prompts.

# Too slow on QwQ 32B model, will use outside Groq API.

import os
import json
import re

from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
from datasets import load_dataset, Dataset
import torch


# Needs at least 2 H100 GPUs with 80GB each.
GPUs_per_dp_rank = 1
DP_size = torch.cuda.device_count()

sampling_params = SamplingParams(
    n=64,
    temperature=1.0,
    top_p=0.95,
    min_tokens=10,
    max_tokens=32767
)
model = "agentica-org/DeepScaleR-1.5B-Preview"
run_name = "DeepScaleR_1_5_B_results"

DEBUG_K = 5 # 0 if not debugging.



def remove_boxed(s):
    if s is None:
        return None
    
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

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


# def get_deepscaler_questions(split="train", num_samples=None) -> Dataset:
#     """Load questions from DeepScaleR Preview Dataset with optional sample size."""
#     data = load_dataset('agentica-org/DeepScaleR-Preview-Dataset', split=split)
    
#     if num_samples is not None:
#         # Randomly sample the specified number of examples
#         data = data.shuffle(seed=42).select(range(min(num_samples, len(data))))
    
#     # As done in DeepScaleR paper.
#     SYSTEM_PROMPT = "Let's think step by step and output the final answer within \\boxed{}."    
#     data = data.map(lambda x: {
#         'prompt': f"{x['problem']} {SYSTEM_PROMPT}"
#     })
#     return data

def load_jsonl(file_path):
    SYSTEM_PROMPT = "Let's think step by step and output the final answer within \\boxed{}."
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
        data = [{
            'prompt': f"{x['problem']} {SYSTEM_PROMPT}",
            **x
        } for x in data]
        return data


def main(dp_size, dp_rank, dp_master_ip, dp_master_port, GPUs_per_dp_rank):
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    # set devices for each dp_rank
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(dp_rank * GPUs_per_dp_rank, (dp_rank + 1) *
                              GPUs_per_dp_rank))
    
    # Load the data
    # dataset = get_deepscaler_questions(num_samples=DEBUG_K)
    dataset = load_jsonl("./DeepScaleR-eval/tweak_dataset.jsonl")
    print(f"Loaded {len(dataset)} records")
    prompts = [x['prompt'] for x in dataset]
    answers = [x['answer'] for x in dataset]

    # Sample prompts.
    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]

    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    promts_per_rank = len(prompts) // dp_size
    start = dp_rank * promts_per_rank
    end = start + promts_per_rank

    prompts = prompts[start:end]
    answers = answers[start:end]

    if len(prompts) == 0:
        # if any rank has no prompts to process,
        # we need to set a placeholder prompt
        prompts = ["Placeholder"]
    print(f"DP rank {dp_rank} needs to process {len(prompts)} prompts")

    # Create a sampling params object.
    # since we are doing data parallel, every rank can have different
    # sampling params. here we set different max_tokens for different
    # ranks for demonstration.
    # sampling_params = SamplingParams(temperature=0.8,
    #                                  top_p=0.95,
    #                                  max_tokens=16 * (dp_rank + 1))

    # Create an LLM.
    llm = LLM(
        model=model,
        tensor_parallel_size=GPUs_per_dp_rank,
    )

    # Create directory for saving results
    folder = f"data/{run_name}"
    os.makedirs(folder, exist_ok=True)
    
    # Process one prompt at a time and save incrementally
    results_file = f"{folder}/{dp_rank}.jsonl"
    
    # Track completed prompts for progress info
    total_prompts = len(prompts)
    completed = 0
    
    with open(results_file, "w") as f:
        for i, (prompt, answer) in enumerate(zip(prompts, answers)):
            prompt_id = i + start  # Include global ID for tracking
            print(f"Processing prompt {i+1}/{total_prompts} (global ID: {prompt_id})")
            
            try:
                # Process just one prompt at a time
                output = llm.generate([prompt], sampling_params)
                
                # Create result for this prompt
                result = {
                    'prompt': prompt,
                    'answer': answer,
                    'prompt_id': prompt_id,
                    'outputs': [
                        {
                            'output': o.text,
                            'extracted_answer': extract_solution(o.text),
                            'reward': correctness_reward_func(o.text, answer)
                        }
                        for o in output[0].outputs
                    ],
                }
                
                # Save this single result to file immediately
                f.write(json.dumps(result) + "\n")
                f.flush()  # Ensure it's written to disk
                
                completed += 1
                print(f"✓ Completed {completed}/{total_prompts} prompts")
                
            except Exception as e:
                # Save error information if a prompt fails
                error_result = {
                    'prompt': prompt,
                    'answer': answer,
                    'prompt_id': prompt_id,
                    'error': str(e),
                    'outputs': []
                }
                f.write(json.dumps(error_result) + "\n")
                f.flush()
                print(f"✗ Error processing prompt {i+1}: {e}")
    
    print(f"Rank {dp_rank} completed processing {completed}/{total_prompts} prompts")


if __name__ == "__main__":
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()

    # # Debug without multiprocessing.
    # main(DP_size, 0, dp_master_ip, dp_master_port, GPUs_per_dp_rank)

    from multiprocessing import Process
    procs = []
    for i in range(DP_size):
        proc = Process(target=main,
                       args=(DP_size, i, dp_master_ip, dp_master_port,
                             GPUs_per_dp_rank))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
