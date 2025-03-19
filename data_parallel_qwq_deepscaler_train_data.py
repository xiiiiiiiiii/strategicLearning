# SPDX-License-Identifier: Apache-2.0
# usage:
# VLLM_USE_V1=1 python examples/offline_inference/data_parallel.py
# we need to have a launcher to create multiple data parallel
# ranks. And each rank will create a vLLM instance to process its own prompts.

# Set the start method to 'spawn' before creating any processes
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
import json
import re

from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
from datasets import load_dataset, Dataset
import torch


GPUs_per_dp_rank = 1
DP_size = 2 # torch.cuda.device_count()

sampling_params = SamplingParams(
    n=10,
    temperature=1.0,
    top_p=0.95,
    min_tokens=10,
    max_tokens=32767
)
# model = "Qwen/QwQ-32B"
model = "agentica-org/DeepScaleR-1.5B-Preview"

DEBUG_K = 6 # 0 if not debugging.

SYSTEM_PROMPT = """You are a powerful math problem solving assistant.

Think step by step and output the final answer within \boxed{}

Format your response as follows:
<think>
[Your detailed step-by-step reasoning here]
</think>
[your summary of the solution]
\boxed{[Your final answer here, with no extra text]}
"""

def extract_last_boxed_value(text):
    # Find all boxed values in the text
    matches = re.findall(r"\\boxed{([^}]+)}", text)
    
    # Return the last match if exists, otherwise None
    return matches[-1] if matches else None


def correctness_reward_func(response: str, actual_answers: str) -> float:
    extracted_answer = extract_last_boxed_value(response)
    return 1.0 if extracted_answer == actual_answers else 0.0


def get_deepscaler_questions(split="train", num_samples=None) -> Dataset:
    """Load questions from DeepScaleR Preview Dataset with optional sample size."""
    data = load_dataset('agentica-org/DeepScaleR-Preview-Dataset', split=split)
    
    if num_samples is not None:
        # Randomly sample the specified number of examples
        data = data.shuffle(seed=42).select(range(min(num_samples, len(data))))
    
    data = data.map(lambda x: {
        'prompt': [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x['problem']}
        ]
    })
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
    
    hf_dataset = get_deepscaler_questions(num_samples=DEBUG_K)
    prompts = hf_dataset['prompt']
    answers = hf_dataset['answer']
    trace = hf_dataset['solution']

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
    trace = trace[start:end]

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
        tensor_parallel_size=GPUs_per_dp_rank
    )
    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts)
    assert len(outputs) == len(answers)
    assert len(outputs) == len(trace)

    
    results = [
        {
            'prompt': prompt,
            'answer': answer,
            'trace': trace,
            'outputs': [
                {
                    'output':o.text,
                    'extracted_answer': extract_last_boxed_value(o.text),
                    'reward': correctness_reward_func(o.text, answer)
                }
                for o in output.outputs
            ],
        }
        for prompt, answer, trace, output in zip(prompts, answers, trace, outputs)
    ]

    # # Debug print.
    # for i, result in enumerate(results[0:10]):
    #     print(f"DP rank {dp_rank} result {i} Result: {result}")
    
    # Create directory before writing
    os.makedirs("data/data_parallel_results", exist_ok=True)
    
    # Now safely write to the file
    with open(f"data/data_parallel_results/{dp_rank}.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":    
    spawn_ctx = multiprocessing.get_context('spawn')
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()
    procs = []
    for i in range(DP_size):
        proc = spawn_ctx.Process(
            target=main,
            args=(DP_size, i, dp_master_ip, dp_master_port, GPUs_per_dp_rank)
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()