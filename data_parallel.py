# SPDX-License-Identifier: Apache-2.0
# usage:
# VLLM_USE_V1=1 python examples/offline_inference/data_parallel.py
# we need to have a launcher to create multiple data parallel
# ranks. And each rank will create a vLLM instance to process its own prompts.
import os
import json
import re

from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
from datasets import load_dataset, Dataset
import torch


GPUs_per_dp_rank = 1
DP_size = torch.cuda.device_count()

sampling_params = SamplingParams(
    n=20,
    temperature=1.0,
    top_p=0.95,
    min_tokens=10,
    max_tokens=32767
)
model = "agentica-org/DeepScaleR-1.5B-Preview"

DEBUG_K = 5 # 0 if not debugging.

SYSTEM_PROMPT = """You are a powerful math problem solving assistant. For each math problem:

1. Think step-by-step, breaking down the problem into simpler parts
2. Show your calculations clearly
3. Verify your solution matches the question requirements
4. Format your response as follows:

<think>
[Your detailed step-by-step reasoning here]
</think>
[your summary of the solution]
\\boxed{[Your final answer here, with no extra text]}
"""

def extract_last_boxed_value(text):
    # Find all boxed values in the text
    matches = re.findall(r"\\boxed{([^}]+)}", text)
    
    # Return the last match if exists, otherwise None
    return matches[-1] if matches else None


def correctness_reward_func(response: str, actual_answers: str) -> float:
    extracted_answer = extract_last_boxed_value(response)
    return 1.0 if extracted_answer == actual_answers else 0.0

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': f"{SYSTEM_PROMPT}\n\n{x['question']}",
        'trace': x['answer'],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore


def main(dp_size, dp_rank, dp_master_ip, dp_master_port, GPUs_per_dp_rank):
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    # set devices for each dp_rank
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(dp_rank * GPUs_per_dp_rank, (dp_rank + 1) *
                              GPUs_per_dp_rank))
    
    hf_dataset = get_gsm8k_questions()
    prompts = hf_dataset['prompt']
    answers = hf_dataset['answer']
    trace = hf_dataset['trace']
    if DEBUG_K > 0:
        prompts = prompts[0:DEBUG_K]
        answers = answers[0:DEBUG_K]
        trace = trace[0:DEBUG_K]

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
    from multiprocessing import Process
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()
    procs = []
    for i in range(DP_size):
        proc = Process(target=main,
                       args=(DP_size, i, dp_master_ip, dp_master_port,
                             GPUs_per_dp_rank))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()