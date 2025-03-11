# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use Ray Data for running offline batch inference
distributively on a multi-nodes cluster.

Learn more about Ray Data in https://docs.ray.io/en/latest/data/data.html
"""

from typing import Any

import numpy as np
import ray
import ray.data as rd
from datasets import load_dataset, Dataset
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams

assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"

# Create a sampling params object.
sampling_params = SamplingParams(temperature=1.0, top_p=0.95) #, max_tokens=32768)

# Set tensor parallelism per instance.
tensor_parallel_size = 1  # Don't split model across GPUs, it is small enough to fit into one GPU.

# Set number of instances. Each instance will use tensor_parallel_size GPUs.
num_instances = 2

model = "agentica-org/DeepScaleR-1.5B-Preview"

SYSTEM_PROMPT = """You are a powerful math problem solving assistant. For each math problem:

1. Think step-by-step, breaking down the problem into simpler parts
2. Show your calculations clearly
3. Verify your solution matches the question requirements
4. Format your response as follows:

<reasoning>
[Your detailed step-by-step reasoning here]
</reasoning>
<answer>
[Your final numerical answer here, with no units or extra text]
</answer>
"""

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        # 'trace': x['answer'],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore
  
hf_dataset = get_gsm8k_questions()


# Read one text file from S3. Ray Data supports reading multiple files
# from cloud storage (such as JSONL, Parquet, CSV, binary format).
# ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
ds = rd.from_huggingface(hf_dataset)


# Create a class to do batch inference.
class LLMPredictor:

    def __init__(self):
        # Create an LLM.
        self.llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size)

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.

        # Example data:
  #       {'question': 'Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?',
  # 'answer': '990',
  # 'prompt': [{'content': 'You are a powerful math problem solving assistant. For each math problem:\n\n1. Think step-by-step, breaking down the problem into simpler parts\n2. Show your calculations clearly\n3. Verify your solution matches the question requirements\n4. Format your response as follows:\n\n<reasoning>\n[Your detailed step-by-step reasoning here]\n</reasoning>\n<answer>\n[Your final numerical answer here, with no units or extra text]\n</answer>\n',
  #   'role': 'system'},
  #  {'content': 'Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?',
  #   'role': 'user'}]}]
        # print(f"Batch: {batch}")
        outputs = self.llm.generate(batch["question"], sampling_params)
        # print(f"outputs: {outputs}")
        # print()
        # print()
        prompt: list[str] = []
        generated_text: list[str] = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(' '.join([o.text for o in output.outputs]))
        return {
            "prompt": prompt,
            "generated_text": generated_text,
        }



# For tensor_parallel_size > 1, we need to create placement groups for vLLM
# to use. Every actor has to have its own placement group.
def scheduling_strategy_fn():
    # One bundle per tensor parallel worker
    pg = ray.util.placement_group(
        [{
            "GPU": 2,
            "CPU": 1
        }] * (tensor_parallel_size),
        strategy="STRICT_PACK",
    )
    return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
        pg, placement_group_capture_child_tasks=True))


resources_kwarg: dict[str, Any] = {}
# if tensor_parallel_size == 1:
#     # For tensor_parallel_size == 1, we simply set num_gpus=1.
#     resources_kwarg["num_gpus"] = 1
# else:
# Otherwise, we have to set num_gpus=0 and provide
# a function that will create a placement group for
# each instance.
resources_kwarg["num_gpus"] = 1
resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

# Apply batch inference for all input data.
ds = ds.map_batches(
    LLMPredictor,
    # Set the concurrency to the number of LLM instances.
    concurrency=num_instances,
    # Specify the batch size for inference.
    batch_size=32,
    **resources_kwarg,
)

# # Peek first 10 results.
# # NOTE: This is for local testing and debugging. For production use case,
# # one should write full result out as shown below.
# outputs = ds.take(limit=10)
# for output in outputs:
#     prompt = output["prompt"]
#     generated_text = output["generated_text"]
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# Write inference output data out as Parquet files to S3.
# Multiple files would be written to the output destination,
# and each task would write one or more files separately.
#
# ds.write_parquet("s3://<your-output-bucket>")
ds.write_json("data/output_data.jsonl")
