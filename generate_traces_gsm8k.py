import torch
import os
import re
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trace_generator import TraceGenerator, TraceGeneratorConfig
from trl import GRPOTrainer, GRPOConfig

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

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()
  
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
  
dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

model_name = "agentica-org/DeepScaleR-1.5B-Preview"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
output_file = os.path.join(data_dir, "gsm8k_traces.jsonl")

# Configure the trace generator
config = TraceGeneratorConfig(
    model_name_or_path=model_name,
    output_file=output_file,
    num_samples=100,  # Adjust as needed
    num_generations_per_prompt=16,
    max_prompt_length=256,
    max_completion_length=1024,
    temperature=1.0,
    use_vllm=True,
    torch_dtype="bfloat16",
    batch_size=8
)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=None
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Set the model's generation config to use left padding
model.generation_config.padding_side = 'left'

# Initialize the trace generator
trace_generator = TraceGenerator(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func
    ],
    config=config,
    dataset=dataset
)

# Generate traces
traces_df = trace_generator.generate_traces()
print(f"Generated {len(traces_df)} traces")
print(f"Traces saved to {output_file}")