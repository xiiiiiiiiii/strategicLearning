# Trace Generator

A utility for generating model completions with reward information by leveraging the GRPOTrainer infrastructure.

## Overview

The `TraceGenerator` class inherits from `GRPOTrainer` to reuse its efficient generation and reward computation mechanisms, while focusing solely on trace generation (without training). This tool allows you to:

1. Generate multiple completions for each prompt in a dataset
2. Compute rewards for each completion using one or more reward functions
3. Save the results in a structured format for later analysis or use in training

## Installation

Ensure you have the core dependencies:

```bash
pip install transformers datasets torch accelerate pandas tqdm
```

For faster generation, vLLM is supported but optional:

```bash
pip install vllm
```

## Multiple Reward Functions

You can combine multiple reward functions with optional weights:

```python
# Define multiple reward functions for math solutions
def length_reward(completions, **kwargs):
    # Rewards longer, more detailed solutions
    return [float(min(len(completion), 1000)) / 1000 for completion in completions]

def correctness_reward(completions, **kwargs):
    # Rewards solutions that follow the correct format and contain an answer
    return [1.0 if "the answer is" in c.lower() and any(char.isdigit() for char in c) else 0.2 for c in completions]

def step_by_step_reward(completions, **kwargs):
    # Rewards solutions that show clear step-by-step reasoning
    return [min(1.0, 0.2 * completion.count("\n")) for completion in completions]

# Configure with multiple reward functions and weights
config = TraceGeneratorConfig(
    model_name_or_path="agentica-org/DeepScaleR-1.5B-Preview",
    dataset_name_or_path="openai/gsm8k",
    reward_funcs=[correctness_reward, step_by_step_reward, length_reward],
    reward_weights=[0.5, 0.3, 0.2],  # 50% correctness, 30% steps, 20% length
    num_samples=30,
    max_completion_length=1024,
    use_model_default_chat_template=True
)
```

## Advanced Features

- **Accelerated Generation**: Use vLLM for faster generation with the `--use_vllm` flag
- **Chat Format Support**: Apply model-specific chat templates with `--use_chat_template`
- **Distributed Processing**: Leverages the Accelerate library for multi-GPU processing
- **Batched Processing**: Control batch size for efficient processing

## Output Format

The generated traces are saved in JSONL format with the following structure:

```json
{
  "prompt": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four eggs. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
  "completion": "<reasoning>\nJanet's ducks lay 16 eggs per day.\nShe eats 3 eggs for breakfast each morning.\nShe uses 4 eggs to bake muffins for her friends daily.\nSo the total number of eggs she consumes is 3 + 4 = 7 eggs per day.\nThe remaining eggs are sold at the farmers' market.\nThe number of eggs she sells = 16 - 7 = 9 eggs per day.\nEach egg sells for $2.\nSo her daily income from selling eggs is 9 × $2 = $18 per day.\n</reasoning>\n<answer>\n18\n</answer>",
  "reward": 0.95
}
```

When using multiple reward functions, the final reward is a weighted combination of all individual rewards.