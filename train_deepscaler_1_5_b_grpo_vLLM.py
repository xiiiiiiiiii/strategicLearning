
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

model_name = "agentica-org/DeepScaleR-1.5B-Preview"
output_dir="outputs/DeepScaleR_1_5_B_GRPO_tweak"
run_name="DeepScaleR_1_5_B_GRPO_tweak"

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# Load the data
dataset = load_jsonl("./DeepScaleR-eval/grpo_tweak_dataset.jsonl")
print(f"Loaded {len(dataset)} records")

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


training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=2e-6,  # Slightly lower learning rate for the larger model
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=10,
    per_device_train_batch_size=8,  # Reduced batch size for larger model
    gradient_accumulation_steps=1,  # Increased gradient accumulation to compensate
    num_generations=8,
    max_prompt_length=256,
    max_completion_length=1024,  # Increased max completion length
    num_train_epochs=1,
    save_steps=10,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
    use_vllm=True
)
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

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=correctness_reward_func,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
