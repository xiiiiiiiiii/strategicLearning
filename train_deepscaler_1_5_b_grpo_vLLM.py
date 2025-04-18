
# Steps to run:

# 1 - set accelerate config:
# accelerate config
# example input and output.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
# This machine                                                                                                                                                            
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?                                                                                                                                    
# multi-GPU                                                                                                                                                               
# How many different machines will you use (use more than 1 for multi-node training)? [1]:                                                                                
# Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:                                          
# Do you wish to optimize your script with torch dynamo?[yes/NO]:                                                                                                         
# Do you want to use DeepSpeed? [yes/NO]:                                                                                                                                 
# Do you want to use FullyShardedDataParallel? [yes/NO]:                                                                                                                  
# Do you want to use TensorParallel? [yes/NO]:                                                                                                                            
# Do you want to use Megatron-LM ? [yes/NO]:                                                                                                                              
# How many GPU(s) should be used for distributed training? [1]:2                                                                                                          
# What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:                                                                       
# Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------Do you wish to use mixed precision?
# no                                                                                                                                                                      
# accelerate configuration saved at /home/user/.cache/huggingface/accelerate/default_config.yaml  

# 2 - set wandb key to monitor training:
# export WANDB_API_KEY=...

# 3 - in first terminal session, start vLLM server on last CUDA device id, for 8 GPUs on a node, would be CUDA_VISIBLE_DEVICES=7 vllm serve --model your_model_name --tensor-parallel-size 1
# CUDA_VISIBLE_DEVICES=7 trl vllm-serve --model "agentica-org/DeepScaleR-1.5B-Preview"
# NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=7 trl vllm-serve --model "agentica-org/DeepScaleR-1.5B-Preview"

# 4 - in second terminal session, run the script, use num_processes = (num_total_gpus - 1) processes if using vLLM.
# accelerate launch --num_processes 7 train_deepscaler_1_5_b_grpo_vLLM.py 


import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

model_name = "agentica-org/DeepScaleR-1.5B-Preview"
output_dir="outputs/DeepScaleR_1_5_B_GRPO_tweak"
run_name="DeepScaleR_1_5_B_GRPO_tweak"

def load_jsonl(file_path):
    SYSTEM_PROMPT = "Let's think step by step and output the final answer within \\boxed{}."
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
        data = [{
            'prompt': f"{x['problem']} {SYSTEM_PROMPT}",
            **x
        } for x in data]
        return data

# Load the data
dataset = load_jsonl("./DeepScaleR-eval/grpo_tweak_dataset.jsonl")
print(f"Loaded {len(dataset)} records")

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

def single_correctness_reward_func(response: str, actual_answers: str) -> float:
    extracted_answer = extract_solution(response)
    return 1.0 if extracted_answer == actual_answers else 0.0

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    extracted_responses = [extract_solution(r) for r in completions]
    rewards = [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
    return rewards


# total steps: (num_samples * num_train_epochs) / gradient_accumulation_steps
# per_device_train_batch_size * num_generations should be divisible by num_processes.
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=2e-6,  # Slightly lower learning rate for the larger model
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    per_device_train_batch_size=1,  # Reduced batch size for larger model
    gradient_accumulation_steps=8,  # Increased gradient accumulation to compensate
    num_generations=7,
    max_prompt_length=1024,
    max_completion_length=16384,  # Increased max completion length
    num_train_epochs=16,
    save_steps=1,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
    use_vllm=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # torch_dtype=torch.bfloat16,
    device_map=None
) #.to("cuda")
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
