#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TraceGenerator: A class for generating and evaluating traces with rewards
that inherits from GRPOTrainer for efficient reuse of generation and reward logic.
"""

import os
import torch
import json
import pandas as pd
from typing import Union, List, Optional, Any, Callable, Dict
from dataclasses import dataclass, field

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from datasets import Dataset, IterableDataset
from tqdm import tqdm
from accelerate import Accelerator

from trl import GRPOTrainer, GRPOConfig
from typing import Callable as RewardFunc
from trl.data_utils import maybe_apply_chat_template
from trl.import_utils import is_vllm_available

# Check if vLLM is available and import if needed
if is_vllm_available():
    from vllm import LLM, SamplingParams

@dataclass
class TraceGeneratorConfig:
    """Configuration for trace generation"""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dataset_name_or_path: str = field(
        metadata={"help": "Path to dataset or dataset identifier from huggingface.co/datasets"}
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use"}
    )
    reward_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to reward model or identifier from huggingface.co/models"}
    )
    reward_funcs: Optional[List[Callable]] = field(
        default=None,
        metadata={"help": "List of reward functions to evaluate generated completions. Each function should accept parameters like 'completions', 'prompts', etc. and return a list of float rewards."}
    )
    output_file: str = field(
        default="traces.jsonl",
        metadata={"help": "Output file to save traces"}
    )
    num_samples: Optional[int] = field(
        default=100,
        metadata={"help": "Number of prompts/questions to process from the dataset. For each prompt, the script will generate num_generations_per_prompt different completions. Set to None or 0 to use all samples in the dataset."}
    )
    num_generations_per_prompt: int = field(
        default=10,
        metadata={"help": "Number of different completions to generate for each prompt/question. Higher values provide more diverse outputs for the same prompt."}
    )
    max_prompt_length: Optional[int] = field(
        default=8192,
        metadata={"help": "Maximum prompt length in tokens"}
    )
    max_completion_length: int = field(
        default=8192,
        metadata={"help": "Maximum completion length in tokens"}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Generation temperature"}
    )
    use_vllm: bool = field(
        default=False,
        metadata={"help": "Whether to use vLLM for generation"}
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={"help": "Torch dtype for model loading. Use string format like 'float16', 'bfloat16', 'float32', etc. or 'auto' to automatically determine dtype"}
    )
    use_model_default_chat_template: bool = field(
        default=False,
        metadata={"help": "Whether to apply the model's built-in chat template to prompts. Each model (Qwen, DeepScaleR, etc.) has its own format with different markers (e.g., <|im_start|>, <｜User｜>). When enabled, the script automatically formats prompts using the template registered in the model's Hugging Face configuration."}
    )
    batch_size: int = field(
        default=8, 
        metadata={"help": "Number of prompts to process in parallel. Higher values are more efficient but require more memory. This is an optimization parameter and doesn't affect the final output."}
    )

class TraceGenerator(GRPOTrainer):
    """
    A class that generates traces and computes rewards for them.
    Inherits from GRPOTrainer to reuse the generation and reward computation logic.
    
    This class focuses specifically on generating completions and computing
    their rewards without the training aspects of GRPOTrainer.
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        config: Optional[TraceGeneratorConfig] = None,
        dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
    ):
        """
        Initialize the TraceGenerator.
        
        Args:
            model: The model to use for generation.
            reward_funcs: The reward functions to use for evaluation.
            config: Configuration for the trace generator.
            dataset: The dataset to generate traces from.
            processing_class: The tokenizer for the model.
        """
        # Create a GRPOConfig from the TraceGeneratorConfig
        if config is None:
            config = TraceGeneratorConfig()
            
        grpo_args = GRPOConfig(
            f"{model if isinstance(model, str) else model.config._name_or_path}-traces"
        )
        
        # Transfer parameters from TraceGeneratorConfig to GRPOConfig
        grpo_args.max_prompt_length = config.max_prompt_length
        grpo_args.max_completion_length = config.max_completion_length
        grpo_args.num_generations = config.num_generations_per_prompt
        grpo_args.temperature = config.temperature
        grpo_args.use_vllm = config.use_vllm
        grpo_args.log_completions = True
        
        # Initialize the parent class
        super().__init__(
            model=model,
            processing_class=processing_class,
            reward_funcs=reward_funcs,
            args=grpo_args,
            train_dataset=dataset,
        )
        
        # Store the trace generator config
        self.trace_config = config
        
    def generate_traces(self, dataset=None, num_samples=None, output_file=None):
        """
        Generate traces for the given dataset and compute rewards.
        
        Args:
            dataset: The dataset to generate traces for. If None, uses the dataset from initialization.
            num_samples: Number of samples to process. If None, uses the value from config.
            output_file: Where to save the traces. If None, uses the value from config.
            
        Returns:
            A DataFrame containing the generated traces and their rewards.
        """
        # Use parameters from initialization if not provided
        dataset = dataset or self.train_dataset
        num_samples = num_samples or self.trace_config.num_samples
        output_file = output_file or self.trace_config.output_file
        
        if dataset is None:
            raise ValueError("No dataset provided for trace generation")
        
        # Limit the dataset if needed
        if num_samples and num_samples > 0:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        # Initialize accelerator for distributed processing if needed
        accelerator = Accelerator()
        
        all_traces = []
        batch_size = self.trace_config.batch_size
        
        # Open file for streaming if output_file is specified and this is the main process
        output_file_handle = None
        if output_file and accelerator.is_main_process:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            output_file_handle = open(output_file, 'w', encoding='utf-8')
            
        # Process the dataset in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating traces"):
            batch = dataset[i:i+batch_size]
            
            # Reuse the _prepare_inputs method from GRPOTrainer to generate completions and compute rewards
            processed_inputs = self._prepare_inputs(batch)
            
            # Extract the data we want to save
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in batch]
            completions_text = self.processing_class.batch_decode(
                processed_inputs["completion_ids"], 
                skip_special_tokens=True
            )
            
            # Get the original advantages which contain the normalized rewards
            rewards = processed_inputs["advantages"].cpu().numpy()
            
            # Create trace entries and stream to file if specified
            batch_traces = []
            for prompt, completion, reward in zip(prompts_text, completions_text, rewards):
                trace = {
                    "prompt": prompt,
                    "completion": completion,
                    "reward": float(reward),
                }
                batch_traces.append(trace)
                
                # Stream save each trace if output file is specified
                if output_file_handle and accelerator.is_main_process:
                    output_file_handle.write(json.dumps(trace) + '\n')
                    output_file_handle.flush()  # Ensure it's written immediately
            
            # Collect all traces for the return value
            all_traces.extend(batch_traces)
            
        # Gather traces from all processes if using distributed setup
        if accelerator.num_processes > 1:
            all_traces = accelerator.gather_for_metrics(all_traces)
        
        # Close the output file if it was opened
        if output_file_handle:
            output_file_handle.close()
            print(f"Saved {len(all_traces)} traces to {output_file}")
        
        # Create a DataFrame from the traces
        traces_df = pd.DataFrame(all_traces)
        return traces_df