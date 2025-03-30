# Strategic Learning for DeepScaler 1.5B

High Level: To solve AIME questions, find similar questions, train on similar questions with GRPO and then try solving original AIME question.

This repository contains tools to start completing high level goal above by analyzing performance of DeepScaler, generating reasoning traces on similar training problems with solutions, and training DeepScaler 1.5B with GRPO on training problems (without using tesing reasoning traces or solutions).

Heads up: repo uses git lfs to handle large data files!

## Main Files

### Analysis Scripts
- **aime_eval_analysis.ipynb**: Analyzes DeepScaler 1.5B Preview performance on AIME questions (using DeepScaler's team own eval data) and identifies questions ripe for improvement.
- **eval_check.ipynb**: Performs similar analysis using vLLM-generated traces on a subset of questions as verification.
- **deepscaler-matches-bm-25.ipynb**: Finds matches in DeepScaler's training dataset for questions identified as ripe for performance improvement.
- **match_gen_analysis.ipynb**: Analyzes DeepScaler 1.5B performance on the matched training dataset questions.

### Data Generation & Training
- **data_parallel_deepscaler_1_5_B_gen.py**: Generates traces with vLLM for training data.
- **train_deepscaler_1_5_b_grpo_vLLM.py**: Trains DeepScaler 1.5B with GRPO on questions from DeepScaler's dataset that are similar to AIME questions identified for improvement.

### Toy Example Supporting Scripts
- **train_grpo....py**: Simple GRPO training scripts on smaller models and shorter traces.

### Generated Data
- **deepscaler_eval_logs**: DeepScaler 1.5B Eval data from Deepscaler team itself.
- **deepscaler_eval_check**: Sample check of AIME problem reasoning traces.
- **top_87_ripe_0_DeepScaleR_1_5_B_results**: Reasoning traces generated with Deepscaler 1.5B using vLLM for first ripe question of AIME24.

## Server Setup

This section provides instructions for setting up and configuring servers for the Strategic Learning project.

### Server Requirements

- Ubuntu with NVIDIA GPUs (tested on single node with 1-8 H100 or H200)
- Installed CUDA drivers

### Hardware Notes

Performance tests reveal that newer hardware (A100, H100, H200) is generally more cost-effective, providing better performance per dollar and faster results.


### SSH Login

```bash
ssh -o StrictHostKeyChecking=no -i path/to/public/ssh/key.pem -p [server ssh port] [server user]@[server public ip]
```

### GPU Monitoring and Troubleshooting

```bash
# Check GPU status
nvidia-smi

# Live updating GPU status
watch -n 1 nvidia-smi
```

If `nvidia-smi` resets after bootup and you lose access to GPUs, a reboot often helps:
```bash
sudo reboot
```

### Initial Setup

```bash
git clone https://github.com/xiiiiiiiiii/strategicLearning.git
cd strategicLearning
```

### Installation Options

#### Option 1: Generation of Traces using vLLM only (no training) using uv (fastest and easiest)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
source .venv/bin/activate
uv pip freeze | grep vllm # check installation
```

#### Option 2: GRPO Training

Before installation, check for potential conflicts:
```bash
pip uninstall tensorflow -y  # Creates CUDA conflict with PyTorch
```

If the above command succeeds:
```bash
pip install accelerate "trl[vllm]" transformers datasets torch wandb
# For distributed training, add deepspeed:
pip install accelerate "trl[vllm]" transformers datasets torch wandb deepspeed
```

> **Important**: Installing `trl[vllm]` is critical. Simply installing `pip install trl vllm` will not install compatible versions.

If unable to uninstall tensorflow, see the Conda installation section below.

### Conda Installation (Alternative)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init
source ~/.bashrc
conda create -n py311 python=3.11 -y
conda activate py311
```

#### Troubleshooting Conda Accelerate Issues

If accelerate is not picking the right python environment with right version of accelerate:
```bash
conda deactivate
pip uninstall accelerate -y
conda activate base
pip uninstall accelerate -y
rm /home/[username]/.local/bin/accelerate
conda activate py311
```

### Accelerate Configuration (required for GRPO with TRL)

For GRPO training with TRL, set up accelerate:
```bash
accelerate config
```

#### Example Accelerate Configuration (Multi-GPU)
```
In which compute environment are you running?
This machine

Which type of machine are you using?
multi-GPU

How many different machines will you use (use more than 1 for multi-node training)? [1]:

Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:

Do you wish to optimize your script with torch dynamo?[yes/NO]:

Do you want to use DeepSpeed? [yes/NO]:

Do you want to use FullyShardedDataParallel? [yes/NO]:

Do you want to use TensorParallel? [yes/NO]:

Do you want to use Megatron-LM ? [yes/NO]:

How many GPU(s) should be used for distributed training? [1]:2

What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:

Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]:

Do you wish to use mixed precision?
no
```

#### Example DeepSpeed Configuration (Stage 3).  ONLY WORKS WITH stage 3! Stage 1 and 2 are ignored through code!
```bash
accelerate config
```

```
In which compute environment are you running?
This machine

Which type of machine are you using?
multi-GPU

How many different machines will you use (use more than 1 for multi-node training)? [1]:

Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:

Do you wish to optimize your script with torch dynamo?[yes/NO]:

Do you want to use DeepSpeed? [yes/NO]: yes

Do you want to specify a json file to a DeepSpeed config? [yes/NO]:

What should be your DeepSpeed's ZeRO optimization stage?
3

Where to offload optimizer states?
none

Where to offload parameters?
none

How many gradient accumulation steps you're passing in your script? [1]: 8

Do you want to use gradient clipping? [yes/NO]:

Do you want to save 16-bit model weights when using ZeRO Stage-3? [yes/NO]:

Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]:

Do you want to enable Mixture-of-Experts training (MoE)? [yes/NO]:

How many GPU(s) should be used for distributed training? [1]:7

Do you wish to use mixed precision?
no
```