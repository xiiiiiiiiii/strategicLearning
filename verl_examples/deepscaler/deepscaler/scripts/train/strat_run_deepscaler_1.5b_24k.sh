# Running:
# ./strat_run_deepscaler_1.5b_24k.sh
# Script will find next available experiment directory and use that.
# Script will find latest checkpoint in that experiment directory and use that.
# It will confirm the experiment name and model path before running. To skip confirmation, set SKIP_CONFIRMATION=true.
# SKIP_CONFIRMATION=true ./strat_run_deepscaler_1.5b_24k.sh

#!/bin/bash
set -x

# Find next available experiment directory
BASE_NAME="deepscaler-finetune-1.5b-24k"
CHECKPOINT_DIR="checkpoints/deepscaler-finetune"
index=0

while [ -d "${CHECKPOINT_DIR}/${BASE_NAME}_${index}" ]; do
    index=$((index + 1))
done

EXPERIMENT_NAME="${BASE_NAME}_${index}"
echo "Experiment output will be written to: ${CHECKPOINT_DIR}/${EXPERIMENT_NAME}"

# Default model path
DEFAULT_MODEL_PATH="agentica-org/DeepScaleR-1.5B-Preview"

# Check if previous experiments exist
if [ $index -gt 0 ]; then
    PREV_EXP_DIR="${CHECKPOINT_DIR}/${BASE_NAME}_$((index-1))"
    CHECKPOINT_PATTERN="${PREV_EXP_DIR}/actor/global_step_*"
    
    # Find the latest checkpoint
    LATEST_CHECKPOINT=""
    MAX_STEP=0
    
    for checkpoint in $CHECKPOINT_PATTERN; do
        if [ -d "$checkpoint" ]; then
            # Extract step number
            STEP=$(echo "$checkpoint" | sed -E 's/.*global_step_([0-9]+)/\1/')
            
            # Update if this checkpoint has a higher step number
            if [ "$STEP" -gt "$MAX_STEP" ]; then
                MAX_STEP=$STEP
                LATEST_CHECKPOINT=$checkpoint
            fi
        fi
    done
    
    # If we found a checkpoint, use it
    if [ -n "$LATEST_CHECKPOINT" ]; then
        MODEL_PATH="$LATEST_CHECKPOINT"
        echo "Using existing checkpoint: $MODEL_PATH"
    else
        MODEL_PATH="$DEFAULT_MODEL_PATH"
        echo "No checkpoints found. Using default model: $MODEL_PATH"
    fi
else
    MODEL_PATH="$DEFAULT_MODEL_PATH"
    echo "No previous experiments found. Using default model: $MODEL_PATH"
fi

# Ask for confirmation unless SKIP_CONFIRMATION is set
if [ "${SKIP_CONFIRMATION}" != "true" ]; then
    echo -e "\nTraining will use:"
    echo "  - Experiment name: ${EXPERIMENT_NAME}"
    echo "  - Model path: ${MODEL_PATH}"
    echo -e "\nPress ENTER to continue or Ctrl+C to abort..."
    read -r
    echo "Continuing with training..."
fi

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS

# export WANDB_API_KEY=...

# Need to specify custom reward function.

# Train over a single node, with 4 H100-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./train_${index}.parquet \
    data.val_files=./aime.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=24576 \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.n_val=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='deepscaler-finetune' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=2 \
    trainer.test_freq=2 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=10 "${@:1}"
