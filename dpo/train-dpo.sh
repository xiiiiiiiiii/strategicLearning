
MODEL_PATH="agentica-org/DeepScaleR-1.5B-Preview"
EXPERIMENT_NAME="deepscaler-dpo-finetune-1.5b-24k"

# Light-R1 DPO used 360-LLaMA-Factory directly
deepspeed --num_gpus=2 src/train.py \
    --stage dpo \
    --do_train \
    --max_steps -1 \
    --model_name_or_path $MODEL_PATH \
    --template qwen \
    --dataset r1-similar-ripe-dpo \
    --preprocessing_num_workers 2 \
    --finetuning_type full \
    --sequence_parallel_size 2 \
    --gradient_checkpointing True \
    --flash_attn auto  \
    --pref_beta 0.3 \
    --pref_loss nca_pair \
    --cache_dir .cache \
    --overwrite_cache \
    --cutoff_len 24576 \
    --output_dir $EXPERIMENT_NAME \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type constant \
    --save_strategy steps \
    --save_steps 10 \
    --logging_steps 1 \
    --warmup_ratio 0.0 \
    --save_total_limit 200 \
    --learning_rate 5e-7 \
    --save_only_model True \
    --num_train_epochs 3.0 \
    --plot_loss \
    --seed 42 \
    --do_eval false \
    --deepspeed ./examples/deepspeed/ds_z0_config.json \
    --report_to wandb \
    --overwrite_output_dir \
    --ddp_timeout 180000000 \