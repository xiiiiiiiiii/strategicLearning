# Tested with 2 & 4 GPUs ??? TODO: check 4 GPUs.

MODEL_PATH="agentica-org/DeepScaleR-1.5B-Preview"
EXPERIMENT_NAME="deepscaler-sft-1.5b"

nproc_per_node=2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./train.parquet \
    data.val_files=./train.parquet \
    data.max_length=24576 \
    data.train_batch_size=1 \
    data.micro_batch_size=2 \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    model.partial_pretrain="${MODEL_PATH}" \
    trainer.default_local_dir=./model_save_path \
    trainer.project_name='deepscaler-finetune' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=10 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null
