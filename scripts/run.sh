#!/bin/bash

clear

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_VERSION_SIZE="3.1-8B"
# MODEL_VERSION_SIZE="3.2-1B"
# MODEL_VERSION_SIZE="3.2-3B"

CONFIG="recipes/configs/imodoranu/llama3_full_ft_multi_gpu.yaml"

#OPTIMIZER=adamw
#OPTIMIZER=frugal_micro_adam_torch_fsdp2
#OPTIMIZER=frugal_micro_adam_cuda_fsdp2
#OPTIMIZER=micro_adam_cuda_fsdp2
#OPTIMIZER=custom_ema_adamw
#OPTIMIZER=signsgd

for OPTIMIZER in frugal_micro_adam_torch_fsdp2; do
    for LR in 2e-5; do #  2e-5 1e-5 9e-6 8e-6 7e-6 6e-6 5e-6; do # 5e-5 4e-5 3e-5 2e-5 1e-5 9e-6 8e-6 7e-6
        tune run \
            --nproc_per_node 8 \
            full_finetune_distributed \
            --config $CONFIG \
            model_version_size=${MODEL_VERSION_SIZE} \
            optimizer_name=${OPTIMIZER} \
            learning_rate=${LR} \
            weight_decay=0
    done
done

#    for EMA_DECAY in 10 25 50 100; do
#        optimizer.custom_ema_adamw.ema_decay=${EMA_DECAY}
#        optimizer.frugal_micro_adam_torch_fsdp2.use_sign_sgd=0 \
#        optimizer.frugal_micro_adam_torch_fsdp2.normalization=none