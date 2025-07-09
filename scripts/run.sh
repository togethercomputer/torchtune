#!/bin/bash

clear

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

#MODEL="3.1_8B"
MODEL="3.2_1B"

CONFIG="recipes/configs/imodoranu/llama${MODEL}_full_ft_multi_gpu.yaml"

# tune run full_finetune_single_device --config "recipes/configs/imodoranu/llama${MODEL}_full_ft_single_gpu.yaml"

#OPTIMIZER=adamw
#OPTIMIZER=frugal_micro_adam_torch_fsdp2
#OPTIMIZER=frugal_micro_adam_cuda_fsdp2
#OPTIMIZER=micro_adam_cuda_fsdp2
#OPTIMIZER=custom_ema_adamw

for OPTIMIZER in adamw; do
    for LR in 1e-5 2e-5 3e-5; do # 5e-5 4e-5 3e-5 2e-5 1e-5 9e-6 8e-6 7e-6
        tune run \
            --nproc_per_node 8 \
            full_finetune_distributed \
            --config $CONFIG \
            optimizer_name=${OPTIMIZER} \
            learning_rate=${LR}
    done
done


#    for EMA_DECAY in 10 25 50 100; do
#        optimizer.custom_ema_adamw.ema_decay=${EMA_DECAY}
#        optimizer.frugal_micro_adam_torch_fsdp2.use_sign_sgd=0 \
#        optimizer.frugal_micro_adam_torch_fsdp2.normalization=none