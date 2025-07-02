#!/bin/bash

clear

#MODEL="3.1_8B"
MODEL="3.2_1B"

# tune run full_finetune_single_device --config "recipes/configs/imodoranu/llama${MODEL}_full_ft_single_gpu.yaml"

#for optimizer in "adamw" "frugal_micro_adam"; do
#    tune run \
#        --nproc_per_node 8 \
#        full_finetune_distributed \
#        --config "recipes/configs/imodoranu/llama${MODEL}_full_ft_multi_gpu.yaml" \
#        optimizer.name=$optimizer \
#        dataset.split=train_1M
#done


OPTIMIZER=adamw
#OPTIMIZER=frugal_micro_adam
#OPTIMIZER=frugal_micro_adam_fsdp2

LR=1e-5
tune run \
    --nproc_per_node 8 \
    full_finetune_distributed \
    --config "recipes/configs/imodoranu/llama${MODEL}_full_ft_multi_gpu.yaml" \
    optimizer_name=${OPTIMIZER} \
    learning_rate=${LR}
