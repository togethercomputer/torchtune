#!/bin/bash

clear

#MODEL="3.1_8B"
MODEL="3.2_1B"

# tune run full_finetune_single_device --config "recipes/configs/imodoranu/llama${MODEL}_full_ft_single_gpu.yaml"

OPTIMIZER=adamw
#OPTIMIZER=frugal_micro_adam_fsdp2

for LR in 1e-5 2e-5 3e-5; do
    tune run \
        --nproc_per_node 8 \
        full_finetune_distributed \
        --config "recipes/configs/imodoranu/llama${MODEL}_full_ft_multi_gpu.yaml" \
        optimizer_name=${OPTIMIZER} \
        learning_rate=${LR}
done
