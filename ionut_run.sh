#!/bin/bash

clear

# tune run full_finetune_single_device --config recipes/configs/imodoranu/llama3.1_8B_full_ft_single_gpu.yaml

for optimizer in "adamw" "frugal_micro_adam"; do
    tune run \
        --nproc_per_node 8 \
        full_finetune_distributed \
        --config recipes/configs/imodoranu/llama3.1_8B_full_ft_multi_gpu.yaml \
        optimizer.name=$optimizer \
        dataset.split=train_1M
done

#tune run \
#    --nproc_per_node 8 \
#    full_finetune_distributed \
#    --config recipes/configs/imodoranu/llama3.1_8B_full_ft_multi_gpu.yaml \
#    optimizer.name=adamw