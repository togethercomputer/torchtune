#!/bin/bash

clear

#MODEL="3.1_8B"
MODEL="3.2_1B"

# tune run full_finetune_single_device --config "recipes/configs/imodoranu/llama${MODEL}_full_ft_single_gpu.yaml"

#OPTIMIZER=adamw
#OPTIMIZER=frugal_micro_adam_fsdp2
OPTIMIZER=micro_adam_cuda_fsdp2

for LR in 9e-6 8e-6 7e-6 6e-6 5e-6; do
    tune run \
        --nproc_per_node 8 \
        full_finetune_distributed \
        --config "recipes/configs/imodoranu/llama${MODEL}_full_ft_multi_gpu.yaml" \
        optimizer_name=${OPTIMIZER} \
        learning_rate=${LR} \
        optimizer.micro_adam_cuda_fsdp2.decay_all_weights=1
done


#        optimizer.frugal_micro_adam_fsdp2.use_sign_sgd=0 \
#        optimizer.frugal_micro_adam_fsdp2.normalization=none