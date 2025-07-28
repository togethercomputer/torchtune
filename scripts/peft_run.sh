#!/bin/bash

clear

#MODEL_VERSION_SIZE="3.1-8B"
MODEL_VERSION_SIZE="3.2-1B"
# MODEL_VERSION_SIZE="3.2-3B"

CONFIG="recipes/configs/imodoranu/llama3_peft_multi_gpu.yaml"

OPTIMIZER=adamw
#OPTIMIZER=custom_ema_adamw
#OPTIMIZER=signsgd
#OPTIMIZER=trionosd

for LR in 3e-4; do
    tune run \
        --nproc_per_node 8 \
        lora_finetune_distributed \
        --config $CONFIG \
        model_version_size=${MODEL_VERSION_SIZE} \
        optimizer_name=${OPTIMIZER} \
        learning_rate=${LR} \
        batch_size=2 \
        gradient_accumulation_steps=32
done

#python3 ~/workplace/projects/torchtune/evaluation/main_scrape_eval_allocated.py \
#    --filter=${OPTIMIZER} \
#    --model_version_size=${MODEL_VERSION_SIZE} \
#    --exact=0 \
#    --shots=8
