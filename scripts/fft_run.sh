#!/bin/bash

####SBATCH --ntasks=1
####SBATCH --cpus-per-task=32
####SBATCH --mem=500G
####SBATCH --gres=gpu:8
####SBATCH --time=3-00:00:00
####SBATCH --partition=threeday
####SBATCH --job-name=tune-fft
####SBATCH --output=/home/imodoranu/workplace/projects/torchtune/scripts/logs/%x_%j.out   # %x = job name, %j = job ID
####SBATCH --error=/home/imodoranu/workplace/projects/torchtune/scripts/logs/%x_%j.err    # separate stderr log
####SBATCH --nodelist=research-common-13

mkdir -p ~/workplace/projects/torchtune/scripts/logs

source ~/.bashrc

conda activate tune

cd ~/workplace/projects/torchtune

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#MODEL_VERSION_SIZE="3.1-8B"
#MODEL_VERSION_SIZE="3.2-1B"
# MODEL_VERSION_SIZE="3.2-3B"
MODEL_VERSION_SIZE="2.5-7B"

CONFIG="recipes/configs/imodoranu/fft_multi_gpu.yaml"

#OPTIMIZER=adamw
#OPTIMIZER=frugal_micro_adam_torch_fsdp2
#OPTIMIZER=frugal_micro_adam_cuda_fsdp2
#OPTIMIZER=micro_adam_cuda_fsdp2
#OPTIMIZER=custom_ema_adamw
#OPTIMIZER=signsgd
#OPTIMIZER=trionosd

for OPTIMIZER in adamw; do #  signsgd; do
    for LR in 9e-5; do #  8e-5 7e-5; do # 2e-5 1e-5 9e-6 8e-6 7e-6 6e-6 5e-6; do # 5e-5 4e-5 3e-5 2e-5 1e-5 9e-6 8e-6 7e-6
#        CUDA_VISIBLE_DEVICES=0
        tune run \
            --nproc_per_node 8 \
            full_finetune_distributed \
            --config $CONFIG \
            model_version_size=${MODEL_VERSION_SIZE} \
            optimizer_name=${OPTIMIZER} \
            learning_rate=${LR} \
            weight_decay=0 \
            batch_size=1 \
            gradient_accumulation_steps=64 \
            dataset_split=train \
            compile=True
#            optimizer.trionosd.ns_lr=1e-3 \
#            optimizer.trionosd.ns_iters=1 \
#            optimizer.trionosd.ns_reg_U=200 \
#            optimizer.trionosd.ns_reg_V=200
    done
#    python3 ~/workplace/projects/torchtune/evaluation/main_scrape_eval_allocated.py \
#        --filter=${OPTIMIZER} \
#        --model_version_size=${MODEL_VERSION_SIZE} \
#        --exact=0 \
#        --shots=8
# python3 main_scrape_eval_allocated.py --filter=adamw --model_version_size=3.1-8B --exact=0 --shots=8
done