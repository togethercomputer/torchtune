#!/bin/bash

clear

# tune run full_finetune_single_device --config recipes/configs/imodoranu/llama3.1_8B_full_single_device.yaml

tune run --nproc_per_node 4 full_finetune_distributed --config recipes/configs/imodoranu/llama3.1_8B_full_distributed.yaml