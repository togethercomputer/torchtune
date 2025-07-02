#!/bin/bash
clear

setuptune

cd $HF_HOME
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir Meta-Llama-3.1-8B-Instruct --ignore-patterns "original/consolidated.00.pth"
tune download meta-llama/Llama-3.2-1B-Instruct --output-dir /tmp/Llama-3.2-1B-Instruct --ignore-patterns "original/consolidated.00.pth"

python3 download_data.py --dataset=nvidia/OpenMathInstruct-2
