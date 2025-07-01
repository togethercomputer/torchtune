#!/bin/bash
clear

MODEL="meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR="${HF_HOME}/Llama-3.2-1B-Instruct"

echo "🚀 Downloading '${MODEL}' to '${OUTPUT_DIR}'"
tune download $MODEL --output-dir $OUTPUT_DIR --ignore-patterns "original/consolidated.00.pth"