#!/bin/bash

# This Bash script runs the Python script with arguments

# Run the Python script with command-line arguments
python layer_similarity.py --model_path "unsloth/gpt-oss-20b-unsloth-bnb-4bit" \
                      --dataset "openai/gsm8k" \
                      --dataset_column "question" \
                      --batch_size 4 \
                      --max_length 1024 \
                      --layers_to_skip 4 \
                      --dataset_size 2000 \
                      --dataset_subset "train" 
