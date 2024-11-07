#!/bin/bash

# Set the CUDA device to use
CUDA_VISIBLE_DEVICES=0 

# Run the inference script with specified parameters
python inference_vlm.py \
    --model_path "liuhaotian/llava-v1.6-mistral-7b"\
    --output_folder outputs/ \
    --temperature 0.2 \
    --top_p 0.9 \
    --repetition_penalty 1.0 \
    --max_new_tokens 512 \
    --device "cuda" \
    --task "all" \
    --w_reason \
    --mode "vtqa"