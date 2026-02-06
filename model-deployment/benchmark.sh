#!/bin/bash

# Get dataset 

#wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json


# run bnehcmark
guidellm benchmark \
  --target "https://model-serve-qwen3-32b.impactsummit.nxtgen.cloud" \
  --max-seconds 15 \
  --data "prompt_tokens=256,output_tokens=128" \
  --rate-type "sweep"


#--target "https://model-serve-scale-route-vllm-scale.apps.cluster-9lksb.9lksb.sandbox1114.opentlc.com" \