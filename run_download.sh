#!/usr/bin/env bash

export HF_HOME=/storage/brno2/home/xhorni20/.cache/huggingface

python download_dataset.py \
    --model_name_or_path="aaaaaaaaaaaaaa" \
    --output_dir="aaaaaaaaaaaaaaaaaaaaaa" \
	--dataset_name="facebook/voxpopuli" \
	--dataset_config_name="en" \
	--train_split_name="train" \
	--eval_split_name="validation" \
	--preprocessing_num_workers="1" \
    --trust_remote_code \