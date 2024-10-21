#!/usr/bin/env bash

export HF_HOME=/storage/brno2/home/xhorni20/.cache/huggingface

python scripts/download_dataset.py \
    --model_name_or_path="aaaaaaaaaaaaaa" \
    --output_dir="aaaaaaaaaaaaaaaaaaaaaa" \
	--dataset_name="librispeech_asr" \
	--dataset_config_name="clean" \
	--train_split_name="train.100" \
	--eval_split_name="validation" \
	--preprocessing_num_workers="1" \
    --trust_remote_code \