#!/usr/bin/env bash

sleep 30
# Padded to max length
CUDA_VISIBLE_DEVICES="0" python pretrain_bart.py \
    --dataset_name="facebook/voxpopuli" \
    --dataset_config_name="en" \
    --model_name_or_path="facebook/bart-base" \
    --trust_remote_code \
    --torch_dtype="bfloat16" \
    --torch_compile \
    --preprocessing_num_workers="16" \
    --mlm_probability="0.2" \
    --max_seq_length="512" \
    --wandb_project="pretrain_seq2seq_bart" \
    --output_dir="./bart-base/training_denois-nopad/" \
    --dataloader_num_workers="8" \
	--dataloader_prefetch_factor="2" \
    --data_collator="denoising" \
    --overwrite_output_dir \
	--num_train_epochs="700" \
    --per_device_train_batch_size="96" \
	--per_device_eval_batch_size="96" \
    --gradient_accumulation_steps="1" \
    --learning_rate="3e-4" \
    --lr_scheduler_type="cosine_with_min_lr" \
	--lr_scheduler_kwargs="{\"min_lr\": 1e-6}" \
    --warmup_steps="1000" \
    --eval_strategy="epoch" \
    --eval_on_start \
    --logging_steps="10" \
    --include_tokens_per_second \
	--save_strategy="epoch" \
    --save_total_limit="1" \
    --bf16 \
    --do_train --do_eval \
    --report_to="wandb" \