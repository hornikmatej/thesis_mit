#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="0" python run_speech_recognition_ctc.py \
	--dataset_name="facebook/voxpopuli" \
	--model_name_or_path="facebook/wav2vec2-base-10k-voxpopuli-ft-en" \
	--dataset_config_name="en" \
	--train_split_name="train" \
	--eval_split_name="validation" \
	--test_split_name="test" \
	--output_dir="./wav2vec2-base_10k-ft-en-vox" \
	--preprocessing_num_workers="4" \
	--dataloader_num_workers="16" \
	--dataloader_prefetch_factor="2" \
	--overwrite_output_dir \
	--num_train_epochs="5" \
	--per_device_train_batch_size="96" \
	--per_device_eval_batch_size="96" \
	--gradient_accumulation_steps="1" \
	--learning_rate="1e-4" \
	--warmup_steps="300" \
	--eval_strategy="steps" \
	--text_column_name="normalized_text" \
	--save_strategy="epoch" \
	--eval_steps="1000" \
    --eval_on_start \
	--logging_steps="10" \
	--layerdrop="0.0" \
	--save_total_limit="1" \
	--freeze_feature_encoder \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” \
	--bf16 \
	--do_train --do_eval --do_predict \
    --trust_remote_code \
    --report_to="wandb" \
	--wandb_project="ctc" \
	--sclite_path="/home/azureuser/media-disk/mh_dp/SCTK/bin/sclite" \


sleep 30

CUDA_VISIBLE_DEVICES="0" python run_speech_recognition_ctc.py \
	--dataset_name="facebook/voxpopuli" \
	--model_name_or_path="facebook/wav2vec2-base-100k-voxpopuli" \
	--dataset_config_name="en" \
	--train_split_name="train" \
	--eval_split_name="validation" \
	--test_split_name="test" \
	--output_dir="./wav2vec2-base_100k-vox" \
	--preprocessing_num_workers="4" \
	--dataloader_num_workers="16" \
	--dataloader_prefetch_factor="2" \
	--overwrite_output_dir \
	--num_train_epochs="5" \
	--per_device_train_batch_size="96" \
	--per_device_eval_batch_size="96" \
	--gradient_accumulation_steps="1" \
	--learning_rate="1e-4" \
	--warmup_steps="300" \
	--eval_strategy="steps" \
	--text_column_name="normalized_text" \
	--save_strategy="epoch" \
	--eval_steps="1000" \
    --eval_on_start \
	--logging_steps="10" \
	--layerdrop="0.0" \
	--save_total_limit="1" \
	--freeze_feature_encoder \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” \
	--bf16 \
	--do_train --do_eval --do_predict \
    --trust_remote_code \
    --report_to="wandb" \
	--wandb_project="ctc" \
	--sclite_path="/home/azureuser/media-disk/mh_dp/SCTK/bin/sclite" \