#!/usr/bin/env bash
# -m torch.distributed.launch \
# 	--nproc_per_node 8
	
CUDA_VISIBLE_DEVICES="0" python run_speech_recognition_ctc.py \
	--dataset_name="librispeech_asr" \
	--model_name_or_path="facebook/wav2vec2-base" \
	--dataset_config_name="clean" \
	--train_split_name="train.100" \
	--eval_split_name="validation" \
	--output_dir="./wav2vec2base-librispeech-clean-100h" \
	--preprocessing_num_workers="2" \
	--overwrite_output_dir \
	--num_train_epochs="3" \
	--per_device_train_batch_size="64" \
	--per_device_eval_batch_size="64" \
	--gradient_accumulation_steps="1" \
	--learning_rate="3e-4" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--text_column_name="text" \
	--save_strategy="epoch" \
	--eval_steps="100" \
	--logging_steps="1" \
	--layerdrop="0.0" \
	--save_total_limit="1" \
	--freeze_feature_encoder \
	--gradient_checkpointing \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” \
	--bf16 \
	--group_by_length \
	--do_train --do_eval \
    --trust_remote_code \
    --report_to="wandb" \