#!/usr/bin/env bash
# python -m torch.distributed.launch --nproc_per_node 8 run_speech_recognition_seq2seq.py \
CUDA_VISIBLE_DEVICES="0" python run_speech_recognition_seq2seq.py \
	--dataset_name="librispeech_asr" \
	--model_name_or_path="./seq2seq_wav2vec2_bart-base" \
	--dataset_config_name="clean" \
	--train_split_name="train.100" \
	--eval_split_name="validation" \
	--output_dir="./seq2seq_wav2vec2_bart-base/training_librespeech" \
	--preprocessing_num_workers="16" \
	--length_column_name="input_length" \
	--overwrite_output_dir \
	--num_train_epochs="5" \
	--per_device_train_batch_size="64" \
	--per_device_eval_batch_size="64" \
	--gradient_accumulation_steps="1" \
	--learning_rate="3e-4" \
	--warmup_steps="400" \
	--evaluation_strategy="steps" \
	--text_column_name="text" \
	--save_strategy="epoch" \
	--eval_steps="400" \
	--logging_steps="10" \
	--save_total_limit="1" \
    --freeze_feature_encoder \
	--bf16 \
    --task="transcribe" \
	--group_by_length \
	--predict_with_generate \
	--do_train --do_eval \
	--do_lower_case \
    --trust_remote_code \
    --report_to="wandb" \

## REFERENCE:
	# --dataset_name="librispeech_asr" \
	# --model_name_or_path="./" \
	# --dataset_config_name="clean" \
	# --train_split_name="train.100" \
	# --eval_split_name="validation" \
	# --output_dir="./" \
	# --preprocessing_num_workers="16" \
	# --length_column_name="input_length" \
	# --overwrite_output_dir \
	# --num_train_epochs="5" \
	# --per_device_train_batch_size="8" \ * 8 devices
	# --per_device_eval_batch_size="8" \
	# --gradient_accumulation_steps="1" \
	# --learning_rate="3e-4" \
	# --warmup_steps="400" \
	# --evaluation_strategy="steps" \
	# --text_column_name="text" \
	# --save_steps="400" \
	# --eval_steps="400" \
	# --logging_steps="10" \
	# --save_total_limit="1" \
	# --freeze_feature_extractor \
	# --gradient_checkpointing \
	# --fp16 \
	# --group_by_length \
	# --predict_with_generate \
	# --do_train --do_eval \
	# --do_lower_case