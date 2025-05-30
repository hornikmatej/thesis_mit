#!/usr/bin/env bash
# python -m torch.distributed.launch --nproc_per_node 8 run_speech_recognition_seq2seq.py \
CUDA_VISIBLE_DEVICES="0" python run_speech_recognition_seq2seq.py \
	--dataset_name="facebook/voxpopuli" \
	--model_name_or_path="./seq2seq_wav2vec2_bart-base" \
	--dataset_config_name="en" \
	--train_split_name="train" \
	--eval_split_name="validation" \
	--test_split_name="test" \
	--output_dir="./seq2seq_wav2vec2_bart-base/training_voxpopuli" \
	--preprocessing_num_workers="1" \
	--dataloader_num_workers="16" \
	--dataloader_prefetch_factor="2" \
	--length_column_name="input_length" \
	--overwrite_output_dir \
	--num_train_epochs="5" \
	--per_device_train_batch_size="96" \
	--per_device_eval_batch_size="96" \
	--gradient_accumulation_steps="1" \
	--learning_rate="1e-4" \
	--warmup_steps="100" \
	--eval_strategy="steps" \
	--text_column_name="normalized_text" \
	--save_strategy="no" \
	--eval_steps="1000" \
	--logging_steps="10" \
	--save_total_limit="1" \
    --freeze_feature_encoder \
	--bf16 \
    --task="transcribe" \
	--predict_with_generate \
	--do_train --do_eval --do_predict \
	--do_lower_case \
    --trust_remote_code \
    --report_to="wandb" \
	--sclite_path="/home/azureuser/media-disk/mh_dp/SCTK/bin/sclite" \
	--wandb_project="seq2seq_encoder-decoder_fe" \
	--cache_dir="/home/azureuser/media-disk/mh_dp/preprocessed_dataset_voxpopuli" \	
