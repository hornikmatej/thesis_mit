#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence speech recognition.
"""

import logging
import os
import io
import aiohttp

import datasets
import evaluate
import wandb
import torch
import torch._dynamo


from tqdm import tqdm
from datasets import DatasetDict, load_dataset, DownloadConfig


from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from config import get_settings
from src.dataclass_args import ModelArguments, DataTrainingArguments
from src.custom_trainer import DebugSeq2SeqTrainer
from src.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from src.logger_setup import setup_logger
from src.utils import count_parameters, count_all_parameters, ProfCallback
import soundfile as sf

settings = get_settings()
logger = logging.getLogger(__name__)
torch._dynamo.config.suppress_errors = True

download_config = DownloadConfig(
    resume_download=True,
    max_retries=3,  # Optionally increase the number of retries
    storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
)

# Monitoring
WANDB_KEY = settings.wandb_token.get_secret_value()


def main():
    # 1. Parse input arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # add wandb run name to training_args
    training_args.run_name = f"{data_args.dataset_name}_{data_args.dataset_config_name}_split-{data_args.train_split_name}_wav2vec2-bart_bs{training_args.per_device_train_batch_size}_lr{training_args.learning_rate}_ep{training_args.num_train_epochs}"
    WANDB_PROJECT = data_args.wandb_project

    # 2. Setup logging
    log_level = setup_logger(training_args)

    if log_level != logging.DEBUG:
        run: wandb.Run = None
        if "wandb" in training_args.report_to:
            wandb.login(key=WANDB_KEY)
            run = wandb.init(
                project=WANDB_PROJECT,
                job_type="training",
                anonymous="allow",
                name=training_args.run_name,
            )
    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.warning(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load dataset
    raw_datasets = DatasetDict()

    if model_args.cache_dir is not None:
        logger.warning(f"Using cache dir {model_args.cache_dir} for the datasets.")
        vectorized_datasets = datasets.load_from_disk(model_args.cache_dir)
    else:
        if training_args.do_train:
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=data_args.train_split_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                num_proc=(
                    data_args.preprocessing_num_workers
                    if not data_args.streaming
                    else None
                ),
                streaming=data_args.streaming,
                download_config=download_config,
            )

        if training_args.do_eval:
            raw_datasets["eval"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=data_args.eval_split_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                num_proc=(
                    data_args.preprocessing_num_workers
                    if not data_args.streaming
                    else None
                ),
                streaming=data_args.streaming,
                download_config=download_config,
            )
        if training_args.do_predict:
            raw_datasets["test"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=data_args.test_split_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                num_proc=(
                    data_args.preprocessing_num_workers
                    if not data_args.streaming
                    else None
                ),
                streaming=data_args.streaming,
                download_config=download_config,
            )

        if (
            data_args.audio_column_name
            not in next(iter(raw_datasets.values())).column_names
        ):
            raise ValueError(
                f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
                "Make sure to set `--audio_column_name` to the correct audio column - one of "
                f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
            )

        if (
            data_args.text_column_name
            not in next(iter(raw_datasets.values())).column_names
        ):
            raise ValueError(
                f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
                "Make sure to set `--text_column_name` to the correct text column - one of "
                f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
            )
    logger.warning(f"Loaded {data_args.dataset_name} dataset")

    # 5. Load pretrained model, tokenizer, and feature extractor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # SpecAugment for whisper models
    if getattr(config, "model_type", None) == "whisper":
        config.update({"apply_spec_augment": model_args.apply_spec_augment})

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        (
            model_args.feature_extractor_name
            if model_args.feature_extractor_name
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="sdpa",
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    num_params_encoder, num_params_decoder, num_params_total = count_parameters(model)
    logger.warning(
        f"Number of trainable parameters - Encoder: {num_params_encoder:,}, Decoder: {num_params_decoder:,}, Total: {num_params_total:,}"
    )

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if model_args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    if (
        hasattr(model.generation_config, "is_multilingual")
        and model.generation_config.is_multilingual
    ):
        # We only need to set the language and task ids in a multilingual setting
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task)
        model.generation_config.language = data_args.language
        model.generation_config.task = data_args.task
    elif data_args.language is not None:
        raise ValueError(
            "Setting language token for an English-only checkpoint is not permitted. The language argument should "
            "only be set for multilingual checkpoints."
        )

    if model_args.forced_decoder_ids is not None:
        logger.warning(
            "The use of `forced_decoder_ids` is deprecated and will be removed in v4.41."
            "Please use the `language` and `task` arguments instead"
        )
        model.generation_config.forced_decoder_ids = model_args.forced_decoder_ids
    else:
        model.generation_config.forced_decoder_ids = None
        model.config.forced_decoder_ids = None

    if model_args.suppress_tokens is not None:
        logger.warning(
            "The use of `suppress_tokens` is deprecated and will be removed in v4.41."
            "Should you need `suppress_tokens`, please manually set them in the fine-tuning script."
        )
        model.generation_config.suppress_tokens = model_args.suppress_tokens
    logger.warning(f"Loaded model {model_args.model_name_or_path}")

    # 6. Resample speech dataset if necessary
    if model_args.cache_dir is None:
        dataset_sampling_rate = (
            next(iter(raw_datasets.values()))
            .features[data_args.audio_column_name]
            .sampling_rate
        )
        if dataset_sampling_rate != feature_extractor.sampling_rate:
            raw_datasets = raw_datasets.cast_column(
                data_args.audio_column_name,
                datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
            )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = (
        data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    )
    min_input_length = (
        data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    )
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    do_lower_case = data_args.do_lower_case
    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0) > 0
    )
    if model_args.cache_dir is None:
        if data_args.max_train_samples is not None and training_args.do_train:
            if data_args.streaming:
                raw_datasets["train"] = raw_datasets["train"].take(
                    data_args.max_train_samples
                )
            else:
                raw_datasets["train"] = raw_datasets["train"].select(
                    range(data_args.max_train_samples)
                )

        if data_args.max_eval_samples is not None and training_args.do_eval:
            if data_args.streaming:
                raw_datasets["eval"] = raw_datasets["eval"].take(
                    data_args.max_eval_samples
                )
            else:
                raw_datasets["eval"] = raw_datasets["eval"].select(
                    range(data_args.max_eval_samples)
                )
        # WARNING: Not needed, since i was take the whole dataset
        # if training_args.do_predict:
        #     if data_args.streaming:
        #         raw_datasets["test"] = raw_datasets["test"].take(
        #             data_args.max_predict_samples
        #         )
        #     else:
        #         raw_datasets["test"] = raw_datasets["test"].select(
        #             range(data_args.max_predict_samples)
        #         )

        def prepare_dataset(batch):
            # process audio
            sample = batch[audio_column_name]
            inputs = feature_extractor(
                sample["array"],
                sampling_rate=sample["sampling_rate"],
                return_attention_mask=forward_attention_mask,
            )
            # process audio length
            batch[model_input_name] = inputs.get(model_input_name)[0]
            batch["input_length"] = len(sample["array"])
            if forward_attention_mask:
                batch["attention_mask"] = inputs.get("attention_mask")[0]

            # process targets
            input_str = (
                batch[text_column_name].lower()
                if do_lower_case
                else batch[text_column_name]
            )
            batch["labels"] = tokenizer(input_str).input_ids
            if log_level == logging.DEBUG:
                logger.debug("------ MODEL CONFIG ------")
                logger.debug(f"{model.config.decoder_start_token_id=}")
                logger.debug(f"{model.config.pad_token_id=}")
                logger.debug(f"{model.config.eos_token_id=}")
                logger.debug(f"Original audio array length: {len(sample['array'])}")
                logger.debug(f"Audio array start: {sample['array'][:10]}")
                # Log feature extraction details
                logger.debug(
                    f"Extracted features length: {len(inputs.get(model_input_name)[0])}"
                )
                logger.debug(
                    f"Extracted features start: {inputs.get(model_input_name)[0][:10]}"
                )
                tokens = tokenizer(input_str)
                logger.debug(f"Original text: {input_str}")
                logger.debug(f"Tokenized input_ids: {tokens.input_ids}")
                logger.debug(f"Decoded tokens: {tokenizer.decode(tokens.input_ids)}")
                logger.debug("------ Debugging ------")
                logger.debug(f"Sample: {sample}")
                logger.debug(f"After ft_extractor: {inputs}")
                logger.debug(f"Batch: {batch}")
                logger.debug(f"Model input name: {model_input_name}")
                logger.debug(
                    f"Start decoder token: {model.config.decoder_start_token_id}"
                )
                logger.debug("------ END Debugging ------")
                # Save the audio sample for debugging
                audio_path = os.path.join(
                    training_args.output_dir, "debug", f"{batch['id']}.mp3"
                )
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                with io.BytesIO() as buffer:
                    sf.write(
                        buffer,
                        batch["input_values"],
                        batch["audio"]["sampling_rate"],
                        format="MP3",
                    )
                    with open(audio_path, "wb") as f:
                        f.write(buffer.getvalue())
                # save also from feature extractor
                audio_path = os.path.join(
                    training_args.output_dir, "debug", f"{batch['id']}_orig.mp3"
                )
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                with io.BytesIO() as buffer:
                    sf.write(
                        buffer,
                        sample["array"],
                        sample["sampling_rate"],
                        format="MP3",
                    )
                    with open(audio_path, "wb") as f:
                        f.write(buffer.getvalue())
            return batch

        vectorized_datasets = DatasetDict()
        with training_args.main_process_first(desc="dataset map pre-processing"):
            if training_args.do_train:
                if data_args.streaming:
                    vectorized_datasets["train"] = raw_datasets["train"].map(
                        prepare_dataset,
                        remove_columns=raw_datasets["train"].column_names,
                    )
                else:
                    vectorized_datasets["train"] = raw_datasets["train"].map(
                        prepare_dataset,
                        remove_columns=raw_datasets["train"].column_names,
                        num_proc=data_args.preprocessing_num_workers,
                        desc="preprocess train dataset ",
                    )

            if training_args.do_eval:
                if data_args.streaming:
                    vectorized_datasets["eval"] = raw_datasets["eval"].map(
                        prepare_dataset,
                        remove_columns=raw_datasets["eval"].column_names,
                    )
                else:
                    vectorized_datasets["eval"] = raw_datasets["eval"].map(
                        prepare_dataset,
                        remove_columns=raw_datasets["eval"].column_names,
                        num_proc=data_args.preprocessing_num_workers,
                        desc="preprocess eval dataset",
                    )
            if training_args.do_predict:
                if data_args.streaming:
                    vectorized_datasets["test"] = raw_datasets["test"].map(
                        prepare_dataset,
                        remove_columns=raw_datasets["test"].column_names,
                    )
                else:
                    vectorized_datasets["test"] = raw_datasets["test"].map(
                        prepare_dataset,
                        remove_columns=raw_datasets["test"].column_names,
                        num_proc=data_args.preprocessing_num_workers,
                        desc="preprocess test dataset",
                    )

        # filter data that is shorter than min_input_length or longer than
        # max_input_length
        def is_audio_in_length_range(length):
            return length > min_input_length and length < max_input_length

        if data_args.streaming:
            if training_args.do_train:
                vectorized_datasets["train"] = vectorized_datasets["train"].filter(
                    is_audio_in_length_range, input_columns=["input_length"]
                )
            if training_args.do_eval:
                vectorized_datasets["eval"] = vectorized_datasets["eval"].filter(
                    is_audio_in_length_range, input_columns=["input_length"]
                )
            if training_args.do_predict:
                vectorized_datasets["test"] = vectorized_datasets["test"].filter(
                    is_audio_in_length_range, input_columns=["input_length"]
                )
        else:
            vectorized_datasets = vectorized_datasets.filter(
                is_audio_in_length_range,
                num_proc=num_workers,
                input_columns=["input_length"],
            )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        # WARNING: Change in order not to rewrite the cache
        output_dataset_dir = os.path.join(
            data_args.preprocessed_data_dir, "preprocessed_dataset"
        )
        vectorized_datasets.save_to_disk(
            output_dataset_dir,
            max_shard_size="1GB",
            num_proc=training_args.dataloader_num_workers,
        )
        logger.warning(
            f"Data preprocessing finished. Saving dataset to {output_dataset_dir}."
        )
        return

    logger.warning("Data preprocessing finished.")

    # 8. Load Metric
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        wer = metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # 9. Create a single speech processor
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)

    # 10. Define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
        log_level=training_args.get_process_log_level(),
        debug_output_dir=os.path.join(training_args.output_dir, "debug"),
        sclite_path=data_args.sclite_path,
    )

    # 11. Initialize Trainer
    trainer = DebugSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        processing_class=feature_extractor,
        data_collator=data_collator,
        compute_metrics=(
            compute_metrics if training_args.predict_with_generate else None
        ),
        debug_dir=os.path.join(training_args.output_dir, "debug"),
        actual_tokenizer=tokenizer,
    )

    # 12. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if data_args.torch_profile:
            import glob

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    skip_first=3, wait=1, warmup=1, active=2, repeat=2
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "wandb/latest-run/tbprofile"
                ),
                profile_memory=True,
                with_stack=True,
                record_shapes=True,
            ) as prof:

                trainer.add_callback(ProfCallback(prof=prof))
                train_result = trainer.train()
            # create a wandb Artifact
            profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
            profile_art.add_file(
                glob.glob("wandb/latest-run/tbprofile/*.pt.trace.json")[0],
                "trace.pt.trace.json",
            )
            run.log_artifact(profile_art)
        else:
            train_result = trainer.train()
        trainer.save_model()  # Saves the feature extractor too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(
            max_train_samples, len(vectorized_datasets["train"])
        )
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 13. Evaluation
    results = {}
    if training_args.do_eval:
        logger.warning("*** Evaluate on validation ***")

        metrics = trainer.evaluate(
            metric_key_prefix="eval_dev",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )

        # Log metrics as usual
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(vectorized_datasets["eval"])
        )
        if not data_args.streaming:
            metrics["eval_samples"] = min(
                max_eval_samples, len(vectorized_datasets["eval"])
            )

        trainer.log_metrics("eval_dev", metrics)
        trainer.save_metrics("eval_dev", metrics)
    if training_args.do_predict:
        logger.warning("*** Evaluate on test ***")

        metrics = trainer.evaluate(
            eval_dataset=vectorized_datasets["test"],
            metric_key_prefix="eval_test",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )

        # Log metrics as usual
        if not data_args.streaming:
            metrics["eval_samples"] = len(vectorized_datasets["test"])

        trainer.log_metrics("eval_test", metrics)
        trainer.save_metrics("eval_test", metrics)

    # 14 Run timing on one sample

    if training_args.do_train:
        logger.warning(
            "Measuring training step speed on a 10-second sample from the test dataset."
        )

        # Select a sample from the test dataset with length ~10 seconds
        test_dataset = vectorized_datasets["test"]
        sampling_rate = feature_extractor.sampling_rate
        target_length = 10 * sampling_rate  # 10 seconds in samples
        input_lengths = test_dataset["input_length"]
        differences = [abs(length - target_length) for length in input_lengths]
        min_diff_idx = differences.index(min(differences))
        selected_sample = test_dataset[min_diff_idx]

        # Prepare the batch using the data collator
        batch = [selected_sample]
        batch = data_collator(batch)
        batch = {k: v.to(trainer.args.device) for k, v in batch.items()}

        # Set model to training mode
        model.train()

        # Warm-up iterations (5 iterations)
        num_warmup = 5
        for _ in range(num_warmup):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            model.zero_grad()

        # Measurement iterations (500 iterations)
        num_iterations = 500
        total_time = 0.0

        for _ in tqdm(
            range(num_iterations), desc="Training steps for one sample", leave=False
        ):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            end_event.record()
            torch.cuda.synchronize()
            time_taken = (
                start_event.elapsed_time(end_event) / 1000
            )  # Convert to seconds
            total_time += time_taken

            model.zero_grad()  # Reset gradients for the next iteration

        # Compute average time per step
        avg_time_per_step = total_time / num_iterations
        logger.warning(
            f"Average time per training step: {avg_time_per_step:.6f} seconds"
        )

        # Count parameters
        trainable_params, total_params = count_all_parameters(model)
        logger.warning(f"Trainable parameters: {trainable_params:,}")
        logger.warning(f"Total parameters: {total_params:,}")

        # Log metrics to wandb
        if "wandb" in training_args.report_to:
            wandb.define_metric("avg_time_per_step")
            # Link your two y-axis metrics to the custom x-axis
            wandb.define_metric("trainable_params", step_metric="avg_time_per_step")
            wandb.define_metric("total_params", step_metric="avg_time_per_step")
            wandb.log(
                {
                    "avg_time_per_step": avg_time_per_step,
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "test_sample_index": min_diff_idx,
                }
            )

    # 15. Write Training Stats
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = (
                f"{data_args.dataset_name} {data_args.dataset_config_name}"
            )
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    if "wandb" in training_args.report_to:
        wandb.finish()

    return results


if __name__ == "__main__":
    main()
