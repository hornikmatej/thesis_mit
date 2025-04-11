#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

"""Fine-tuning a 🤗 Transformers CTC model for automatic speech recognition"""

import functools
import json
import logging
import os
import re
import sys
import warnings
import wandb
import aiohttp
import numpy as np
import subprocess
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any


import datasets
import evaluate
import torch
import pandas as pd
from datasets import DatasetDict, load_dataset, DownloadConfig

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from dataclass_args_ctc import ModelArguments, DataTrainingArguments
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)
download_config = DownloadConfig(
    resume_download=True,
    max_retries=3,  # Optionally increase the number of retries
    storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
)

# Monitoring
WANDB_KEY = settings.wandb_token.get_secret_value()


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    feature_extractor_input_name: Optional[str] = "input_values"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            {
                self.feature_extractor_input_name: feature[
                    self.feature_extractor_input_name
                ]
            }
            for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        return batch


def create_vocabulary_from_data(
    datasets: DatasetDict,
    word_delimiter_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    pad_token: Optional[str] = None,
):
    # Given training and test labels create vocabulary
    def extract_all_chars(batch):
        all_text = " ".join(batch["target_text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = datasets.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=datasets["train"].column_names,
    )

    # take union of all unique characters in each dataset
    print(vocabs.values())
    vocab_list = [dataset["vocab"][0] for dataset in vocabs.values()]
    vocab_set = functools.reduce(lambda a, b: set(a) | set(b), vocab_list)

    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_set))}

    # replace white space with delimiter token
    if word_delimiter_token is not None:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    return vocab_dict


def parse_sclite_dtl_file(dtl_file_path: str):
    """
    Parses the sclite .dtl output file and extracts relevant metrics,
    calculating percentages from raw counts.

    Args:
        dtl_file_path: The path to the .dtl file generated by sclite.

    Returns:
        A dictionary containing the parsed metrics (substitutions, deletions, insertions).
        Returns None if parsing fails or the file doesn't exist.
    """
    metrics = {}
    try:
        with open(dtl_file_path, "r") as f:
            sclite_output = f.read()

        # Extract raw counts using regular expressions
        sub_count_match = re.search(
            r"Percent Substitution\s*=.*?\(\s*(\d+)\s*\)", sclite_output
        )
        del_count_match = re.search(
            r"Percent Deletions\s*=.*?\(\s*(\d+)\s*\)", sclite_output
        )
        ins_count_match = re.search(
            r"Percent Insertions\s*=.*?\(\s*(\d+)\s*\)", sclite_output
        )
        ref_words_match = re.search(r"Ref\. words\s*=\s*\(\s*(\d+)\s*\)", sclite_output)
        total_error_match = re.search(
            r"Percent Total Error\s*=.*?\(\s*(\d+)\s*\)", sclite_output
        )
        correct_match = re.search(
            r"Percent Correct\s*=.*?\(\s*(\d+)\s*\)", sclite_output
        )

        # Extract sentence level
        sent_count_match = re.search(r"sentences\s*(\d+)", sclite_output)
        sent_err_count_match = re.search(
            r"with errors.*?\(\s*(\d+)\s*\)", sclite_output
        )

        # Calculate percentages if counts are found
        if ref_words_match:
            ref_words = int(ref_words_match.group(1))

            if sub_count_match:
                substitutions = int(sub_count_match.group(1))
                metrics["substitutions"] = round((substitutions / ref_words) * 100, 2)

            if del_count_match:
                deletions = int(del_count_match.group(1))
                metrics["deletions"] = round((deletions / ref_words) * 100, 2)

            if ins_count_match:
                insertions = int(ins_count_match.group(1))
                metrics["insertions"] = round((insertions / ref_words) * 100, 2)

            if total_error_match:
                total_errors = int(total_error_match.group(1))
                metrics["word_errors"] = round((total_errors / ref_words) * 100, 2)

            if correct_match:
                correct_words = int(correct_match.group(1))
                metrics["word_accuracy"] = round((correct_words / ref_words) * 100, 2)

        if sent_count_match and sent_err_count_match:
            total_sentences = int(sent_count_match.group(1))
            sentence_errors = int(sent_err_count_match.group(1))
            metrics["sentence_errors"] = round(
                (sentence_errors / total_sentences) * 100, 2
            )

    except FileNotFoundError:
        logger.warning(f"Sclite .dtl file not found: {dtl_file_path}")
        return None
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to parse sclite output: {e}")
        return None

    return metrics


def count_all_parameters(model):
    """
    Count the number of trainable and total parameters in the model.

    Args:
        model: PyTorch model.

    Returns:
        tuple: (trainable_params, total_params)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


class DebugTrainer(Trainer):
    """
    Custom Trainer that adds debugging capabilities during training and evaluation.
    """

    def __init__(
        self,
        *args,
        debug_dir: str = "debug_output",
        actual_tokenizer=None,
        sclite_path=str,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.debug_dir = debug_dir
        self.actual_tokenizer = actual_tokenizer
        self.sclite_path = sclite_path
        os.makedirs(debug_dir, exist_ok=True)

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Union[Dict[str, float], Dict[str, float]]:
        # Run the standard evaluation loop
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # If we're not just computing loss, run debug analysis
        if not prediction_loss_only:
            self._run_debug_analysis(eval_output, metric_key_prefix)

        return eval_output

    def _run_debug_analysis(self, eval_output: Dict[str, Any], prefix: str):
        """
        Run debug analysis after evaluation.
        """
        pred_ids = eval_output.predictions[0]
        eval_output.label_ids[eval_output.label_ids == -100] = (
            self.actual_tokenizer.pad_token_id
        )
        pred_str = self.actual_tokenizer.batch_decode(pred_ids)
        label_str = self.actual_tokenizer.batch_decode(
            eval_output.label_ids, group_tokens=False
        )

        # Run debug analysis using the data collator
        batch_idx = f"{prefix}_{self.state.global_step}"

        # Save predictions and references to CSV
        debug_path = os.path.join(self.debug_dir, f"debug_batch_{batch_idx}.csv")
        df = pd.DataFrame({"label": label_str, "prediction": pred_str})
        df.to_csv(debug_path, index=False)

        # Create SCLITE format files (.trn)
        sclite_files = [
            debug_path.replace(".csv", f"_{type}.trn") for type in ["hyp", "ref"]
        ]

        # Save hypothesis and reference files in SCLITE format
        for strings, file_to_save in zip([pred_str, label_str], sclite_files):
            with open(file_to_save, "w") as file_handler:
                for index, string in enumerate(strings):
                    file_handler.write(f"{string} (utterance_{index})\n")

        # Run SCLITE evaluation
        dtl_file_path = debug_path.replace(
            ".csv", "_hyp.trn.dtl"
        )  # Construct .dtl file path
        sclite_cmd = f"{self.sclite_path} -F -D -i wsj -r {sclite_files[1]} trn -h {sclite_files[0]} trn -o snt sum dtl"
        logger.info(f"Running SCLITE evaluation with command: {sclite_cmd}")
        process = subprocess.Popen(sclite_cmd.split())  # nosec

        try:
            process.wait(30)  # Wait up to 30 seconds for SCLITE to complete
        except subprocess.TimeoutExpired:
            process.kill()
            logger.warning("Sclite evaluation timed out.")

        # Parse sclite output from the .dtl file
        metrics = parse_sclite_dtl_file(dtl_file_path)

        # Log to WandB
        if metrics:
            wandb.log(metrics)  # Log with batch index as step
            logger.info(f"Logged SCLITE metrics to WandB: {metrics}")
        else:
            logger.warning(
                "Skipping WandB logging due to parsing failure or missing file."
            )

        logger.info(f"Debug information saved to {debug_path}")
        logger.info(f"SCLITE analysis files: {sclite_files}")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # add wandb run name to training_args
    training_args.run_name = (
        f"{data_args.dataset_name}_{data_args.dataset_config_name}_"
        f"split-{data_args.train_split_name}_wav2vec2-bart_"
        f"bs{training_args.per_device_train_batch_size}_"
        f"lr{training_args.learning_rate}_"
        f"ep{training_args.num_train_epochs}"
    )
    WANDB_PROJECT = data_args.wandb_project

    run: wandb.Run = None
    if "wandb" in training_args.report_to:
        wandb.login(key=WANDB_KEY)
        run = wandb.init(
            project=WANDB_PROJECT,
            job_type="training",
            anonymous="allow",
            name=training_args.run_name,
        )
    # Detecting last checkpoint.
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
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 1. First, let's load the dataset
    raw_datasets = DatasetDict()

    if model_args.cache_dir is not None:
        logger.warning(f"Using cache dir {model_args.cache_dir} for the datasets.")
        vectorized_datasets = datasets.load_from_disk(model_args.cache_dir)
    else:
        if training_args.do_train:
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                # cache_dir=model_args.cache_dir,
                split=data_args.train_split_name,
                token=data_args.token,
                trust_remote_code=data_args.trust_remote_code,
                num_proc=data_args.preprocessing_num_workers,
                download_config=download_config,
            )
            if data_args.audio_column_name not in raw_datasets["train"].column_names:
                raise ValueError(
                    f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'."
                    " Make sure to set `--audio_column_name` to the correct audio column - one of"
                    f" {', '.join(raw_datasets['train'].column_names)}."
                )
            if data_args.text_column_name not in raw_datasets["train"].column_names:
                raise ValueError(
                    f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
                    "Make sure to set `--text_column_name` to the correct text column - one of "
                    f"{', '.join(raw_datasets['train'].column_names)}."
                )
            if data_args.max_train_samples is not None:
                raw_datasets["train"] = raw_datasets["train"].select(
                    range(data_args.max_train_samples)
                )
        if training_args.do_eval:
            raw_datasets["eval"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                # cache_dir=model_args.cache_dir,
                split=data_args.eval_split_name,
                token=data_args.token,
                trust_remote_code=data_args.trust_remote_code,
                num_proc=data_args.preprocessing_num_workers,
                download_config=download_config,
            )

            if data_args.max_eval_samples is not None:
                raw_datasets["eval"] = raw_datasets["eval"].select(
                    range(data_args.max_eval_samples)
                )
        if training_args.do_predict:
            raw_datasets["test"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=data_args.test_split_name,
                # cache_dir=model_args.cache_dir,
                token=data_args.token,
                trust_remote_code=data_args.trust_remote_code,
                num_proc=data_args.preprocessing_num_workers,
                download_config=download_config,
            )

    # 2. We remove some special characters from the datasets such as `,` and `.`
    chars_to_ignore_regex = (
        f'[{"".join(data_args.chars_to_ignore)}]'
        if data_args.chars_to_ignore is not None
        else None
    )
    text_column_name = data_args.text_column_name

    def remove_special_characters(batch):
        if chars_to_ignore_regex is not None:
            batch["target_text"] = (
                re.sub(chars_to_ignore_regex, "", batch[text_column_name]).lower() + " "
            )
        else:
            batch["target_text"] = batch[text_column_name].lower() + " "
        return batch

    with training_args.main_process_first(
        desc="dataset map special characters removal"
    ):
        raw_datasets = raw_datasets.map(
            remove_special_characters,
            remove_columns=[text_column_name],
            desc="remove special characters from datasets",
        )

    # save special tokens for tokenizer
    word_delimiter_token = data_args.word_delimiter_token
    unk_token = data_args.unk_token
    pad_token = data_args.pad_token

    # 3. Next, let's load the config as we might need it to create, load the tokenizer, config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # 4. Next, if no tokenizer file is defined,
    # we create the vocabulary of the model by extracting all unique characters from
    # the training and evaluation datasets
    # We need to make sure that only first rank saves vocabulary
    # make sure all processes wait until vocab is created
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    tokenizer_kwargs = {}
    if tokenizer_name_or_path is None:
        # save vocab in training output dir
        tokenizer_name_or_path = training_args.output_dir

        vocab_file = os.path.join(tokenizer_name_or_path, "vocab.json")

        with training_args.main_process_first():
            if training_args.overwrite_output_dir and os.path.isfile(vocab_file):
                try:
                    os.remove(vocab_file)
                except OSError:
                    # in shared file-systems it might be the case that
                    # two processes try to delete the vocab file at the some time
                    pass

        with training_args.main_process_first(desc="dataset map vocabulary creation"):
            if not os.path.isfile(vocab_file):
                os.makedirs(tokenizer_name_or_path, exist_ok=True)
                vocab_dict = create_vocabulary_from_data(
                    raw_datasets,
                    word_delimiter_token=word_delimiter_token,
                    unk_token=unk_token,
                    pad_token=pad_token,
                )

                # save vocab dict to be loaded into tokenizer
                with open(vocab_file, "w") as file:
                    json.dump(vocab_dict, file)

        # if tokenizer has just been created
        # it is defined by `tokenizer_class` if present in config else by `model_type`
        tokenizer_kwargs = {
            "config": config if config.tokenizer_class is not None else None,
            "tokenizer_type": (
                config.model_type if config.tokenizer_class is None else None
            ),
            "unk_token": unk_token,
            "pad_token": pad_token,
            "word_delimiter_token": word_delimiter_token,
        }

    # 5. Now we can instantiate the feature extractor, tokenizer and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    # load feature_extractor and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        **tokenizer_kwargs,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # adapt config
    config.update(
        {
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": model_args.layerdrop,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "ctc_zero_infinity": model_args.ctc_zero_infinity,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
            "activation_dropout": model_args.activation_dropout,
            "add_adapter": model_args.add_adapter,
        }
    )

    # create model
    model = AutoModelForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # freeze encoder
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # 6. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`

    # make sure that dataset decodes audio with correct sampling rate
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

    # derive max & min input length for sample rate & max duration
    max_input_length = (
        data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    )
    min_input_length = (
        data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    )
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    feature_extractor_input_name = feature_extractor.model_input_names[0]

    # `phoneme_language` is only relevant if the model is fine-tuned on phoneme classification
    phoneme_language = data_args.phoneme_language

    # Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    def prepare_dataset(batch):
        # load audio
        sample = batch[audio_column_name]

        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"]
        )
        batch[feature_extractor_input_name] = getattr(
            inputs, feature_extractor_input_name
        )[0]
        # take length of raw audio waveform
        batch["input_length"] = len(sample["array"].squeeze())

        # encode targets
        additional_kwargs = {}
        if phoneme_language is not None:
            additional_kwargs["phonemizer_lang"] = phoneme_language

        batch["labels"] = tokenizer(batch["target_text"], **additional_kwargs).input_ids
        return batch

    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=num_workers,
            desc="preprocess datasets",
        )

        def is_audio_in_length_range(length):
            return length > min_input_length and length < max_input_length

        # filter data that is shorter than min_input_length
        vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["input_length"],
        )

    # 7. Next, we can prepare the training.
    # Let's use word error rate (WER) as our evaluation metric,
    # instantiate a data collator and the trainer

    # Define evaluation metrics during training, *i.e.* word error rate, character error rate
    eval_metrics = {
        metric: evaluate.load(metric, cache_dir=model_args.cache_dir)
        for metric in data_args.eval_metrics
    }

    # cached dataset
    if data_args.preprocessing_only:
        logger.info(
            f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}"
        )
        return

    # For languages like Chinese with large vocabulary size, we need to discard logits
    # and only keep the argmax, otherwise we run out of memory during evaluation.
    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels

    def compute_metrics(pred):
        pred_ids = pred.predictions[0]
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        metrics = {
            k: v.compute(predictions=pred_str, references=label_str)
            for k, v in eval_metrics.items()
        }

        return metrics

    # Now save everything to be able to create a single processor later
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    try:
        processor = AutoProcessor.from_pretrained(training_args.output_dir)
    except (OSError, KeyError):
        warnings.warn(
            "Loading a processor from a feature extractor config that does not"
            " include a `processor_class` attribute is deprecated and will be removed in v5. Please add the following "
            " attribute to your `preprocessor_config.json` file to suppress this warning: "
            " `'processor_class': 'Wav2Vec2Processor'`",
            FutureWarning,
        )
        processor = Wav2Vec2Processor.from_pretrained(training_args.output_dir)

    # Instantiate custom data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, feature_extractor_input_name=feature_extractor_input_name
    )

    # Initialize Trainer
    trainer = DebugTrainer(
        model=model,
        data_collator=data_collator,
        actual_tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        processing_class=processor,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        debug_dir=os.path.join(training_args.output_dir, "debug"),
        sclite_path=data_args.sclite_path,
    )

    # 8. Finally, we can start training

    # Training
    if training_args.do_train:
        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

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

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate  on validation ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval_dev",
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(vectorized_datasets["eval"])
        )
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
        )

        metrics["eval_samples"] = len(vectorized_datasets["test"])

        trainer.log_metrics("eval_test", metrics)
        trainer.save_metrics("eval_test", metrics)

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

        # Warm-up iterations (15 iterations)
        num_warmup = 15
        for _ in range(num_warmup):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            model.zero_grad()

        # Measurement iterations (500 iterations)
        num_iterations = 500
        total_time = []

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
            total_time.append(time_taken)

            model.zero_grad()  # Reset gradients for the next iteration

        # Compute average time per step
        avg_time_per_step = sum(total_time) / num_iterations
        std_dev = np.std(total_time)
        logger.warning(
            f"Average time per training step: {avg_time_per_step:.6f} seconds"
        )
        logger.warning(f"Standard deviation of training steps: {std_dev:.6f} seconds")

        # Count parameters
        trainable_params, total_params = count_all_parameters(model)
        logger.warning(f"Trainable parameters: {trainable_params:,}")
        logger.warning(f"Total parameters: {total_params:,}")

        # Log metrics to wandb
        if "wandb" in training_args.report_to:
            data_total = [[avg_time_per_step, trainable_params]]
            data_trainable = [[avg_time_per_step, total_params]]
            wandb.log(
                {
                    "model_speed2size1": wandb.plot.scatter(
                        wandb.Table(
                            data=data_total,
                            columns=["Time per step", "Trainable parameters"],
                        ),
                        "Time per step",
                        "Trainable parameters",
                    ),
                }
            )
            wandb.log(
                {
                    "model_speed2size2": wandb.plot.scatter(
                        wandb.Table(
                            data=data_trainable,
                            columns=["Time per step", "Total parameters"],
                        ),
                        "Time per step",
                        "Total parameters",
                    ),
                }
            )
            wandb.log(
                {
                    "test_sample_index": min_diff_idx,
                }
            )

    # Write model card and (optionally) push to hub
    config_name = (
        data_args.dataset_config_name
        if data_args.dataset_config_name is not None
        else "na"
    )
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
        "tags": ["automatic-speech-recognition", data_args.dataset_name],
        "dataset_args": (
            f"Config: {config_name}, Training split: {data_args.train_split_name}, Eval split:"
            f" {data_args.eval_split_name}"
        ),
        "dataset": f"{data_args.dataset_name.upper()} - {config_name.upper()}",
    }
    if "common_voice" in data_args.dataset_name:
        kwargs["language"] = config_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    if "wandb" in training_args.report_to:
        wandb.finish()

    return results


if __name__ == "__main__":
    main()
