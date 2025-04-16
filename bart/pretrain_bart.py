#!/usr/bin/env python
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Dict, Tuple, Union
from collections import defaultdict

import datasets
import evaluate
import wandb
import torch
from datasets import load_dataset
from numpy.random import permutation, poisson

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding
from transformers.data.data_collator import _torch_collate_batch
from dotenv import load_dotenv

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
logger = logging.getLogger(__name__)

load_dotenv("../.env")
WANDB_KEY = os.getenv("WANDB_TOKEN")

@dataclass
class DataCollatorForTextInfilling:
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    poisson_lambda: float = 3.0
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError

    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
                 ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {"input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        batch["decoder_input_ids"] = self.shift_tokens_right(batch["input_ids"])

        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        return batch

    def shift_tokens_right(self, inputs):
        shifted_inputs = torch.roll(inputs, 1, dims=-1)
        shifted_inputs[:, 0] = self.tokenizer.eos_token_id

        # replace eos tokens at the end of sequences with pad tokens
        end_with_eos = (shifted_inputs[:, -1] == self.tokenizer.eos_token_id)
        shifted_inputs[end_with_eos, -1] = self.tokenizer.pad_token_id

        # find positions where token is eos and its following token is a padding token
        last_eos_indices = torch.where(
            (shifted_inputs[:, :-1] == self.tokenizer.eos_token_id) &
            (shifted_inputs[:, 1:] == self.tokenizer.pad_token_id)
        )

        shifted_inputs[last_eos_indices] = self.tokenizer.pad_token_id
        return shifted_inputs

    def mask_tokens(self,
                    inputs: torch.Tensor,
                    special_tokens_mask: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # determine how many tokens we need to mask in total
        is_token = ~(inputs == self.tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.float().sum() * self.mlm_probability))

        if num_to_mask == 0:
            return inputs, labels

        # generate a sufficient number of span lengths
        poisson_distribution = torch.distributions.Poisson(rate=self.poisson_lambda)
        lengths = poisson_distribution.sample(sample_shape=(num_to_mask,))
        while torch.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = torch.cat([lengths, poisson_distribution.sample(sample_shape=(num_to_mask,))])

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = torch.argmin(torch.abs(torch.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[:idx + 1]

        # select span start indices
        token_indices = is_token.nonzero(as_tuple=False)
        span_starts = torch.randperm(token_indices.shape[0])[:lengths.shape[0]]

        # prepare mask
        masked_indices = token_indices[span_starts]
        mask = torch.full_like(inputs, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = inputs.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while torch.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[special_tokens_mask] = False
        inputs[mask.bool()] = self.tokenizer.mask_token_id
        labels[~mask.bool()] = -100

        # remove mask tokens that are not starts of spans
        to_remove = mask.bool() & mask.bool().roll(1, 1)
        new_inputs = torch.full_like(inputs, fill_value=self.tokenizer.pad_token_id)
        for i, example in enumerate(torch.split(inputs, split_size_or_sections=1, dim=0)):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0:new_example.shape[0]] = new_example

        return new_inputs, labels

@dataclass
class DataCollatorForDenoisingTasks:
    """Data collator used denoising language modeling task in BART.
    The implementation is based on
    https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/fairseq/data/denoising_dataset.py.
    The default paramters is based on BART paper https://arxiv.org/abs/1910.13461.
    """

    tokenizer: PreTrainedTokenizerBase
    mask_ratio: float = 0.15
    poisson_lambda: float = 3.0
    permutate_sentence_ratio: float = 1.0
    pad_to_multiple_of: int = 16

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """Batching, adding whole word mask and permutate sentences
        Args:
            examples (dict): list of examples each examples contains input_ids field
        """
        # Use torch tensors for batching
        batch = self.tokenizer.pad(examples, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt")
        batch["decoder_input_ids"] = self.shift_tokens_right(batch["input_ids"])

        do_permutate = False
        if self.permutate_sentence_ratio > 0.0:
            batch["input_ids"] = self.permutate_sentences(batch["input_ids"])
            do_permutate = True

        if self.mask_ratio:
            batch["input_ids"], batch["labels"] = self.add_whole_word_mask(batch["input_ids"], do_permutate)

        return batch

    def shift_tokens_right(self, inputs):
        shifted_inputs = torch.roll(inputs, 1, dims=-1)
        shifted_inputs[:, 0] = self.tokenizer.eos_token_id

        # replace eos tokens at the end of sequences with pad tokens
        end_with_eos = (shifted_inputs[:, -1] == self.tokenizer.eos_token_id)
        shifted_inputs[end_with_eos, -1] = self.tokenizer.pad_token_id

        # find positions where token is eos and its following token is a padding token
        last_eos_indices = torch.where(
            (shifted_inputs[:, :-1] == self.tokenizer.eos_token_id) &
            (shifted_inputs[:, 1:] == self.tokenizer.pad_token_id)
        )

        shifted_inputs[last_eos_indices] = self.tokenizer.pad_token_id
        return shifted_inputs

    def permutate_sentences(self, inputs):
        results = inputs.clone()
        full_stops = (inputs == self.tokenizer.eos_token_id)

        # Find sentence ends
        full_stops_shifted = full_stops[:, 1:] & (~full_stops[:, :-1])
        sentence_ends = torch.nonzero(full_stops_shifted, as_tuple=False)
        sentence_ends[:, 1] += 2
        num_sentences = torch.unique(sentence_ends[:, 0], return_counts=True)[1]
        num_to_permute = torch.ceil((num_sentences * 2 * self.permutate_sentence_ratio) / 2.0).to(torch.int)

        # Split sentence ends by batch
        unique_batches, counts = torch.unique(sentence_ends[:, 0], return_counts=True)
        split_indices = torch.cumsum(counts, dim=0)[:-1].tolist()  # indices to split at
        sentence_ends_split = torch.split(sentence_ends[:, 1], counts.tolist())

        for i in range(inputs.shape[0]):
            if num_sentences[i] == 0:
                continue
            substitutions = torch.randperm(num_sentences[i])[: num_to_permute[i]]
            ordering = torch.arange(0, num_sentences[i])
            ordering[substitutions] = substitutions[torch.randperm(num_to_permute[i])]

            index = 0
            for j in ordering:
                start = sentence_ends_split[i][j - 1] if j > 0 else 0
                end = sentence_ends_split[i][j]
                sentence = inputs[i, start:end]
                results[i, index : index + sentence.shape[0]] = sentence
                index += sentence.shape[0]
        return results

    def add_whole_word_mask(self, inputs, do_permutate):
        labels = inputs.clone()

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True) for val in labels
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device)

        is_token = ~(labels == self.tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.float().sum().item() * self.mask_ratio))
        if num_to_mask == 0:
            return inputs, labels

        # generate a sufficient number of span lengths
        lengths = torch.tensor(poisson(lam=self.poisson_lambda, size=(num_to_mask,)), device=labels.device)
        while torch.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = torch.cat([lengths, torch.tensor(poisson(lam=self.poisson_lambda, size=(num_to_mask,)), device=labels.device)])

        lengths = lengths[lengths > 0]
        idx = torch.argmin(torch.abs(torch.cumsum(lengths, 0) - num_to_mask)).item() + 1
        lengths = lengths[: idx + 1]

        token_indices = torch.nonzero(is_token, as_tuple=False)
        span_starts = torch.tensor(permutation(token_indices.shape[0])[: lengths.shape[0]], device=labels.device)
        masked_indices = token_indices[span_starts]
        mask = torch.full_like(labels, fill_value=False, dtype=torch.bool)

        for mi in masked_indices:
            mask[mi[0], mi[1]] = True
        lengths -= 1

        max_index = labels.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while torch.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[mi[0], mi[1]] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        mask[special_tokens_mask] = False
        inputs[mask] = self.tokenizer.mask_token_id

        if not do_permutate:
            labels[mask] = -100
        else:
            labels[special_tokens_mask] = -100

        to_remove = (mask == 1) & torch.roll((mask == 1), 1, 1)
        new_inputs = torch.full_like(labels, fill_value=self.tokenizer.pad_token_id)

        for i, example in enumerate(torch.split(inputs, 1, dim=0)):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0 : new_example.shape[0]] = new_example

        return new_inputs, labels

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    data_collator: str = field(
        default="Not given",
        metadata={
            "help": (
                "The data collator to be used for training. It can be one of the following: "
                "'text_infilling', 'denoising'."
            )
        },
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the wandb project to use. If not specified, will use the name of the current directory."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # wandb
    training_args.run_name = (
        f"{data_args.dataset_name}_"
        f"bs{training_args.per_device_train_batch_size}_"
        f"lr{training_args.learning_rate}_"
        f"ep{training_args.num_train_epochs}_"
        f"dc-{data_args.data_collator}_"
        f"mlm{data_args.mlm_probability}_"
        f"max_seqL{data_args.max_seq_length}"
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
    # Set seed before initializing model.
    set_seed(training_args.seed)
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        streaming=data_args.streaming,
        trust_remote_code=model_args.trust_remote_code,
    )
    logger.warning(f"Loaded {data_args.dataset_name} dataset, keys: {raw_datasets.keys()}")
    
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else: 
        raise ValueError("Pls tokenizer where?")
    
    torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )

    # Preprocessing the datasets.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "normalized_text" # Specific to VoxPopuli

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    logger.warning(f"Max sequence length: {max_seq_length}")

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=False,
            )

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=[text_column_name],
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=[text_column_name],
                )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=False)

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on every text in dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # the small remainder, and if the total_length < max_seq_length
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result


        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
            else:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )
    
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # TODO: logits are always first (coped for the future)
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

    pad_to_multiple_of_8 = data_args.pad_to_max_length
    if data_args.data_collator == "text_infilling":
        data_collator = DataCollatorForTextInfilling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )
    elif data_args.data_collator == "denoising":
        data_collator = DataCollatorForDenoisingTasks(
            tokenizer=tokenizer,
            mask_ratio=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )
    else: 
        raise ValueError(f"Unknown data collator: {data_args.data_collator}")

    logger.warning("Data collator initialized with mask ratio: %s", data_args.mlm_probability)
    # sample_dataset = None
    # if training_args.do_train:
    #     sample_dataset = train_dataset
    # elif training_args.do_eval:
    #     sample_dataset = eval_dataset

    # if sample_dataset is not None:
    #     # Get 16 examples
    #     sample_examples = [sample_dataset[i] for i in range(min(16, len(sample_dataset)))]
    #     batch = data_collator(sample_examples)
    #     input_ids = batch["input_ids"]
    #     labels = batch["labels"]
    #     decoder_input_ids = batch.get("decoder_input_ids", None)

    #     logger.warning("Logging 16 examples after data collator:")
    #     for i in range(input_ids.shape[0]):
    #         logger.warning(f"Example {i+1}:")
    #         logger.warning("  Encoded input_ids: %s", input_ids[i].tolist())
    #         logger.warning("  Decoded input: %s", tokenizer.decode(input_ids[i], skip_special_tokens=False))
    #         logger.warning("  Encoded labels: %s", labels[i].tolist())
    #         logger.warning("  Decoded labels: %s", tokenizer.decode([t if t != -100 else tokenizer.pad_token_id for t in labels[i].tolist()], skip_special_tokens=False))
    #         if decoder_input_ids is not None:
    #             logger.warning("  Encoded decoder_input_ids: %s", decoder_input_ids[i].tolist())
    #             logger.warning("  Decoded decoder_input: %s", tokenizer.decode(decoder_input_ids[i], skip_special_tokens=False))


    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = len(train_dataset)
        metrics["train_samples"] = max_train_samples

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate on validation ***")

        metrics = trainer.evaluate()

        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if "wandb" in training_args.report_to:
        wandb.finish()

if __name__ == "__main__":
    main()