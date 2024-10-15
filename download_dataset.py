import os
import sys

from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from datasets import load_dataset, DatasetDict

from src.dataclass_args import ModelArguments, DataTrainingArguments

# set env variable for HF_HOME
os.environ["HF_HOME"] = "/storage/brno2/home/xhorni20/.cache/huggingface"


def parse_args():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    return parser.parse_args_into_dataclasses()


def download_dataset(model_args, data_args, training_args):
    raw_datasets = DatasetDict()
    raw_datasets["train"] = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=data_args.train_split_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        num_proc=data_args.preprocessing_num_workers,
    )
    raw_datasets["eval"] = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=data_args.eval_split_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        num_proc=data_args.preprocessing_num_workers,
    )
    return raw_datasets


def main():
    model_args, data_args, training_args = parse_args()
    _ = download_dataset(model_args, data_args, training_args)
    print("Dataset downloaded and cached successfully.")


if __name__ == "__main__":
    main()
