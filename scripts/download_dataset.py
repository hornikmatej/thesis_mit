import os
import logging
import sys

import datasets
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers.trainer_utils import is_main_process
from datasets import load_dataset, DatasetDict

from src.dataclass_args import ModelArguments, DataTrainingArguments


logger = logging.getLogger(__name__)
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
    # 1. Parse command line arguments
    model_args, data_args, training_args = parse_args()

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # 3. Download dataset
    _ = download_dataset(model_args, data_args, training_args)
    logger.info("Dataset downloaded successfully.")


if __name__ == "__main__":
    main()
