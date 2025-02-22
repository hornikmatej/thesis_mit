from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    feature_extractor_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "feature extractor name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
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
    freeze_feature_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the feature encoder layers of the model."},
    )
    freeze_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the entire encoder of the seq2seq model."},
    )
    forced_decoder_ids: List[List[int]] = field(
        default=None,
        metadata={
            "help": "Deprecated. Please use the `language` and `task` arguments instead."
        },
    )
    suppress_tokens: List[int] = field(
        default=None,
        metadata={
            "help": (
                "Deprecated. The use of `suppress_tokens` should not be required for the majority of fine-tuning examples."
                "Should you need to use `suppress_tokens`, please manually update them in the fine-tuning script directly."
            )
        },
    )
    apply_spec_augment: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply *SpecAugment* data augmentation to the input features. This is currently only relevant for Wav2Vec2, HuBERT, WavLM and Whisper models."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "The name of the dataset column containing the audio data. Defaults to 'audio'"
        },
    )
    text_column_name: str = field(
        default="text",
        metadata={
            "help": "The name of the dataset column containing the text data. Defaults to 'text'"
        },
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0,
        metadata={
            "help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    preprocessed_data_dir: str = field(
        default="./",
        metadata={
            "help": (
                "Where to store the preprocessed data. If not set, the preprocessed data will be stored in the same"
                " directory as the original data."
            )
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    test_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'test'"
        },
    )
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={
            "help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."
        },
    )
    streaming: bool = field(
        default=False,
        metadata={
            "help": "Whether to use streaming inference for downloading dataset."
        },
    )
    sclite_path: str = field(
        default="/home/matej/fitvut/dp_mit/SCTK/bin/sclite",
        metadata={"help": "Path to the SCLITE executable"},
    )
    wandb_project: str = field(
        default="seq2seq_encoder-decoder",
        metadata={"help": "Weights & Biases project name"},
    )
    torch_profile: bool = field(
        default=False,
        metadata={"help": "Whether to profile the training using PyTorch profiler"},
    )
