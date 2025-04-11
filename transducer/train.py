#!/usr/bin/env python3
import logging
import pathlib
from argparse import ArgumentParser

from common import MODEL_TYPE_LIBRISPEECH
from librispeech.lightning import LibriSpeechRNNTModule
from librispeech.lightning_wav2vec2 import LibriSpeechRNNTModuleWav2Vec2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def get_trainer(args):
    checkpoint_dir = args.exp_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/val_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=True,
        verbose=True,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=True,
        verbose=True,
    )
    callbacks = [
        checkpoint,
        train_checkpoint,
    ]
    return Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=args.epochs,
        accelerator="gpu",
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        # precision="fp16-mixed"
    )


def get_lightning_module(args):
    if args.model_type == MODEL_TYPE_LIBRISPEECH:
        return LibriSpeechRNNTModule(
            librispeech_path=str(args.dataset_path),
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )
    else:
        raise ValueError(f"Encountered unsupported model type {args.model_type}.")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model-type", type=str, choices=[MODEL_TYPE_LIBRISPEECH], required=True
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
        required=True,
    )
    parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        help="Path to datasets.",
        required=True,
    )
    parser.add_argument(
        "--sp-model-path",
        type=pathlib.Path,
        help="Path to SentencePiece model.",
        required=True,
    )
    parser.add_argument(
        "--exp-dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--epochs",
        default=120,
        type=int,
        help="Number of epochs to train for. (Default: 120)",
    )
    parser.add_argument(
        "--gradient-clip-val", default=10.0, type=float, help="Value to clip gradient values to. (Default: 10.0)"
    )
    return parser.parse_args()


def init_logger():
    fmt = "%(message)s"
    level = logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    args = parse_args()
    init_logger()
    model = get_lightning_module(args)
    trainer = get_trainer(args)
    trainer.fit(model)
