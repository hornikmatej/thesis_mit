#!/usr/bin/env python3
"""Evaluate the lightning module by loading the checkpoint, the SentencePiece model, and the global_stats.json.
Example:
python eval.py --model-type tedlium3 --checkpoint-path ./experiments/checkpoints/epoch=119-step=254999.ckpt
    --dataset-path ./datasets/tedlium --sp-model-path ./spm_bpe_500.model
"""
import logging
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter

import torch
import torchaudio
from librispeech.lightning import LibriSpeechRNNTModule
from librispeech.lightning_wav2vec2 import LibriSpeechRNNTModuleWav2Vec2

logger = logging.getLogger(__name__)


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def run_eval_subset(model, dataloader, subset):
    total_edit_distance = 0
    total_length = 0
    with torch.no_grad():
        for idx, (batch, transcripts) in enumerate(dataloader):
            actual = transcripts[0]
            predicted = model(batch)
            total_edit_distance += compute_word_level_distance(actual, predicted)
            total_length += len(actual.split())
            if idx % 100 == 0:
                logger.info(f"Processed elem {idx}; WER: {total_edit_distance / total_length}")
    logger.info(f"Final WER for {subset} set: {total_edit_distance / total_length}")


def run_eval(model):
    test_dataloader = model.test_dataloader()
    # dev_dataloader = model.val_dataloader()
    # run_eval_subset(model, dev_dataloader, "dev")
    run_eval_subset(model, test_dataloader, "test")



def get_lightning_module(args):
    if args.mtype == "emformer":
        return LibriSpeechRNNTModule.load_from_checkpoint(
            args.checkpoint_path,
            librispeech_path="libri_dataset/",
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )
    elif args.mtype == "wav2vec2":
        return LibriSpeechRNNTModuleWav2Vec2.load_from_checkpoint(
            args.checkpoint_path,
            librispeech_path="libri_dataset/",
            sp_model_path=str(args.sp_model_path),
        )


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--mtype",
        default="emformer",
        type=str,
        choices=["emformer", "wav2vec2"],
        help="Model type to use. (Default: 'emformer')",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        help="Path to checkpoint to use for evaluation.",
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        help="Path to dataset.",
    )
    parser.add_argument(
        "--sp-model-path",
        type=pathlib.Path,
        help="Path to SentencePiece model.",
    )
    return parser.parse_args()


def init_logger():
    fmt = "%(message)s"
    level = logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    args = parse_args()
    init_logger()
    model = get_lightning_module(args).to(device="cuda")
    run_eval(model)
