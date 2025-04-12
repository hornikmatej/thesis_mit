#!/usr/bin/env python3
import logging
import pathlib
from argparse import ArgumentParser
import torch
from common import MODEL_TYPE_LIBRISPEECH
from librispeech.lightning import LibriSpeechRNNTModule
from librispeech.lightning_wav2vec2 import LibriSpeechRNNTModuleWav2Vec2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from dotenv import load_dotenv


def verify_model_setup(model):
    """Verify that the model is correctly set up before training."""
    logging.info("=== Model Setup Verification ===")
    logging.info(f"Model type: {model.__class__.__name__}")
    

    
    # Check dataloader
    logging.info("Testing dataloader...")
    try:
        train_dataloader = model.train_dataloader()
        for i, batch in enumerate(train_dataloader):
            if batch is not None:
                logging.info(f"Batch {i+1} type: {type(batch)}")
            if hasattr(batch, "_fields"):
                for field in batch._fields:
                    value = getattr(batch, field)
                    if hasattr(value, "shape"):
                        logging.info(f"  {field} shape: {value.shape}, dtype: {value.dtype}")
            if i >= 9:
                break
        logging.info("Dataloader test successful")
    except Exception as e:
        logging.error(f"Dataloader test failed: {e}")
    
    # Test forward pass
    logging.info("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        if batch is not None:
            # Move batch to the same device as model
            if hasattr(batch, "_fields"):
                # Ensure batch tensors are moved to the correct device
                device = next(model.parameters()).device # Get model device
                batch_on_device = type(batch)(*(getattr(batch, f).to(device) if torch.is_tensor(getattr(batch, f)) else getattr(batch, f) for f in batch._fields))
                loss = model._step(batch_on_device, 0, "test") # Use batch_on_device
                logging.info(f"Forward pass successful, loss: {loss.item()}")
            else:
                logging.warning("Batch structure not recognized for forward pass test.")
        else:
            logging.warning("Last batch was None, skipping forward pass test.")

    
    logging.info("=== Verification Complete ===")


def get_trainer(args):
    checkpoint_dir = args.exp_dir / "checkpoints"
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=True,
        verbose=True,
    )
    # Initialize WandbLogger
    wandb_logger = WandbLogger(project="librispeech-rnnt", log_model="all") # Set your project name

    callbacks = [
        train_checkpoint,
    ]
    return Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=args.epochs,
        log_every_n_steps=1,
        accelerator="gpu",
        accumulate_grad_batches=2,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        val_check_interval=100,
        logger=wandb_logger,
        precision="16-mixed",
    )


def get_lightning_module(args):
    if args.mtype == "emformer":
        return LibriSpeechRNNTModule(
            librispeech_path="libri_dataset/",
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )
    elif args.mtype == "wav2vec2":
        return LibriSpeechRNNTModuleWav2Vec2(
            librispeech_path="libri_dataset/",
            sp_model_path=str(args.sp_model_path),
            # Removed batch_size argument
        )

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--mtype",
        default="emformer",
        type=str,
        choices=["emformer", "wav2vec2"],
        help="Model type to use. (Default: 'emformer')",
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
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
        default=1,
        type=int,
        help="Number of epochs to train for. (Default: 1)",
    )
    parser.add_argument(
        "--gradient-clip-val", default=2.0, type=float, help="Value to clip gradient values to. (Default: 2.0)"
    )
    return parser.parse_args()


def init_logger():
    fmt = "%(message)s"
    level = logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    load_dotenv() # Load environment variables from .env file
    args = parse_args()
    init_logger()
    model = get_lightning_module(args)
    
    # verify_model_setup(model) # Restore verify call
    # exit(0) # Restore exit call
    trainer = get_trainer(args)
    trainer.fit(model)
