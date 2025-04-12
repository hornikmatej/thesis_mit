import os
from typing import List
from collections import namedtuple

import sentencepiece as spm
import torch
import torchaudio
from common import (
    batch_by_token_count,
    post_process_hypos,
    WarmupLR,
)
from pytorch_lightning import LightningModule
from torchaudio.models import RNNTBeamSearch
from typing import Optional, List, Tuple
from torchaudio.models.rnnt import _Predictor, _Joiner
from torchaudio.models import RNNT
from transformers import Wav2Vec2Model, AutoFeatureExtractor

Batch = namedtuple("Batch", ["features", "attention_mask", "feature_lengths", "targets", "target_lengths"])

class Wav2Vec2HiddenStates(Wav2Vec2Model):
    """
    Wav2Vec2Model with a modified forward method to return just last hidden state.
    """
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        outputs = super().forward(
            input_values=input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs.last_hidden_state

class Wav2vec2RNNT(RNNT):
    @torch.jit.export
    def transcribe_streaming(
        self,
        sources: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        state: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        raise NotImplementedError("No streaming for Wav2Vec2Model.")
    
    @torch.jit.export
    def transcribe(
        self,
        sources: torch.Tensor,
        sources_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: for this RNNTBeamSearch needs to be modified to accept attention_mask
        return self.transcriber(input_values=sources.squeeze(1)), sources_lengths

    def forward(
        self,
        sources: torch.Tensor,
        attention_mask: Optional[torch.Tensor], # Expects attention_mask from feature_extractor
        source_lengths,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        predictor_state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        source_encodings = self.transcriber(
            input_values=sources,
            attention_mask=attention_mask,
        ) # [B, T_encoded, D_hidden]
        source_lengths = torch.full((source_encodings.size(0),), source_encodings.size(1), dtype=torch.int32).to(source_encodings.device)
        target_encodings, target_lengths, predictor_state = self.predictor(
            input=targets,
            lengths=target_lengths,
            state=predictor_state,
        )
        output, source_lengths, target_lengths = self.joiner(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
        )

        return (
            output,
            source_lengths,
            target_lengths,
            predictor_state,
        )

def wav2vec2_rnnt_model(
    *,
    encoding_dim: int, # Wav2Vec2-base output dim
    num_symbols: int,
    symbol_embedding_dim: int,
    num_lstm_layers: int,
    lstm_layer_norm: bool,
    lstm_layer_norm_epsilon: float,
    lstm_dropout: float,
) -> Wav2vec2RNNT:
    encoder = Wav2Vec2HiddenStates.from_pretrained("facebook/wav2vec2-base")
    predictor = _Predictor(
        num_symbols,
        encoding_dim,
        symbol_embedding_dim=symbol_embedding_dim,
        num_lstm_layers=num_lstm_layers,
        lstm_hidden_dim=symbol_embedding_dim,
        lstm_layer_norm=lstm_layer_norm,
        lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
        lstm_dropout=lstm_dropout,
    )
    joiner = _Joiner(encoding_dim, num_symbols)
    return Wav2vec2RNNT(encoder, predictor, joiner)

# Re-introduce CustomDataset
class CustomDataset(torch.utils.data.Dataset):
    r"""Sort LibriSpeech samples by target length and batch to max token count."""

    def __init__(self, base_dataset, max_token_limit):
        super().__init__()
        self.base_dataset = base_dataset

        fileid_to_target_length = {}
        idx_target_lengths = [
            (idx, self._target_length(fileid, fileid_to_target_length))
            for idx, fileid in enumerate(self.base_dataset._walker)
        ]

        assert len(idx_target_lengths) > 0

        idx_target_lengths = sorted(idx_target_lengths, key=lambda x: x[1], reverse=True)

        assert max_token_limit >= idx_target_lengths[0][1]

        self.batches = batch_by_token_count(idx_target_lengths, max_token_limit)

    def _target_length(self, fileid, fileid_to_target_length):
        if fileid not in fileid_to_target_length:
            speaker_id, chapter_id, _ = fileid.split("-")

            file_text = speaker_id + "-" + chapter_id + self.base_dataset._ext_txt
            file_text = os.path.join(self.base_dataset._path, speaker_id, chapter_id, file_text)

            with open(file_text) as ft:
                for line in ft:
                    fileid_text, transcript = line.strip().split(" ", 1)
                    fileid_to_target_length[fileid_text] = len(transcript)

        return fileid_to_target_length[fileid]

    def __getitem__(self, idx):
        return [self.base_dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)


class LibriSpeechRNNTModuleWav2Vec2(LightningModule):
    def __init__(
        self,
        *,
        librispeech_path: str,
        sp_model_path: str,
        sort_by_length: bool = True,  # Whether to sort samples by length
    ):
        super().__init__()
        self.validation_step_outputs = []

        self.model = wav2vec2_rnnt_model(
            encoding_dim=768, # Wav2Vec2-base output dim
            num_symbols=4097,
            symbol_embedding_dim=512,
            num_lstm_layers=3,
            lstm_layer_norm=True,
            lstm_layer_norm_epsilon=1e-3,
            lstm_dropout=0.3,
        )
        self.loss = torchaudio.transforms.RNNTLoss(reduction="sum", clamp=1.0)
        # Restore original optimizer and scheduler settings
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, eps=1e-8) # betas=(0.9, 0.999)
        self.warmup_lr_scheduler = WarmupLR(self.optimizer, 200)

        # --- Feature Extractor ---
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.sampling_rate = self.feature_extractor.sampling_rate

        self.train_data_pipeline = None
        self.valid_data_pipeline = None

        self.librispeech_path = librispeech_path
        # Removed self.batch_size
        self.sort_by_length = sort_by_length

        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.blank_idx = self.sp_model.get_piece_size()
        print(f"Blank index: {self.blank_idx}")

    def _extract_labels(self, samples: List):
        targets = [self.sp_model.encode(sample[2].lower()) for sample in samples]
        lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
        # Correct padding value for RNNT loss
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(elem) for elem in targets],
            batch_first=True,
            padding_value=1.0,
        ).to(dtype=torch.int32)
        return targets, lengths
    
    def _extract_features(self, samples: List):
        """Extracts features using Wav2Vec2 feature extractor."""
        raw_audio = [sample[0].squeeze().numpy() for sample in samples] # Assuming sample[0] is waveform tensor
        # assume all samples in the batch have the target sampling rate
        # // 320 because the feature extractor sub-samples by 320 for wav2vec2-base
        feature_lengths = torch.tensor([len(audio) for audio in raw_audio], dtype=torch.int32) # length in samples

        # Process batch with feature extractor
        processed = self.feature_extractor(
            raw_audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        features = processed.input_values
        attention_mask = processed.attention_mask
        return features, attention_mask, feature_lengths

    _train_extract_features = _extract_features
    _valid_extract_features = _extract_features

    def _collate_fn(self, samples: List):
        """Collates samples into a batch using Wav2Vec2 feature extraction."""
        if not samples:
            return None
        
        features, attention_mask, feature_lengths = self._extract_features(samples)

        targets, target_lengths = self._extract_labels(samples)
        return Batch(features, attention_mask, feature_lengths, targets, target_lengths)

    # Restore original train/valid collate functions
    def _train_collate_fn(self, samples: List):
         return self._collate_fn(samples)

    def _valid_collate_fn(self, samples: List):
         return self._collate_fn(samples)

    # Restore original test collate function logic
    def _test_collate_fn(self, samples: List):
        # Test data usually doesn't need shuffling or complex batching, process one by one
        # However, if using CustomDataset for test, it might return batches
        if isinstance(samples[0], list): # Handle case where CustomDataset returns list of samples
             batch_samples = samples[0]
        else: # Handle case where DataLoader returns single sample in a list
             batch_samples = samples

        # Pass is_train=False if needed by _collate_fn, otherwise remove
        batch = self._collate_fn(batch_samples) 
        transcripts = [s[2] for s in batch_samples]
        return batch, transcripts # Return batch and reference transcripts

    def _step(self, batch, batch_idx, step_type):
        if batch is None:
            print(f"Warning: Skipping empty batch at index {batch_idx} during {step_type}")
            return None

        prepended_targets = batch.targets.new_empty([batch.targets.size(0), batch.targets.size(1) + 1])
        prepended_targets[:, 1:] = batch.targets
        prepended_targets[:, 0] = self.blank_idx
        prepended_target_lengths = batch.target_lengths + 1
        output, src_lengths, _, _ = self.model(
            sources=batch.features,
            attention_mask=batch.attention_mask,
            source_lengths=batch.feature_lengths,
            targets=prepended_targets,
            target_lengths=prepended_target_lengths,
        )
        # print(f"Output shape: {output.shape}, Source lengths: {src_lengths.shape}, Targets shape: {batch.targets.shape}, Target lengths: {batch.target_lengths.shape}")
        # print(f"Source lenghts: {src_lengths}, Target lengths: {batch.target_lengths}")
        # Potential reduction issue if batch_size is 1
        loss = self.loss(output, batch.targets, src_lengths, batch.target_lengths)
        if step_type == "train":
            self.log(f"Losses/{step_type}_loss", loss, on_step=True, on_epoch=True, batch_size=batch.targets.size(0), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return (
            [self.optimizer],
            [
                {"scheduler": self.warmup_lr_scheduler, "interval": "step"},
            ],
        )

    def forward(self, batch: Batch):
        if batch is None:
            print("Warning: Skipping empty batch during forward pass")
            return []
        self.model.eval()
        decoder = RNNTBeamSearch(self.model, self.blank_idx, step_max_tokens=128)
        hypotheses = decoder(batch.features.to(self.device), batch.feature_lengths.to(self.device), 5)
        return post_process_hypos(hypotheses, self.sp_model)[0][0]

    def on_after_backward(self):
        total_norm = 0.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        # Log gradient norm. Adjust logging arguments as needed.
        self.log("grad_norm", total_norm, on_step=True, on_epoch=True, prog_bar=True)

    def training_step(self, batch: Batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "val")
        self.validation_step_outputs.append(loss.detach())
        return loss
    
    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg_loss = torch.mean(torch.stack(self.validation_step_outputs))
            self.log("Losses/val_loss", avg_loss, prog_bar=True)
            self.validation_step_outputs.clear()
        else:
            print("Warning: No validation step outputs to average.")

    def test_step(self, batch_tuple, batch_idx):
        return self._step(batch_tuple[0], batch_idx, "test")

    # --- Dataloaders ---
    # Restore original dataloader logic
    def train_dataloader(self):
        if self.sort_by_length:
            # Using CustomDataset or a similar sorted batching strategy
            dataset = torch.utils.data.ConcatDataset(
                [
                    CustomDataset( # Use CustomDataset again
                        torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="train-clean-100", download=True,),
                        1000, # Example token limit
                    ),
                    # Add other datasets wrapped in CustomDataset if needed
                ]
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=None,  # No additional batching since dataset returns batches
                collate_fn=self._train_collate_fn,
                num_workers=8,
                shuffle=False,  # No shuffle as the dataset is already sorted/batched
                pin_memory=True,
            )
        else:
            # Use standard PyTorch DataLoader batching (requires _train_collate_fn_single)
            dataset = torch.utils.data.ConcatDataset(
                [
                    torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="train-clean-100", download=True,),
                ]
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=32, # Example batch size if not sorting
                collate_fn=self._train_collate_fn_single, # Use the single-sample collate
                num_workers=8,
                shuffle=True,
                pin_memory=True,
            )
        return dataloader

    def val_dataloader(self):
        if self.sort_by_length:
            # Using CustomDataset or a similar sorted batching strategy
            dataset = torch.utils.data.ConcatDataset(
                [
                     CustomDataset( # Use CustomDataset again
                        torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="dev-clean", download=True,),
                        1000, # Example token limit
                    ),
                    # Add other datasets wrapped in CustomDataset if needed
                ]
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=None,
                collate_fn=self._valid_collate_fn,
                num_workers=8,
                pin_memory=True,
            )
        else:
            # Use standard PyTorch DataLoader batching (requires _valid_collate_fn_single)
            dataset = torch.utils.data.ConcatDataset(
                [
                    torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="dev-clean", download=True,),
                ]
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=32, # Example batch size if not sorting
                collate_fn=self._valid_collate_fn_single, # Use the single-sample collate
                num_workers=8,
                shuffle=False,
                pin_memory=True,
            )
        return dataloader
    
    # Re-introduce single-sample collate functions if needed for sort_by_length=False case
    def _train_collate_fn_single(self, samples: List):
        """Collate function for individual samples when not using CustomDataset"""
        return self._collate_fn(samples)
    
    def _valid_collate_fn_single(self, samples: List):
        """Collate function for individual samples when not using CustomDataset"""
        return self._collate_fn(samples)

    def test_dataloader(self):
        # Restore original test dataloader (assuming it didn't use batch_size directly)
        dataset = torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="test-clean", download=True,)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1, # Often test is run sample by sample or with custom batching
            collate_fn=self._test_collate_fn, 
            pin_memory=True,
            num_workers=8, # Added num_workers for consistency
        )
        return dataloader
