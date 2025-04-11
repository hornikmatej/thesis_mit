import os
from functools import partial
from typing import List

import sentencepiece as spm
import torch
import torchaudio
from common import (
    Batch,
    batch_by_token_count,
    FunctionalModule,
    GlobalStatsNormalization,
    piecewise_linear_log,
    post_process_hypos,
    spectrogram_transform,
    WarmupLR,
)
from pytorch_lightning import LightningModule
from torchaudio.models import emformer_rnnt_base, RNNTBeamSearch
from typing import Optional, List, Tuple
from torchaudio.models.rnnt import _Predictor, _Joiner
from torchaudio.models import RNNT
from transformers import Wav2Vec2Model

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
        source_lengths,
        state: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        raise NotImplementedError("No streaming for Wav2Vec2Model.")
    
    def forward(
        self,
        sources: torch.Tensor,
        source_lengths,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        predictor_state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        source_encodings = self.transcriber(
            input_values=sources,
        )
        source_lengths = torch.full((source_encodings.size(0),), source_encodings.size(1), dtype=torch.int32)
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
    encoding_dim: int,
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
        global_stats_path: str,
    ):
        super().__init__()

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
        self.warmup_lr_scheduler = WarmupLR(self.optimizer, 1000)

        self.train_data_pipeline = torch.nn.Sequential(
            FunctionalModule(piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
            torchaudio.transforms.FrequencyMasking(27),
            torchaudio.transforms.FrequencyMasking(27),
            torchaudio.transforms.TimeMasking(100, p=0.2),
            torchaudio.transforms.TimeMasking(100, p=0.2),
            FunctionalModule(partial(torch.nn.functional.pad, pad=(0, 4))),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
        )
        self.valid_data_pipeline = torch.nn.Sequential(
            FunctionalModule(piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
            FunctionalModule(partial(torch.nn.functional.pad, pad=(0, 4))),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
        )

        self.librispeech_path = librispeech_path

        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.blank_idx = self.sp_model.get_piece_size()

    def _extract_labels(self, samples: List):
        targets = [self.sp_model.encode(sample[2].lower()) for sample in samples]
        lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(elem) for elem in targets],
            batch_first=True,
            padding_value=1.0,
        ).to(dtype=torch.int32)
        return targets, lengths

    def _train_extract_features(self, samples: List):
        mel_features = [spectrogram_transform(sample[0].squeeze()).transpose(1, 0) for sample in samples]
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        features = self.train_data_pipeline(features)
        lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
        return features, lengths

    def _valid_extract_features(self, samples: List):
        mel_features = [spectrogram_transform(sample[0].squeeze()).transpose(1, 0) for sample in samples]
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        features = self.valid_data_pipeline(features)
        lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
        return features, lengths

    def _train_collate_fn(self, samples: List):
        features, feature_lengths = self._train_extract_features(samples)
        targets, target_lengths = self._extract_labels(samples)
        return Batch(features, feature_lengths, targets, target_lengths)

    def _valid_collate_fn(self, samples: List):
        features, feature_lengths = self._valid_extract_features(samples)
        targets, target_lengths = self._extract_labels(samples)
        return Batch(features, feature_lengths, targets, target_lengths)

    def _test_collate_fn(self, samples: List):
        return self._valid_collate_fn(samples), [sample[2] for sample in samples]

    def _step(self, batch, batch_idx, step_type):
        if batch is None:
            return None

        prepended_targets = batch.targets.new_empty([batch.targets.size(0), batch.targets.size(1) + 1])
        prepended_targets[:, 1:] = batch.targets
        prepended_targets[:, 0] = self.blank_idx
        prepended_target_lengths = batch.target_lengths + 1
        output, src_lengths, _, _ = self.model(
            batch.features,
            batch.feature_lengths,
            prepended_targets,
            prepended_target_lengths,
        )
        loss = self.loss(output, batch.targets, src_lengths, batch.target_lengths)
        self.log(f"Losses/{step_type}_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return (
            [self.optimizer],
            [
                {"scheduler": self.warmup_lr_scheduler, "interval": "step"},
            ],
        )

    def forward(self, batch: Batch):
        decoder = RNNTBeamSearch(self.model, self.blank_idx)
        hypotheses = decoder(batch.features.to(self.device), batch.feature_lengths.to(self.device), 20)
        return post_process_hypos(hypotheses, self.sp_model)[0][0]

    def training_step(self, batch: Batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch_tuple, batch_idx):
        return self._step(batch_tuple[0], batch_idx, "test")

    def train_dataloader(self):
        dataset = torch.utils.data.ConcatDataset(
            [
                # CustomDataset(
                #     torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="train-clean-360"),
                #     1000,
                # ),
                CustomDataset(
                    torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="train-clean-100", download=True,),
                    1000,
                ),
                # CustomDataset(
                #     torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="train-other-500"),
                #     1000,
                # ),
            ]
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            collate_fn=self._train_collate_fn,
            num_workers=10,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        dataset = torch.utils.data.ConcatDataset(
            [
                CustomDataset(
                    torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="dev-clean", download=True,),
                    1000,
                ),
                # CustomDataset(
                #     torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="dev-other"),
                #     1000,
                # ),
            ]
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            collate_fn=self._valid_collate_fn,
            num_workers=10,
        )
        return dataloader

    def test_dataloader(self):
        dataset = torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="test-clean", download=True,)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=self._test_collate_fn)
        return dataloader
