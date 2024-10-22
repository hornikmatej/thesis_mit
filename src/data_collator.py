import pandas as pd
import os
import subprocess
import logging
from typing import List, Dict, Union, Any
import torch

logger_collator = logging.getLogger(__name__)


class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Enhanced data collator that includes additional debugging and evaluation capabilities using SCTK.

    Args:
        processor: The processor used for processing the data
        decoder_start_token_id: The begin-of-sentence token ID of the decoder
        forward_attention_mask: Whether to return attention_mask
        debug_output_dir: Directory to save debug files
        sclite_path: Path to the SCLITE executable
    """

    def __init__(
        self,
        processor: Any,
        decoder_start_token_id: int,
        forward_attention_mask: bool,
        debug_output_dir: str = "debug_output",
        sclite_path: str = "/home/matej/fitvut/dp_mit/SCTK/bin/sclite",
    ):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id
        self.forward_attention_mask = forward_attention_mask
        self.debug_output_dir = debug_output_dir
        self.sclite_path = sclite_path

        # Create debug output directory if it doesn't exist
        os.makedirs(debug_output_dir, exist_ok=True)

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Original collation logic
        model_input_name = self.processor.model_input_names[0]
        input_features = [
            {model_input_name: feature[model_input_name]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor(
                [feature["attention_mask"] for feature in features]
            )

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

    def save_debug_info(
        self, pred_str: List[str], label_str: List[str], batch_idx: int = 0
    ):
        """
        Save debug information and run SCLITE evaluation.

        Args:
            pred_str: List of prediction strings
            label_str: List of reference/label strings
            batch_idx: Batch index for unique file naming
        """
        # Save predictions and references to CSV
        debug_path = os.path.join(self.debug_output_dir, f"debug_batch_{batch_idx}.csv")
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
        sclite_cmd = f"{self.sclite_path} -F -D -i wsj -r {sclite_files[1]} trn -h {sclite_files[0]} trn -o snt sum dtl"
        logger_collator.info(f"Running SCLITE evaluation with command: {sclite_cmd}")
        process = subprocess.Popen(sclite_cmd.split())  # nosec

        try:
            process.wait(30)  # Wait up to 30 seconds for SCLITE to complete
        except subprocess.TimeoutExpired:
            process.kill()
            logger_collator.warning("Sclite evaluation timed out.")

        return debug_path, sclite_files
