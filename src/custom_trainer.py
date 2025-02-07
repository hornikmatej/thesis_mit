from transformers import Seq2SeqTrainer
import os
import logging
from typing import Dict, List, Optional, Union, Any

from src.utils import write_wandb_pred

logger_trainer = logging.getLogger(__name__)


class DebugSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom Seq2SeqTrainer that adds debugging capabilities during training and evaluation.
    """

    def __init__(
        self, *args, debug_dir: str = "debug_output", actual_tokenizer=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.debug_dir = debug_dir
        self.actual_tokenizer = actual_tokenizer
        os.makedirs(debug_dir, exist_ok=True)

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Union[Dict[str, float], Dict[str, float]]:
        # Run the standard evaluation loop
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # If we're not just computing loss, run debug analysis
        if not prediction_loss_only:
            self._run_debug_analysis(eval_output, metric_key_prefix)

        return eval_output

    def _run_debug_analysis(self, eval_output: Dict[str, Any], prefix: str):
        """
        Run debug analysis after evaluation.
        """
        predictions = eval_output.predictions
        labels = eval_output.label_ids

        # Replace -100 in labels (ignored index)
        labels[labels == -100] = self.actual_tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = self.actual_tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        label_str = self.actual_tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Also log to W&B
        if self.args.report_to == "wandb":
            write_wandb_pred(pred_str, label_str, self.state.global_step)

        # Run debug analysis using the data collator
        if hasattr(self.data_collator, "save_debug_info"):
            debug_path, sclite_files = self.data_collator.save_debug_info(
                pred_str=pred_str,
                label_str=label_str,
                batch_idx=f"{prefix}_{self.state.global_step}",
            )
            logger_trainer.info(f"Debug information saved to {debug_path}")
            logger_trainer.info(f"SCLITE analysis files: {sclite_files}")
