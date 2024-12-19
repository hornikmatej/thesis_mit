from typing import List

import wandb
from transformers import TrainerCallback


def write_wandb_pred(
    pred_str: List[str], label_str: List[str], current_step: int, rows_to_log: int = 20
):
    columns = ["id", "label_str", "hyp_str"]
    wandb.log(
        {
            f"eval_predictions/step_{int(current_step)}": wandb.Table(
                columns=columns,
                data=[
                    [i, ref, hyp]
                    for i, hyp, ref in zip(
                        range(min(len(pred_str), rows_to_log)), pred_str, label_str
                    )
                ],
            )
        },
        current_step,
    )


def count_parameters(model):
    encoder_params = (
        sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        if hasattr(model, "encoder")
        else 0
    )
    decoder_params = (
        sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
        if hasattr(model, "decoder")
        else 0
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return encoder_params, decoder_params, total_params


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()
