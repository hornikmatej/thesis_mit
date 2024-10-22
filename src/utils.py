from typing import List
import wandb


def write_wandb_pred(pred_str: List[str], label_str: List[str], rows_to_log: int = 20):
    current_step = wandb.run.step
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
