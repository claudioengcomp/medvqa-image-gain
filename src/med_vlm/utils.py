import math
from typing import Any, Dict, Optional


def log_metrics(
    *,
    metrics: Dict[str, float],
    step: int,
    writer: Optional[Any] = None,
    prefix: str = "",
    print_to_console: bool = True,
) -> None:
    """
    Logs a dictionary of metrics.

    Example:
        log_metrics(
            metrics={"loss": 1.23, "lr": 1e-4},
            step=100,
            writer=writer,
            prefix="train"
        )

    Will log:
        train/loss
        train/lr
    """
    if prefix:
        prefix = prefix.rstrip("/") + "/"

    if print_to_console:
        msg = f"[step {step}] "
        msg += " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(msg)

    if writer is not None:
        writer.log(
            {f"{prefix}{key}": value for key, value in metrics.items()}, step=step
        )


def adalora_compute_total_steps(
    num_examples: int, batch_size: int, grad_accum: int, epochs: int
) -> int:
    steps_per_epoch = math.ceil(num_examples / batch_size)
    optim_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    return optim_steps_per_epoch * epochs


def blip2_vqa_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def medvlm_vqa_prompt(question: str, answer: str | None = None):
    if answer is None:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer},
            ],
        },
    ]
