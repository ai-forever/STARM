import argparse
import os
from pathlib import Path
from typing import List

import pydantic
import torch
import torch.distributed as dist
import yaml

from evaluate import (
    load_ema_checkpoint,
    load_state_dict_with_unwrap
)

from pretrain import (
    PretrainConfig,
    create_dataloader,
    init_train_state,
    evaluate
)


class InferenceConfig(pydantic.BaseModel):
    checkpoint: str
    output_dir: str

    save_outputs: List[str] = [
        "inputs",
        "labels",
        "puzzle_identifiers",
        "logits",
        "q_halt_logits",
        "q_continue_logits",
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="HRM inference")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint to evaluate.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where predictions will be saved.",
    )

    return parser.parse_args()


def print_metrics(metrics):
    """Pretty-print evaluation metrics."""

    if isinstance(metrics, tuple):
        metrics = metrics[0]

    print("Evaluation Results")

    for split_name, split_metrics in metrics.items():
        print(f"\n[{split_name}]")

        for metric_name, value in split_metrics.items():

            try:
                value = float(value)
            except Exception:
                pass

            if isinstance(value, float):
                if "accuracy" in metric_name:
                    print(f"  {metric_name:<20} {value:.2%}")
                elif "loss" in metric_name:
                    print(f"  {metric_name:<20} {value:.6f}")
                elif "steps" in metric_name:
                    print(f"  {metric_name:<20} {value:.1f}")
                else:
                    print(f"  {metric_name:<20} {value:.6f}")
            else:
                print(f"  {metric_name:<20} {value}")


def launch():

    args = parse_args()

    cfg = InferenceConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
    )

    rank = 0
    world_size = 1

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    checkpoint_dir = os.path.dirname(cfg.checkpoint)

    with open(os.path.join(checkpoint_dir, "all_config.yaml")) as f:
        config = PretrainConfig(**yaml.safe_load(f))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config.checkpoint_path = str(output_dir)
    config.eval_save_outputs = cfg.save_outputs

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=rank,
        world_size=world_size,
    )

    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=rank,
        world_size=world_size,
    )

    train_state = init_train_state(
        config,
        train_metadata,
        world_size=world_size,
    )

    print(f"Loading checkpoint: {cfg.checkpoint}")

    if "_ema" in os.path.basename(cfg.checkpoint):
        load_ema_checkpoint(train_state.model, cfg.checkpoint)
    else:
        state = torch.load(cfg.checkpoint, map_location="cuda")
        load_state_dict_with_unwrap(train_state.model, state)

    ckpt_filename = os.path.basename(cfg.checkpoint)

    if ckpt_filename.startswith("step_"):
        train_state.step = int(
            ckpt_filename.removeprefix("step_").replace("_ema", "")
        )

    train_state.model.eval()

    print("\nRunning inference...\n")

    metrics = evaluate(
        config,
        train_state,
        eval_loader,
        eval_metadata,
        rank=rank,
        world_size=world_size,
    )

    if rank == 0:
        print(f"Predictions saved to: {output_dir}")
        print_metrics(metrics)


if __name__ == "__main__":
    launch()
