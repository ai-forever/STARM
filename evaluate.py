import os
from typing import List, Optional

import pydantic
import torch
import torch.distributed as dist
import yaml
from clearml import Task
from omegaconf import OmegaConf

from pretrain import PretrainConfig, init_train_state, evaluate, create_dataloader


class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    extra_steps: int = 8
    tags: str = "hrm hierarchial_reasoning evaluate"
    project_name: str = "GigaChat/RnD_conveyor/HierarchicalReasoning"
    run_name: Optional[str] = None

    save_outputs: List[str] = ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits", "q_continue_logits"]


def load_state_dict_with_unwrap(
    model: torch.nn.Module,
    state_dict: dict,
    strict: bool = False,
):
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    if model_keys == ckpt_keys:
        model.load_state_dict(state_dict, strict=strict, assign=True)
        return

    stripped = {
        k.removeprefix("_orig_mod."): v
        for k, v in state_dict.items()
    }

    if set(stripped.keys()) == model_keys:
        model.load_state_dict(stripped, strict=strict, assign=True)
        return

    wrapped = {
        f"_orig_mod.{k}": v
        for k, v in state_dict.items()
    }

    if set(wrapped.keys()) == model_keys:
        model.load_state_dict(wrapped, strict=strict, assign=True)
        return

    model.load_state_dict(state_dict, strict=False, assign=True)


def load_ema_checkpoint(
    model: torch.nn.Module,
    ema_checkpoint_path: str,
):
    base_checkpoint = ema_checkpoint_path.removesuffix("_ema")

    print(f"Loading base checkpoint: {base_checkpoint}")
    base_state = torch.load(base_checkpoint, map_location="cuda", weights_only=True)
    load_state_dict_with_unwrap(model, base_state)

    print(f"Loading EMA checkpoint: {ema_checkpoint_path}")
    ema_state = torch.load(ema_checkpoint_path, map_location="cuda", weights_only=True)
    load_state_dict_with_unwrap(model, ema_state)


def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))  # type: ignore

    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    with open(os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

        config.eval_save_outputs = eval_cfg.save_outputs
        config.checkpoint_path = os.path.dirname(eval_cfg.checkpoint)

    ckpt_filename = os.path.basename(eval_cfg.checkpoint)
    is_ema_checkpoint = "_ema" in ckpt_filename

    train_state_step = 0
    if ckpt_filename.startswith("step_"):
        step_part = ckpt_filename.removeprefix("step_")
        train_state_step = int(step_part.replace('_ema', ''))

    # Dataloader
    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1,
                                                     global_batch_size=config.global_batch_size, rank=RANK,
                                                     world_size=WORLD_SIZE)
    eval_loader, eval_metadata = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1,
                                                   global_batch_size=config.global_batch_size, rank=RANK,
                                                   world_size=WORLD_SIZE)

    # Models
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Load weights (with smart prefix handling)
    print(f"Loading checkpoint: {eval_cfg.checkpoint}")
    print(f"  Is EMA: {is_ema_checkpoint}")

    if is_ema_checkpoint:
        # EMA checkpoint — load via special function
        load_ema_checkpoint(train_state.model, eval_cfg.checkpoint)
    else:
        # Normal checkpoint
        state_dict = torch.load(eval_cfg.checkpoint, map_location="cuda")
        load_state_dict_with_unwrap(train_state.model, state_dict)

    train_state.step = train_state_step

    if eval_cfg.run_name is None:
        run_name = 'evaluate_' + str(os.path.basename(config.checkpoint_path))
        run_name += '_' + str(ckpt_filename.removeprefix("step_")) + '_' + str(eval_cfg.extra_steps)
    else:
        run_name = eval_cfg.run_name

    # Evaluate
    print("Starting evaluation")

    train_state.model.eval()
    config.extra_steps = eval_cfg.extra_steps
    metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

    if (RANK == 0) and (metrics[1] is not None):
        print("Report to clearml")
        print(metrics[0])
        print(metrics[1])

        task = Task.init(
            project_name=eval_cfg.project_name or "Default Project",
            task_name=run_name,
            # Auto connect argparse, git, etc.
            tags=[] if eval_cfg.tags is None else eval_cfg.tags.split(),
            auto_connect_frameworks=True,
        )

        conveyor_info = dict()
        if total_gpu_number := os.environ.get('MLS_JOB_TOTAL_GPU', None):
            conveyor_info["world_size"] = total_gpu_number
        if region := os.environ.get('MLS_JOB_REGION_NAME', None):
            conveyor_info["region"] = region
        task.connect(
            conveyor_info,
            name="Conveyor_info"
        )
        logger = task.get_logger()

        for graph_title, steps_data in metrics[1].items():
            # graph_title -> Plot title in ClearML (e.g., '1-3', '4-6')

            for step_key, metric_dict in steps_data.items():
                # step_key -> Step (e.g., '1', '2'... 'best_orig')

                for metric_name, value in metric_dict.items():
                    # metric_name -> Series name (e.g., 'exact')
                    # value -> Value (np.float64)

                    val_float = float(value)

                    if step_key.isdigit():
                        logger.report_scalar(
                            title=graph_title,
                            series=metric_name,
                            value=val_float,
                            iteration=int(step_key)
                        )
                    else:
                        logger.report_scalar(
                            title=graph_title + '_' + "Others",
                            series=metric_name + '_' + step_key,
                            value=val_float,
                            iteration=0
                        )
        task.close()


if __name__ == "__main__":
    launch()
