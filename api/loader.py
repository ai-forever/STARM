"""Rebuild a trained model from a checkpoint directory, local or on the Hugging Face Hub.

Everything about the model comes from the checkpoint, not from the repo's current
``config/`` or ``models/hrm/``:

* ``all_config.yaml`` supplies the architecture knobs — written by ``save_code_and_config``
  at train time.
* ``<module>.py`` in the same directory supplies the architecture *code*, also copied there
  at train time. Checkpoints outlive refactors, so a run trained months ago loads against
  the model code it was trained with rather than today's.

A Hub repo (``sapientinc/HRM-checkpoint-sudoku-extreme``, or ``hf://`` + that) is downloaded
and then treated exactly like a local directory. Such a repo carries only ``all_config.yaml``
and the weights, so the architecture comes from this repo's ``models/`` — the same fallback a
local run without its code copy takes — and ``seq_len`` and the vocab map have to be passed in.

Those copied modules do import the repo's low-level primitives (``models.layers``,
``models.common``, ``models.sparse_embedding``), so the repo has to be importable — but no
architecture setting or hyperparameter is read from it.

``save_train_state`` stores ``torch.compile(ACTLossHead(model)).state_dict()``, so saved keys
carry a ``_orig_mod.model.`` prefix that is stripped to load into a bare ACT wrapper.

``vocab_size`` and ``num_puzzle_identifiers`` are injected from dataset metadata at train time
and so are absent from all_config.yaml; they are read back off the checkpoint's own tensors.
``seq_len`` comes from all_config.yaml and is the model's own dimension: the carry, the
dropout masks and the rotary table are all built for it here, once, and nothing downstream
touches the model again.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from torch import nn

logger = logging.getLogger(__name__)

_STEP_RE = re.compile(r"^step_(\d+)(_ema)?$")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CheckpointError(RuntimeError):
    """The checkpoint directory is missing something the server needs."""


@dataclass
class LoadedModel:
    model: nn.Module
    arch_name: str
    arch_source: str
    checkpoint_file: str
    checkpoint_dir: str
    is_ema: bool
    step: int
    device: torch.device
    vocab_size: int
    num_puzzle_identifiers: int
    #: The length this checkpoint runs at, from all_config.yaml. Every prompt is padded to it.
    seq_len: int
    halt_max_steps: int
    pos_encodings: str
    data_path: Optional[str]
    train_config: Dict[str, Any] = field(default_factory=dict)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


# -- reading the checkpoint directory ------------------------------------------

#: Weight files a Hub repo is likely to use, since it has no step_N naming convention.
_FALLBACK_WEIGHT_NAMES = ("checkpoint", "model.pt", "pytorch_model.bin", "model.safetensors")

#: A Hugging Face repo id: exactly one slash, no path separators beyond it, no spaces.
_HF_REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")

HF_PREFIX = "hf://"


def looks_like_hf_repo(path: str) -> bool:
    """Whether ``path`` names a Hub repo rather than something on disk.

    An explicit ``hf://`` always wins. Otherwise ``org/name`` is taken as a repo id only when
    no such file or directory exists locally, so a relative path never turns into a download.
    """
    if path.startswith(HF_PREFIX):
        return True
    return bool(_HF_REPO_ID.match(path)) and not os.path.exists(path)


def fetch_hf_checkpoint(path: str, revision: Optional[str] = None) -> str:
    """Download a Hub repo and return the local directory holding it."""
    repo_id = path[len(HF_PREFIX):] if path.startswith(HF_PREFIX) else path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise CheckpointError(
            f"{repo_id} looks like a Hugging Face repo, but huggingface_hub is not installed: "
            f"{exc}. Install it, or pass a local directory"
        ) from exc

    logger.info("Downloading %s from the Hugging Face Hub", repo_id)
    try:
        return snapshot_download(repo_id=repo_id, revision=revision)
    except Exception as exc:  # huggingface_hub raises a family of its own errors
        raise CheckpointError(f"cannot download {repo_id}: {exc}") from exc


def resolve_checkpoint(path: str, prefer_ema: bool = True) -> Tuple[str, str]:
    """Return ``(checkpoint_file, checkpoint_dir)`` for a file or a run directory.

    Given a directory, the highest ``step_N`` is picked, preferring EMA weights when
    present: runs with ``use_ema`` report their headline metrics from the EMA copy.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return path, os.path.dirname(path)
    if not os.path.isdir(path):
        raise CheckpointError(f"no such file or directory: {path}")

    steps: Dict[int, Dict[bool, str]] = {}
    for entry in os.listdir(path):
        match = _STEP_RE.match(entry)
        if match and os.path.isfile(os.path.join(path, entry)):
            steps.setdefault(int(match.group(1)), {})[bool(match.group(2))] = entry

    if not steps:
        # A Hub repo names its weights whatever it likes; step_N is this repo's convention.
        for name in _FALLBACK_WEIGHT_NAMES:
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate, path
        raise CheckpointError(
            f"{path} holds no step_N weight file, nor any of {list(_FALLBACK_WEIGHT_NAMES)}. "
            "Point --path_directory at the checkpoint file itself if it is named differently."
        )

    latest = max(steps)
    variants = steps[latest]
    name = variants[True] if (prefer_ema and True in variants) else variants[False]
    return os.path.join(path, name), path


def load_train_config(checkpoint_dir: str) -> Dict[str, Any]:
    """Read the run's all_config.yaml, the sole source of architecture settings."""
    config_file = os.path.join(checkpoint_dir, "all_config.yaml")
    if not os.path.isfile(config_file):
        raise CheckpointError(
            f"{config_file} not found — the architecture is read from it, so the checkpoint "
            "has to stay in the directory training wrote it to"
        )
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "arch" not in config:
        raise CheckpointError(f"{config_file} is not a STARM training config")
    return config


def resolve_arch_source(checkpoint_dir: str, arch_name: str) -> Tuple[Optional[str], str, str]:
    """Locate the architecture code for ``arch_name``.

    ``arch.name`` looks like ``"hrm.hrm_act_v2DG@HierarchicalReasoningModel_ACTV2DG"``.
    ``save_code_and_config`` copies that module into the run directory under its bare
    basename, so ``hrm.hrm_act_v2DG`` is looked for as ``<dir>/hrm_act_v2DG.py``.

    Returns ``(file_path_or_None, module_path, class_name)``; a None path means the
    checkpoint predates the code copy and the repo's own module has to be imported.
    """
    if "@" not in arch_name:
        raise CheckpointError(f"malformed arch.name {arch_name!r}, expected 'module@Class'")
    module_path, class_name = arch_name.split("@", 1)
    candidate = os.path.join(checkpoint_dir, f"{module_path.rsplit('.', 1)[-1]}.py")
    return (candidate if os.path.isfile(candidate) else None), module_path, class_name


def import_arch_class(
    file_path: Optional[str], module_path: str, class_name: str
) -> Tuple[type, str]:
    """Import the architecture class, from the checkpoint's copy when it has one.

    The copied module imports the repo's primitives (``models.layers`` and friends), so the
    repo is put on ``sys.path`` first. It is loaded under a synthetic module name to avoid
    colliding with an already-imported ``models.hrm.<module>``.
    """
    if REPO_ROOT not in sys.path:
        sys.path.append(REPO_ROOT)

    if file_path is None:
        logger.warning(
            "%s.py was not copied into the checkpoint directory; falling back to the repo's "
            "models.%s, which may have changed since this run was trained",
            module_path.rsplit(".", 1)[-1],
            module_path,
        )
        module = importlib.import_module(f"models.{module_path}")
        source = f"models.{module_path} (repo)"
    else:
        synthetic_name = f"starm_checkpoint_arch_{abs(hash(file_path)):x}"
        spec = importlib.util.spec_from_file_location(synthetic_name, file_path)
        if spec is None or spec.loader is None:
            raise CheckpointError(f"cannot load a Python module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        # Registered before exec so dataclasses and type hints inside resolve normally.
        sys.modules[synthetic_name] = module
        try:
            spec.loader.exec_module(module)
        except ImportError as exc:
            raise CheckpointError(
                f"{file_path} could not be imported: {exc}. The checkpoint carries its "
                "architecture module but not the primitives it imports, so the repo's "
                "models/ package has to provide a compatible version."
            ) from exc
        source = file_path

    try:
        return getattr(module, class_name), source
    except AttributeError as exc:
        raise CheckpointError(f"{source} defines no class {class_name!r}") from exc


# -- state dict ----------------------------------------------------------------

def _normalize_key(key: str) -> str:
    """Strip the ``torch.compile`` and loss-head prefixes off a saved key."""
    while key.startswith("_orig_mod."):
        key = key[len("_orig_mod."):]
    if key.startswith("model."):
        key = key[len("model."):]
    return key


def _normalize_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {_normalize_key(k): v for k, v in state.items()}


# -- loading -------------------------------------------------------------------

def resolve_device(spec: str) -> torch.device:
    """Turn a device spec into a device, and make it the process's current one.

    Accepts ``cuda``, ``cuda:2``, a bare index (``2``), or ``cpu``. One process serves one
    checkpoint on one GPU, so a machine with four of them runs four servers on four ports —
    ``--device cuda:0 --port 8080``, ``--device cuda:1 --port 8081``, and so on.

    ``torch.cuda.set_device`` matters beyond bookkeeping: a checkpoint ships its own model
    source, and anything in it that says ``.cuda()`` or a bare ``"cuda"`` resolves to the
    *current* device. Setting it makes that mean the requested GPU rather than GPU 0.
    """
    spec = str(spec).strip()
    if spec.isdigit():
        spec = f"cuda:{spec}"

    try:
        device = torch.device(spec)
    except (RuntimeError, ValueError) as exc:
        raise CheckpointError(f"unusable device {spec!r}: {exc}") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise CheckpointError(
                f"device {spec!r} was requested but torch reports no CUDA device; the models "
                "import flash-attn and do not run on CPU"
            )
        count = torch.cuda.device_count()
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= count:
            raise CheckpointError(
                f"device {spec!r} does not exist: this machine has {count} CUDA device(s), "
                f"so the valid indices are 0..{count - 1}"
            )
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)

    return device


def _read_seq_len(
    train_config: Dict[str, Any],
    arch: Dict[str, Any],
    checkpoint_dir: str,
    override: Optional[int] = None,
) -> int:
    """Read the run's ``seq_len``, from ``override`` if given and all_config.yaml otherwise.

    It is the length the checkpoint was trained at, and the only length it can run at: the
    rotary table is precomputed for ``seq_len + puzzle_emb_len`` positions and the carry is
    allocated to match, so the input has to be exactly that long. Accepted at the top level
    or under ``arch:``, whichever the run wrote it to.
    """
    value = arch.pop("seq_len", None)
    if value is None:
        value = train_config.get("seq_len")
    if override is not None:
        value = override
    if value is None:
        raise CheckpointError(
            f"{checkpoint_dir}/all_config.yaml has no seq_len. It is the sequence length the "
            "run was trained at, and the model cannot be built without it — pass --seq-len, "
            "or add it to the config (top level or under arch:), copying the value from the "
            "training dataset's dataset.json"
        )
    try:
        seq_len = int(value)
    except (TypeError, ValueError):
        raise CheckpointError(f"seq_len in all_config.yaml is {value!r}, expected an int") from None
    if seq_len <= 0:
        raise CheckpointError(f"seq_len in all_config.yaml is {seq_len}, expected a positive int")
    return seq_len


def load_model(
    path: str,
    device: str = "cuda",
    prefer_ema: bool = True,
    fallback_vocab_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    revision: Optional[str] = None,
) -> LoadedModel:
    """Instantiate the checkpoint's architecture and load its weights.

    The model is *not* wrapped in ``ACTLossHead`` (no labels, no loss) and *not* compiled:
    ``torch.compile`` costs minutes of startup and buys nothing for single-request inference.
    """
    # Resolved first: a bad device should fail before gigabytes of weights are read.
    torch_device = resolve_device(device)

    if looks_like_hf_repo(path):
        path = fetch_hf_checkpoint(path, revision=revision)

    checkpoint_file, checkpoint_dir = resolve_checkpoint(path, prefer_ema=prefer_ema)
    is_ema = checkpoint_file.endswith("_ema")
    step_match = _STEP_RE.match(os.path.basename(checkpoint_file))
    step = int(step_match.group(1)) if step_match else -1

    train_config = load_train_config(checkpoint_dir)
    arch = dict(train_config["arch"])
    arch_name = arch.pop("name")
    arch.pop("loss", None)  # inference needs no loss head
    data_path = train_config.get("data_path")
    seq_len = _read_seq_len(train_config, arch, checkpoint_dir, override=seq_len)

    # An EMA file is a partial shadow (trainable params + puzzle_emb.weights), so base
    # weights load first and the shadow is overlaid on top.
    base_file = checkpoint_file[: -len("_ema")] if is_ema else checkpoint_file
    if is_ema and not os.path.isfile(base_file):
        raise CheckpointError(
            f"{checkpoint_file} is an EMA shadow but its base checkpoint {base_file} is missing"
        )

    logger.info("Loading %s (arch %s)", checkpoint_file, arch_name)
    state = _normalize_state_dict(torch.load(base_file, map_location="cpu", weights_only=True))

    embed = state.get("inner.embed_tokens.embedding_weight")
    if embed is not None:
        vocab_size = int(embed.shape[0])
    elif fallback_vocab_size is not None:
        vocab_size = fallback_vocab_size
    else:
        raise CheckpointError(
            "checkpoint has no inner.embed_tokens.embedding_weight to read vocab_size from; "
            f"keys look like: {sorted(state)[:5]}"
        )

    puzzle_weights = state.get("inner.puzzle_emb.weights")
    # Every builder except ARC writes a single "<blank>" identifier.
    num_puzzle_identifiers = int(puzzle_weights.shape[0]) if puzzle_weights is not None else 1

    model_cfg = dict(
        arch,
        # batch_size only sizes CastedSparseEmbedding's non-persistent local_weights buffer,
        # a training-mode staging area: in eval the puzzle embedding indexes self.weights
        # directly. Serving batches are whatever a request carries.
        batch_size=1,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_puzzle_identifiers=num_puzzle_identifiers,
        causal=False,  # non-autoregressive
    )

    arch_file, module_path, class_name = resolve_arch_source(checkpoint_dir, arch_name)
    model_cls, arch_source = import_arch_class(arch_file, module_path, class_name)

    with torch.device(torch_device):
        model: nn.Module = model_cls(model_cfg)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise CheckpointError(f"checkpoint has {len(unexpected)} unexpected keys: {unexpected[:5]}")
    if missing:
        raise CheckpointError(f"checkpoint is missing {len(missing)} keys: {missing[:5]}")

    if is_ema:
        _overlay_ema(model, checkpoint_file)

    model.eval()
    model.to(torch_device)

    return LoadedModel(
        model=model,
        arch_name=arch_name,
        arch_source=arch_source,
        checkpoint_file=checkpoint_file,
        checkpoint_dir=checkpoint_dir,
        is_ema=is_ema,
        step=step,
        device=torch_device,
        vocab_size=vocab_size,
        num_puzzle_identifiers=num_puzzle_identifiers,
        seq_len=seq_len,
        halt_max_steps=int(model.config.halt_max_steps),
        pos_encodings=str(arch.get("pos_encodings", "rope")),
        data_path=data_path,
        train_config=train_config,
    )


def _overlay_ema(model: nn.Module, ema_file: str) -> None:
    """Apply an EMA shadow on top of already-loaded base weights.

    ``EMAHelper.state_dict`` returns its shadow, which ``EMAHelper.register`` fills with the
    trainable parameters plus any buffer matching ``puzzle_emb.weights`` — so this is a partial
    update, and the base weights have to be in place first.
    """
    shadow = _normalize_state_dict(torch.load(ema_file, map_location="cpu", weights_only=True))
    own = model.state_dict()
    applied = 0
    with torch.no_grad():
        for key, value in shadow.items():
            target = own.get(key)
            if target is None:
                logger.warning("EMA key %s has no counterpart in the model, skipping", key)
                continue
            target.copy_(value)
            applied += 1
    logger.info("Applied %d/%d EMA tensors", applied, len(shadow))
