#!/usr/bin/env python3
"""CLI entry point: serve a local STARM checkpoint over HTTP.

    host_model --path_directory /path/to/checkpoints/<project>/<run> --port 8080

``--path_directory`` accepts either the run directory (the newest ``step_N`` is picked,
preferring EMA weights when present) or a single checkpoint file. Either way the
directory has to still hold the ``all_config.yaml`` that training wrote there, since
that is where the architecture comes from.

One process serves one checkpoint on one GPU. A four-GPU machine runs four of them:

    host_model --path_directory .../sudoku/run     --device cuda:0 --port 8080 &
    host_model --path_directory .../arithmetic/run --device cuda:1 --port 8081 &
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.tokenizers import TASKS  # noqa: E402

logger = logging.getLogger("host_model")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="host_model",
        description="Serve a trained STARM checkpoint over an HTTP API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path_directory",
        "--path-directory",
        dest="path_directory",
        required=True,
        help=(
            "Run directory holding step_N + all_config.yaml, a checkpoint file, or a Hugging "
            "Face repo id ('sapientinc/HRM-checkpoint-sudoku-extreme', or hf:// + that). A Hub "
            "repo carries no code and usually no seq_len or vocab map, so --seq-len and "
            "--vocab-map are needed with it"
        ),
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        help=(
            "Sequence length the run was trained at, overriding all_config.yaml. Every prompt "
            "is padded to it; the value is the training dataset's dataset.json seq_len"
        ),
    )
    parser.add_argument(
        "--revision",
        help="Hugging Face revision (branch, tag or commit) to download; defaults to main",
    )
    parser.add_argument("--port", type=int, default=8080, help="TCP port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Address to bind")
    parser.add_argument(
        "--device",
        default="cuda",
        help=(
            "Which GPU to load onto: 'cuda:1', or just '1'. One server, one checkpoint, one "
            "GPU — run several on different ports to fill a multi-GPU machine. flash-attn "
            "makes CUDA the only real option"
        ),
    )
    parser.add_argument(
        "--task",
        choices=TASKS,
        help="Override the task detected from the checkpoint's training data_path",
    )
    parser.add_argument(
        "--vocab-map",
        help=(
            "The vocab_map.json (token -> id) the dataset was built with. Looked for in the "
            "checkpoint directory and the training data_path when omitted, falling back to "
            "the map this repo ships for the task (api/vocab_maps/<task>.json)"
        ),
    )
    parser.add_argument(
        "--identifiers",
        help=(
            "ARC only: the identifiers.json the dataset build wrote — the list of task ids "
            "puzzle_id is resolved against. Looked for next to the checkpoint when omitted, "
            "which a Hugging Face repo never carries"
        ),
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Load the base weights even when an EMA shadow is present",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import uvicorn
    except ImportError:
        logger.error(
            "fastapi/uvicorn are not installed (they are not part of the training image). "
            "Install them with: pip install -r %s",
            REPO_ROOT / "api" / "requirements.txt",
        )
        return 1

    try:
        # models/layers.py imports flash_attn at module scope, so a host without that
        # CUDA-only wheel fails right here rather than at first request.
        from api.loader import (
            CheckpointError,
            fetch_hf_checkpoint,
            load_model,
            load_train_config,
            looks_like_hf_repo,
            resolve_checkpoint,
        )
    except ImportError as exc:
        if "flash_attn" in str(exc):
            logger.error(
                "flash-attn is needed to import the model code but is unavailable: %s\n"
                "Serve from the project dockerfile image, which builds it, or install it "
                "for your CUDA version.",
                exc,
            )
            return 1
        raise

    from api.engine import StarmEngine
    from api.server import create_app
    from api.tokenizers import (
        TokenizerError,
        build_tokenizer,
        builtin_vocab_map,
        find_vocab_map,
        task_from_data_path,
    )

    # Read the run config first: the task it implies determines the tokenizer, and the
    # tokenizer is the authority on vocab_size, which the model config needs.
    try:
        # A Hub repo is downloaded once, here, and everything downstream sees a local path.
        path = args.path_directory
        if looks_like_hf_repo(path):
            path = fetch_hf_checkpoint(path, revision=args.revision)
        _, checkpoint_dir = resolve_checkpoint(path, prefer_ema=not args.no_ema)
        train_config = load_train_config(checkpoint_dir)
    except CheckpointError as exc:
        logger.error("%s", exc)
        return 1

    data_path = train_config.get("data_path")
    task = args.task or (task_from_data_path(data_path) if data_path else None)

    vocab_map_file = args.vocab_map or find_vocab_map(checkpoint_dir, data_path)
    if not vocab_map_file:
        vocab_map_file = builtin_vocab_map(task)
        if vocab_map_file:
            logger.info(
                "No vocab_map.json alongside the checkpoint; using the one this repo ships "
                "for %r: %s", task, vocab_map_file
            )

    try:
        tokenizer = build_tokenizer(
            task=task,
            data_path=data_path,
            vocab_map_file=vocab_map_file,
        )
    except TokenizerError as exc:
        logger.error("%s", exc)
        return 1

    try:
        loaded = load_model(
            path,
            device=args.device,
            prefer_ema=not args.no_ema,
            fallback_vocab_size=tokenizer.vocab_size,
            seq_len=args.seq_len,
        )
    except CheckpointError as exc:
        logger.error("%s", exc)
        return 1

    if loaded.vocab_size != tokenizer.vocab_size:
        logger.error(
            "vocabulary mismatch: the '%s' tokenizer produces %d token ids but the "
            "checkpoint's embedding matrix has %d. Wrong --task, or a vocab_map.json "
            "that does not match this run.",
            tokenizer.name,
            tokenizer.vocab_size,
            loaded.vocab_size,
        )
        return 1

    try:
        engine = StarmEngine(loaded, tokenizer, identifiers_file=args.identifiers)
    except TokenizerError as exc:
        # An ARC checkpoint with no identifiers.json cannot resolve a single task id.
        logger.error("%s", exc)
        return 1

    info = engine.info()
    logger.info(
        "Loaded %s (%s, step %s%s) on %s: %d params, task=%s, vocab=%d, halt_max_steps=%d",
        Path(info["checkpoint"]).name,
        info["arch"],
        info["step"],
        ", ema" if info["ema"] else "",
        info["device"],
        info["num_parameters"],
        info["task"],
        info["vocab_size"],
        info["halt_max_steps"],
    )
    logger.info("Model code: %s", info["arch_source"])
    logger.info("Interactive docs: http://%s:%d/docs", args.host, args.port)

    uvicorn.run(
        create_app(engine),
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
