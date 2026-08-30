"""Per-domain tokenizers for STARM checkpoints.

One module per training domain. A tokenizer is given the task's vocab map (token -> id)
and contributes only the layout logic around it — see ``base.py``.

To add a task: write ``<task>.py`` with a ``Tokenizer`` subclass, then register it in
``_TOKENIZERS`` and add a ``data_path`` hint below so checkpoints of that task are
recognised automatically.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Type

from .arc import ArcTokenizer
from .arithmetic import ArithmeticTokenizer
from .base import Decoded, Encoded, Tokenizer, TokenizerError
from .game_of_life import GameOfLifeTokenizer
from .maze import MazeTokenizer
from .sudoku import SudokuTokenizer

__all__ = [
    "TASKS",
    "VOCAB_MAP_FILENAME",
    "ArcTokenizer",
    "ArithmeticTokenizer",
    "Decoded",
    "Encoded",
    "GameOfLifeTokenizer",
    "MazeTokenizer",
    "SudokuTokenizer",
    "Tokenizer",
    "TokenizerError",
    "BUILTIN_VOCAB_MAPS",
    "build_tokenizer",
    "builtin_vocab_map",
    "find_vocab_map",
    "load_vocab_map",
    "task_from_data_path",
]

#: What ``build_gol_dataset.get_vocab_map`` names its persisted vocabulary.
VOCAB_MAP_FILENAME = "vocab_map.json"

#: Vocabularies shipped with the server, one per task, in ``api/vocab_maps``. Each is the map
#: its ``dataset/build_*_dataset.py`` assigns — those ids are constants in the builder, not
#: something a run chooses — so a checkpoint without its own file can still be served. It is
#: the last resort: an explicit path, the checkpoint directory and the training data_path all
#: come first, and the size is checked against the checkpoint either way.
#:
#: Game of Life is the one task with two, since its builder has two formats; the built-in is
#: ``bin``, its default. A run built with ``format_name: rle`` needs
#: ``api/vocab_maps/game_of_life_rle.json`` passed explicitly.
BUILTIN_VOCAB_MAPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "vocab_maps")

_TOKENIZERS: Dict[str, Type[Tokenizer]] = {
    SudokuTokenizer.name: SudokuTokenizer,
    MazeTokenizer.name: MazeTokenizer,
    ArcTokenizer.name: ArcTokenizer,
    ArithmeticTokenizer.name: ArithmeticTokenizer,
    GameOfLifeTokenizer.name: GameOfLifeTokenizer,
}

TASKS = tuple(_TOKENIZERS)

#: Substrings of the training ``data_path`` that identify a task. Ordered most specific
#: first, so "game_of_life" is not shadowed by a shorter key.
_DATA_PATH_HINTS = (
    ("game_of_life", GameOfLifeTokenizer.name),
    ("game-of-life", GameOfLifeTokenizer.name),
    ("arithmetic", ArithmeticTokenizer.name),
    ("sudoku", SudokuTokenizer.name),
    ("maze", MazeTokenizer.name),
    ("gol", GameOfLifeTokenizer.name),
    ("arc", ArcTokenizer.name),
)


def task_from_data_path(data_path: str) -> Optional[str]:
    """Guess the task from the ``data_path`` recorded in all_config.yaml."""
    needle = data_path.lower().replace("\\", "/")
    for hint, task in _DATA_PATH_HINTS:
        if hint in needle:
            return task
    return None


def load_vocab_map(path: str) -> Dict[str, int]:
    """Read a token -> id mapping as written at dataset build time."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            vocab_map = json.load(f)
    except OSError as exc:
        raise TokenizerError(f"cannot read vocab map {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TokenizerError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(vocab_map, dict):
        raise TokenizerError(f"{path} does not contain a token -> id mapping")
    return vocab_map


def find_vocab_map(*directories: Optional[str]) -> Optional[str]:
    """First ``vocab_map.json`` among the given directories, if any."""
    for directory in directories:
        if not directory:
            continue
        candidate = os.path.join(directory, VOCAB_MAP_FILENAME)
        if os.path.isfile(candidate):
            return candidate
    return None


def builtin_vocab_map(task: Optional[str]) -> Optional[str]:
    """The vocabulary this repo ships for ``task``, if it has one."""
    if not task:
        return None
    candidate = os.path.join(BUILTIN_VOCAB_MAPS, f"{task}.json")
    return candidate if os.path.isfile(candidate) else None


def build_tokenizer(
    task: Optional[str] = None,
    data_path: Optional[str] = None,
    vocab_map_file: Optional[str] = None,
    vocab_size: Optional[int] = None,
) -> Tokenizer:
    """Resolve a tokenizer from a task name (or the training data path) and a vocab map.

    ``vocab_size``, when known from the checkpoint's embedding matrix, is cross-checked
    against the map so a wrong task or a stale map fails at startup rather than returning
    garbage.
    """
    if task is None:
        if data_path is None:
            raise TokenizerError("cannot resolve a tokenizer without a task or a data_path")
        task = task_from_data_path(data_path)
        if task is None:
            raise TokenizerError(
                f"cannot infer the task from data_path {data_path!r}; "
                f"pass --task explicitly (one of: {', '.join(TASKS)})"
            )
    if task not in _TOKENIZERS:
        raise TokenizerError(f"unknown task {task!r}; expected one of: {', '.join(TASKS)}")

    if not vocab_map_file:
        vocab_map_file = builtin_vocab_map(task)
    if not vocab_map_file:
        raise TokenizerError(
            f"no {VOCAB_MAP_FILENAME} found for the {task!r} task, and none is shipped for "
            "it. Pass --vocab-map, or copy the file into the checkpoint directory."
        )

    tokenizer = _TOKENIZERS[task](load_vocab_map(vocab_map_file))

    if vocab_size is not None and tokenizer.vocab_size != vocab_size:
        raise TokenizerError(
            f"{vocab_map_file} defines {tokenizer.vocab_size} tokens but this "
            f"checkpoint's embedding matrix has {vocab_size} — wrong map, or wrong --task"
        )
    return tokenizer
