"""Shared contract for the per-domain tokenizers.

STARM models are non-autoregressive: a checkpoint consumes a token sequence and emits one
logit vector per position. There is no shared vocabulary — every
``dataset/build_*_dataset.py`` builds its own — so a tokenizer here is **not** a
reimplementation of one. It is given the task's ``vocab_map`` (token -> id, as
``build_gol_dataset.get_vocab_map`` persists to ``vocab_map.json``) and supplies only the
layout logic the map cannot express.

``encode`` returns each prompt at its **natural** length. For the grid tasks that length is
fixed by the layout itself (81 cells for sudoku, the 30x30 canvas for ARC, the grid area for
maze); for arithmetic and game-of-life it is simply how long the prompt is. Padding up to the
checkpoint's ``seq_len`` is the engine's job, and ``decode`` trims the pad tail back off, so
the padding never reaches the caller.

Each subclass declares the special tokens it needs from the map, so a map that cannot
serve a task is rejected at startup rather than mid-request.

``Decoded.text`` is canonical — feeding it back to ``encode`` round-trips — for every task
whose answer is itself a task statement. Arithmetic is the exception: its answer is the
operators alone, which is not an expression, so it does not re-encode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

#: How the builders spell pad. A task's own ``pad_token`` is tried first.
PAD_TOKENS = ("<PAD>", "p")


@dataclass(frozen=True)
class Encoded:
    """One tokenized prompt, at its natural length."""

    tokens: np.ndarray  # (length,) int32
    puzzle_identifier: int = 0
    #: Anything ``decode`` needs from the prompt (arithmetic, for one, substitutes
    #: predicted operators back into the original expression).
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])


@dataclass(frozen=True)
class Decoded:
    """A detokenized model answer."""

    text: str


class TokenizerError(ValueError):
    """A malformed prompt, or a vocab map that cannot serve this task."""


class Tokenizer:
    """Base class. One subclass per training domain, in its own module."""

    name: str = ""
    #: Token standing for padding. Most builders write "<PAD>"; arithmetic uses 'p'.
    pad_token: str = "<PAD>"
    #: Tokens beyond pad that this task cannot work without.
    required_tokens: Sequence[str] = ()
    input_help: str = ""

    def __init__(self, vocab_map: Dict[str, int]) -> None:
        if not isinstance(vocab_map, dict) or not vocab_map:
            raise TokenizerError("vocab_map must be a non-empty token -> id mapping")
        for token, value in vocab_map.items():
            if not isinstance(value, int):
                raise TokenizerError(f"vocab_map[{token!r}] is {value!r}, expected an int")

        self.vocab_map = dict(vocab_map)
        self.vocab_size = len(self.vocab_map)
        self._id_to_token = {v: k for k, v in self.vocab_map.items()}
        # Special tokens are spelled differently from run to run — "<EOS>" or "<eos>",
        # "<PAD>" or "<pad>" — so they are matched exactly first, then case-insensitively.
        self._folded = {k.casefold(): k for k in reversed(list(self.vocab_map))}

        missing = [t for t in self.required_tokens if self.find_token(t) is None]
        if missing:
            raise TokenizerError(
                f"vocab_map is missing {missing}, which the {self.name!r} task needs; "
                f"it has: {sorted(vocab_map)}"
            )

        # Pad is written "<PAD>" by most builders and "p" by arithmetic, and a run may use
        # either spelling whatever the task is, so both are tried. Every builder records
        # pad_id 0 in its dataset metadata, which is the fallback for a map that names
        # neither (sudoku's ids are digit + 1).
        self.pad_token = self.find_token(self.pad_token, *PAD_TOKENS) or self.pad_token
        self.pad_id = self.vocab_map.get(self.pad_token, 0)
        # What a short sequence is filled out with. Pad for every task but ARC, whose
        # sequences are padded with <EOS>; a subclass overrides it after calling super().
        self.fill_id = self.pad_id

    # -- to be implemented per domain ------------------------------------------

    def encode(self, text: str, **kwargs: Any) -> Encoded:
        """Tokenize one prompt at its natural length."""
        raise NotImplementedError

    def decode(self, ids: Sequence[int], prompt: Encoded) -> Decoded:
        """Turn one row of predicted ids back into an answer."""
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------------

    def find_token(self, *names: str) -> Optional[str]:
        """The first of ``names`` the map carries, as the map spells it.

        Exact matches win; failing that, case is ignored, so a run writing ``<eos>`` serves a
        task asking for ``<EOS>``. Returns None if the map has none of them.
        """
        for name in names:
            if name in self.vocab_map:
                return name
        for name in names:
            folded = self._folded.get(name.casefold())
            if folded is not None:
                return folded
        return None

    def token_id(self, token: str) -> int:
        name = self.find_token(token)
        if name is None:
            raise TokenizerError(
                f"the vocab map has no token {token!r}; it has: {sorted(self.vocab_map)}"
            )
        return self.vocab_map[name]

    def ids_for(self, tokens: Iterable[str]) -> List[int]:
        return [self.token_id(t) for t in tokens]

    def token_for(self, token_id: int, unknown: str = "?") -> str:
        """Map a predicted id back to its token, or ``unknown`` if the map has no such id."""
        return self._id_to_token.get(int(token_id), unknown)

    def as_tokens(self, ids: List[int], length: Optional[int] = None) -> np.ndarray:
        """Build the id array, optionally filled out to ``length`` with ``fill_id``."""
        if length is not None:
            if len(ids) > length:
                raise TokenizerError(f"prompt is {len(ids)} tokens, longer than {length}")
            ids = ids + [self.fill_id] * (length - len(ids))
        return np.array(ids, dtype=np.int32)

    def pad_batch(self, prompts: Sequence[Encoded], length: int) -> np.ndarray:
        """Stack prompts into one (batch, length) array, right-padding the shorter ones."""
        return np.stack(
            [self.as_tokens(p.tokens.tolist(), length) for p in prompts]
        ).astype(np.int32)

    def trim_padding(self, ids: Sequence[int]) -> List[int]:
        """Drop trailing pad ids; labels are right-padded by every builder."""
        trimmed = [int(i) for i in ids]
        while trimmed and trimmed[-1] == self.pad_id:
            trimmed.pop()
        return trimmed

    def drop_padding(self, ids: Sequence[int]) -> List[int]:
        """Drop every pad id, wherever it sits.

        An answer is the model's own tokens and nothing else: pad is the one thing removed,
        because it is the sequence being long enough rather than something predicted. What
        survives is passed through as-is — a token that looks wrong for the task is the
        model's output and is shown, not corrected.
        """
        return [int(i) for i in ids if int(i) != self.pad_id]

    @staticmethod
    def square_side(length: int) -> int:
        side = math.isqrt(length)
        if side * side != length:
            raise TokenizerError(f"{length} tokens is not a square grid")
        return side
