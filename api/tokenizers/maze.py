"""Maze layout — the grid side of ``dataset/build_maze_dataset.py``.

That builder flattens a square grid row-major, so a prompt's length is the grid area it
carries: a 30x30 maze is 900 tokens, and nothing here needs a training-time seq_len. Which
characters are legal, and what ids they map to, comes from the vocab map.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .base import Decoded, Encoded, Tokenizer, TokenizerError

#: Filler for rows an editor has trimmed. The maze charset spells an open cell as a space,
#: so re-padding with it restores exactly what was removed.
OPEN_CELL = " "


class MazeTokenizer(Tokenizer):
    name = "maze"
    required_tokens = (OPEN_CELL,)

    def __init__(self, vocab_map: Dict[str, int]) -> None:
        super().__init__(vocab_map)
        charset = "".join(sorted(c for c in self.vocab_map if len(c) == 1))
        self.input_help = (
            f"Square grid over the characters {charset!r} ('#' wall, ' ' open, 'S' start, "
            "'G' goal, 'o' path); rows newline-separated, or one flat line whose length is "
            "a perfect square. Rows shorter than the grid width are right-padded with "
            f"{OPEN_CELL!r}, so editors that trim trailing spaces are fine."
        )

    def encode(self, text: str, puzzle_identifier: int = 0, **kwargs: Any) -> Encoded:
        # Strip only leading/trailing blank lines: an all-open row is legitimately empty
        # once an editor has trimmed its trailing spaces.
        rows = text.replace("\r\n", "\n").strip("\n").split("\n")
        if len(rows) == 1:
            # One flat line: the grid side has to come from its length.
            side = self.square_side(len(rows[0]))
            rows = [rows[0][r * side:(r + 1) * side] for r in range(side)]
        side = len(rows)

        ids: List[int] = []
        for r, row in enumerate(rows):
            if len(row) > side:
                raise TokenizerError(
                    f"row {r} has {len(row)} characters but the grid is {side} rows tall, "
                    "so rows must be at most that wide"
                )
            ids.extend(self.ids_for(row.ljust(side, OPEN_CELL)))
        return Encoded(self.as_tokens(ids), puzzle_identifier)

    def decode(self, ids: Sequence[int], prompt: Encoded) -> Decoded:
        side = self.square_side(len(prompt))
        flat = "".join(self.token_for(i) for i in ids[: side * side])
        return Decoded(text=flat)
