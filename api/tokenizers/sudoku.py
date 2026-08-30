"""Sudoku layout — the grid side of ``dataset/build_sudoku_dataset.py``.

That builder reads the 81-character puzzle string and replaces '.' with '0'; the vocab map
supplies the digit -> id part (it stores ``digit + 1``, leaving 0 for pad). A prompt is
always exactly the 81 cells, so nothing here depends on a training-time seq_len.

Which *character* stands for an empty cell is the map's business, not this module's: some
runs write it as '0', others keep the puzzle string's own '.', and both name the same id.
The same goes for pad, written either '<PAD>' or 'p'. Whichever the map uses is what
``decode`` returns, and ``encode`` accepts both spellings regardless.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from .base import Decoded, Encoded, Tokenizer, TokenizerError

SIDE = 9
CELLS = SIDE * SIDE


#: Spellings of the empty cell, most common first; a map has to carry one of them. Pad is
#: handled by the base class, which knows both spellings of it.
BLANK_TOKENS = ("0", ".")


class SudokuTokenizer(Tokenizer):
    name = "sudoku"
    #: 1-9 are spelled the same way everywhere; the empty cell is checked separately, since
    #: it answers to more than one name.
    required_tokens = tuple(str(d) for d in range(1, 10))
    input_help = (
        "81 cells in row-major order, digits 1-9 with '.' or '0' for blanks. "
        "Whitespace and newlines are ignored, so a 9x9 block works too."
    )

    def __init__(self, vocab_map: Dict[str, int]) -> None:
        super().__init__(vocab_map)

        self.blank = next((t for t in BLANK_TOKENS if t in self.vocab_map), None)
        if self.blank is None:
            raise TokenizerError(
                f"vocab_map has no token for an empty cell (tried {list(BLANK_TOKENS)}); "
                f"it has: {sorted(self.vocab_map)}"
            )

    def encode(self, text: str, puzzle_identifier: int = 0, **kwargs: Any) -> Encoded:
        cells = [self.blank if c in BLANK_TOKENS else c for c in text if not c.isspace()]
        # self.blank is exempt: it is a legal cell, and it is not a digit when spelled '.'.
        bad = {c for c in cells if c != self.blank and not c.isdigit()}
        if bad:
            raise TokenizerError(f"unexpected characters in sudoku grid: {sorted(bad)}")
        if len(cells) != CELLS:
            raise TokenizerError(f"expected {CELLS} cells, got {len(cells)}")
        return Encoded(self.as_tokens(self.ids_for(cells)), puzzle_identifier)

    def decode(self, ids: Sequence[int], prompt: Encoded) -> Decoded:
        digits = "".join(self.token_for(i) for i in ids[:CELLS])
        return Decoded(text=digits)
