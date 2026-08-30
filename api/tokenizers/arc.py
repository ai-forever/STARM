"""ARC-AGI layout — rows of colour digits separated by ``<EOS>``.

A sequence is the grid read row-major, with ``<EOS>`` closing every row, and the tail padded
with more ``<EOS>`` up to the run's ``seq_len``::

    444000777<eos>404440707<eos>400040777<eos>444440000<eos><eos><eos>...

So ``<EOS>`` carries the grid's shape — it is where a row ends — and a *predicted* grid's
shape is read back the same way. The token has to exist in the vocab map, and its spelling
is the map's business: ``<EOS>`` and ``<eos>`` both work.

The filler is ``<EOS>`` rather than pad, which is why this tokenizer overrides ``fill_id``.
Padding is therefore indistinguishable from a run of empty rows, and that is fine: both mean
"the grid ended here". ``decode`` passes every ``<EOS>`` through rather than trimming, so an
answer shows exactly the tokens the model emitted.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from .base import Decoded, Encoded, Tokenizer, TokenizerError

EOS_TOKEN = "<EOS>"
MAX_GRID_SIZE = 30  # build_arc_dataset.ARCMaxGridSize

#: Splits a prompt on the EOS token however it is spelled, and on newlines, so a grid can be
#: written either way round.
_ROW_SEPARATOR = re.compile(r"<eos>|\n", re.I)


class ArcTokenizer(Tokenizer):
    name = "arc"
    required_tokens = (EOS_TOKEN,) + tuple(str(d) for d in range(10))
    input_help = (
        "Grid of colour digits 0-9, one row per line or with rows separated by '<eos>' "
        f"(cells may also be separated by spaces or commas). Up to {MAX_GRID_SIZE} rows."
    )

    def __init__(self, vocab_map: Dict[str, int]) -> None:
        super().__init__(vocab_map)
        self.eos_token = self.find_token(EOS_TOKEN)
        self.eos_id = self.vocab_map[self.eos_token]
        # A batch is padded out with EOS, the way the dataset's own sequences are.
        self.fill_id = self.eos_id

    def encode(self, text: str, puzzle_identifier: int = 0, **kwargs: Any) -> Encoded:
        rows: List[List[str]] = []
        for segment in _ROW_SEPARATOR.split(text):
            segment = segment.strip()
            if not segment:
                continue  # a trailing run of <eos> is padding, not an empty row
            cells = re.split(r"[\s,]+", segment) if re.search(r"[\s,]", segment) else list(segment)
            cells = [c for c in cells if c != ""]
            if any(not c.isdigit() or len(c) != 1 for c in cells):
                raise TokenizerError(f"cells must be single digits 0-9, got row {segment!r}")
            rows.append(cells)

        if not rows:
            raise TokenizerError("empty grid")
        width = len(rows[0])
        if any(len(r) != width for r in rows):
            raise TokenizerError("all rows must have the same width")
        height = len(rows)
        if height > MAX_GRID_SIZE or width > MAX_GRID_SIZE:
            raise TokenizerError(
                f"grid {height}x{width} exceeds the {MAX_GRID_SIZE}x{MAX_GRID_SIZE} maximum"
            )

        ids: List[int] = []
        for row in rows:
            ids.extend(self.ids_for(row))
            ids.append(self.eos_id)

        return Encoded(
            self.as_tokens(ids), puzzle_identifier, meta={"height": height, "width": width}
        )

    def decode(self, ids: Sequence[int], prompt: Encoded) -> Decoded:
        """The model's tokens, pad removed and every ``<EOS>`` kept.

        Row breaks are part of what the model predicts, so they are shown rather than turned
        into newlines, and the trailing run of them is the model's own answer about where the
        grid ends.
        """
        return Decoded(text="".join(self.token_for(i) for i in self.drop_padding(ids)))
