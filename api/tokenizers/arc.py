"""ARC-AGI layout — the 30x30 canvas of ``dataset/build_arc_dataset.py``.

``np_grid_to_seq_translational_augment`` places the grid in the corner of a 30x30 canvas,
writes ``<EOS>`` in the row just below it and the column just right of it, and leaves the rest
as pad. Those markers are how a grid's own size survives a fixed-length encoding, and how
``decode`` recovers the size of a *predicted* grid — so ``<EOS>`` has to exist in the vocab
map. Its spelling is the map's business: ``<EOS>`` and ``<eos>`` both work.

The canvas is what makes this task fixed-length: a grid's cells land at absolute positions
within 30x30, so a prompt is always ``MAX_GRID_SIZE ** 2`` tokens regardless of how small the
grid is. Translation augmentation is training-only, so prompts sit at the top-left corner.

Reading the canvas row-major with pad dropped — which is what ``decode`` returns — gives the
grid's rows separated by ``<EOS>``::

    444000777<eos>404440707<eos>400040777<eos>444440000<eos><eos>...

so an answer, or a row lifted straight out of a ``.preds`` file, can be fed back in as a
prompt: ``encode`` reads both that form and plain newline-separated rows.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

import numpy as np

from .base import Decoded, Encoded, Tokenizer, TokenizerError

EOS_TOKEN = "<EOS>"
MAX_GRID_SIZE = 30  # build_arc_dataset.ARCMaxGridSize
CANVAS = MAX_GRID_SIZE * MAX_GRID_SIZE

#: Splits a prompt on the EOS token however it is spelled, and on newlines, so a grid can be
#: written either way round.
_ROW_SEPARATOR = re.compile(r"<eos>|\n", re.I)


class ArcTokenizer(Tokenizer):
    name = "arc"
    required_tokens = (EOS_TOKEN,) + tuple(str(d) for d in range(10))
    input_help = (
        "Grid of colour digits 0-9, one row per line or with rows separated by '<eos>' "
        f"(cells may also be separated by spaces or commas). Up to "
        f"{MAX_GRID_SIZE}x{MAX_GRID_SIZE}."
    )

    def __init__(self, vocab_map: Dict[str, int]) -> None:
        super().__init__(vocab_map)
        self.eos_token = self.find_token(EOS_TOKEN)
        self.eos_id = self.vocab_map[self.eos_token]

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
                f"grid {height}x{width} exceeds the {MAX_GRID_SIZE}x{MAX_GRID_SIZE} canvas"
            )

        grid = np.full((MAX_GRID_SIZE, MAX_GRID_SIZE), self.pad_id, dtype=np.int32)
        for r, row in enumerate(rows):
            grid[r, :width] = self.ids_for(row)
        if height < MAX_GRID_SIZE:
            grid[height, :width] = self.eos_id
        if width < MAX_GRID_SIZE:
            grid[:height, width] = self.eos_id

        return Encoded(
            self.as_tokens(grid.reshape(-1).tolist()),
            puzzle_identifier,
            meta={"height": height, "width": width},
        )

    def decode(self, ids: Sequence[int], prompt: Encoded) -> Decoded:
        """The canvas read row-major, pad dropped and every ``<EOS>`` kept.

        Where the model put the EOS markers is its own answer about the grid's size, so they
        are shown rather than trimmed or turned into newlines.
        """
        return Decoded(text="".join(self.token_for(i) for i in self.drop_padding(ids[:CANVAS])))
