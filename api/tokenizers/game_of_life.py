"""Game of Life layout — the pattern side of ``dataset/build_gol_dataset.py``.

That builder tokenizes ``pattern`` + ``<SEP>`` + ``str(step)`` as the input and the pattern
after that many generations as the label. Its vocabulary is the one task that already ships
as data: ``get_vocab_map`` persists it to ``vocab_map.json`` and reuses that file on later
runs.

Note that the answer is not bound to the prompt's length. The builder pads both to
``max(len(input), len(output))`` over the whole dataset precisely because a Life pattern can
grow. That constant is the run's ``seq_len``, which the engine pads every prompt to, so the
room a grown pattern needs is there — the same room training gave it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .base import Decoded, Encoded, Tokenizer, TokenizerError

SEP = "<SEP>"


class GameOfLifeTokenizer(Tokenizer):
    name = "game_of_life"
    required_tokens = (SEP,) + tuple(str(d) for d in range(10))

    def __init__(self, vocab_map: Dict[str, int]) -> None:
        super().__init__(vocab_map)
        self._pattern_chars = "".join(
            sorted(c for c in self.vocab_map if len(c) == 1 and not c.isdigit())
        )
        self.input_help = (
            "Life pattern followed by the generation count, separated by '|': 'bbo$obb|3'. "
            f"Pattern characters: {self._pattern_chars!r} ('b' dead, 'o' alive, '$' end of "
            "row). The count can also be passed as a separate 'step' field."
        )

    def encode(self, text: str, puzzle_identifier: int = 0, step: Optional[int] = None,
               **kwargs: Any) -> Encoded:
        pattern = "".join(text.split())
        if step is None:
            if "|" not in pattern:
                raise TokenizerError(
                    "provide the generation count either as '<pattern>|<step>' or in the "
                    "'step' field"
                )
            pattern, _, step_str = pattern.rpartition("|")
            if not step_str.isdigit():
                raise TokenizerError(f"step must be a non-negative integer, got {step_str!r}")
            step = int(step_str)
        elif "|" in pattern:
            raise TokenizerError("step given twice: in the text and in the 'step' field")
        if step < 0:
            raise TokenizerError("step must be non-negative")
        if not pattern:
            raise TokenizerError("empty pattern")

        bad = {c for c in pattern if c not in self.vocab_map or len(c) != 1 or c.isdigit()}
        if bad:
            raise TokenizerError(
                f"unexpected characters {sorted(bad)} in pattern; allowed: {self._pattern_chars!r}"
            )

        ids: List[int] = self.ids_for(pattern)
        ids.append(self.token_id(SEP))
        ids.extend(self.ids_for(str(step)))

        return Encoded(
            self.as_tokens(ids), puzzle_identifier, meta={"pattern": pattern, "step": step}
        )

    def decode(self, ids: Sequence[int], prompt: Encoded) -> Decoded:
        """The predicted pattern: the model's tokens with pad removed, nothing substituted.

        A checkpoint that emits ``<SEP>`` mid-answer, or a character the layout does not
        expect, has it shown as it is — that is what it predicted.
        """
        pattern = "".join(self.token_for(i) for i in self.drop_padding(ids))
        return Decoded(text=pattern)
