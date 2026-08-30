"""Arithmetic layout — the masking side of ``dataset/build_arithmetic_dataset.py``.

The task is masked-operator recovery: the prompt is ``<masked expression>=<target>`` and the
label sequence carries the correct operator at every '?' position and pad everywhere else
(``build_arithmetic_dataset.tokenize_example``). Everything the model is asked for therefore
sits at the '?' positions, and that is what ``decode`` returns: the predicted operators, in
the order the masks appear. Which positions to read comes from the prompt, hence
``Encoded.meta``.

This is the one task whose answer is not a whole task statement, so it is also the one whose
``Decoded.text`` does not round-trip through ``encode`` — an operator string is not an
expression. Re-running it means substituting the operators back into the original prompt.

Label and input are the same length here — one label character per prompt character — so a
prompt's own length is all the room the answer needs.

This builder's pad token is ``'p'``, not ``"<PAD>"``, hence the ``pad_token`` override.
"""

from __future__ import annotations

from typing import Any, Sequence

from .base import Decoded, Encoded, Tokenizer, TokenizerError

MASK = "?"
EQUALS = "="


class ArithmeticTokenizer(Tokenizer):
    name = "arithmetic"
    pad_token = "p"  # build_arithmetic_dataset.PAD_TOKEN
    required_tokens = (MASK, EQUALS, pad_token)
    input_help = (
        "Expression with '?' in place of each operator to recover, followed by '=' and the "
        "target value, e.g. '3?5+2?7=42'."
    )

    def encode(self, text: str, puzzle_identifier: int = 0, **kwargs: Any) -> Encoded:
        expr = "".join(text.split())
        if not expr:
            raise TokenizerError("empty expression")
        allowed = "".join(sorted(c for c in self.vocab_map if c != self.pad_token))
        bad = {c for c in expr if c not in self.vocab_map or c == self.pad_token}
        if bad:
            raise TokenizerError(f"unexpected characters {sorted(bad)}; allowed: {allowed!r}")
        if expr.count(EQUALS) != 1:
            raise TokenizerError(f"expression must contain exactly one {EQUALS!r}")
        if MASK not in expr:
            raise TokenizerError(f"expression must contain at least one {MASK!r} to solve for")

        return Encoded(
            self.as_tokens(self.ids_for(expr)), puzzle_identifier, meta={"prompt": expr}
        )

    def decode(self, ids: Sequence[int], prompt: Encoded) -> Decoded:
        """What the model emitted, pad removed.

        The label is an operator at each '?' and pad everywhere else, so a model doing the
        task returns exactly the recovered operators — ``*-`` for ``3?5+2?7=42``. Anything it
        emits away from a mask is its output too, and is shown rather than filtered out
        against the prompt.
        """
        expr: str = prompt.meta["prompt"]
        answer = "".join(self.token_for(i) for i in self.drop_padding(ids[: len(expr)]))
        return Decoded(text=answer)
