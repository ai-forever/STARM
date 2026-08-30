"""Inference engine: text in, text out.

**Adaptive compute time.** The ACT head decides per example, every step, whether the answer
is ready — the head that was trained to do exactly that. A finished example's answer is taken
from the step it finished on, its row leaves the batch, and the remaining rows carry on. So
the recursion depth is the example's own, and ``steps`` in the response reports it;
``halt_max_steps`` is only the ceiling.

The verdict is applied in ``_act_head_says_stop`` rather than left to the model, because every
arch computes it and then gates it behind ``self.training``. ``carry.halted`` is still honoured
alongside it — that is where the ceiling and dense's ``act_inference`` branch come from.

**The checkpoint's length is the length.** Every prompt is padded to ``seq_len`` from
all_config.yaml — the length the run was trained at — and the model is never touched: it is
built for that length once in ``loader.py`` and only ever asked to run a forward pass here.
Training padded every example to the same constant and ``flash_attn_func`` runs without a
key-padding mask (``models/layers.py:130`` and ``:175``), so those pad positions are part of
the input the weights were fitted to. Serving at some other length is a different computation.

Padding is a property of the input tensor, not of the answer: the tokenizers trim the pad tail
back off when they decode, so a caller sees only its own task's text.

**A prompt's answer is its own.** Alone or in a batch of a hundred, a prompt gets the same
text and the same ``steps``: every row is padded to the same ``seq_len``, so no prompt is
reshaped by a neighbour, and halting is per row, so no example is dragged along by a slower
one.

Requests are serialised on a lock: one model, one batch at a time.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .loader import LoadedModel
from .tokenizers import Decoded, Encoded, Tokenizer, TokenizerError

logger = logging.getLogger(__name__)


@dataclass
class Prompt:
    """One request, before tokenization."""

    input: str
    #: ARC only: the task's id, as ARC-AGI names it — ``"007bbfb7"``. Resolved against
    #: ``identifiers.json`` to the index of its learned embedding.
    puzzle_id: Optional[str] = None


@dataclass
class Answer:
    output: str
    #: Recursion steps this sample took before its halting head said stop.
    steps: int
    #: ``halt_max_steps`` from the checkpoint's config — the ceiling on ``steps``.
    max_steps: int


class StarmEngine:
    def __init__(self, loaded: LoadedModel, tokenizer: Tokenizer) -> None:
        self.loaded = loaded
        self.tokenizer = tokenizer
        self._lock = threading.Lock()
        self._puzzle_ids = _read_identifiers(loaded.checkpoint_dir)
        # A one-row table is what a run has when its dataset wrote a single <blank>
        # identifier, and also when it set puzzle_emb_ndim: 0 and has no table at all.
        # Either way there is no per-task embedding to select, ARC or not.
        self._has_puzzle_embeddings = loaded.num_puzzle_identifiers > 1
        if tokenizer.name == "arc" and self._has_puzzle_embeddings and not self._puzzle_ids:
            # Nothing could be answered: every ARC request names a task id, and this file is
            # the only record of which ids the run's puzzle embeddings belong to.
            raise TokenizerError(
                f"this is an ARC checkpoint but {loaded.checkpoint_dir}/identifiers.json is "
                "missing or empty. It is the list of task ids the run was trained on, written "
                "by the dataset build; copy it next to the checkpoint"
            )

    # -- introspection ---------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        loaded = self.loaded
        return {
            "task": self.tokenizer.name,
            "arch": loaded.arch_name,
            "arch_source": loaded.arch_source,
            "checkpoint": loaded.checkpoint_file,
            "step": loaded.step,
            "ema": loaded.is_ema,
            "device": str(loaded.device),
            "vocab_size": loaded.vocab_size,
            "num_puzzle_identifiers": loaded.num_puzzle_identifiers,
            "halt_max_steps": loaded.halt_max_steps,
            "pos_encodings": loaded.pos_encodings,
            "sequence_length": loaded.seq_len,
            "num_parameters": loaded.num_parameters(),
            "input_format": self.tokenizer.input_help,
            "training_data_path": loaded.data_path,
        }

    # -- generation ------------------------------------------------------------

    def generate(
        self, prompts: Union[Prompt, Sequence[Prompt]]
    ) -> Union[Answer, List[Answer]]:
        """Tokenize, run the ACT loop, detokenize — one entry point.

        Takes a single ``Prompt`` or a sequence of them and mirrors that shape in the
        result: one prompt in, one ``Answer`` out; a sequence in, a list out. Batching is
        an implementation detail and never an answer's business: a prompt gets the same
        output and the same ``steps`` whether it arrives alone or with a hundred others.
        """
        single = isinstance(prompts, Prompt)
        items: List[Prompt] = [prompts] if single else list(prompts)
        if not items:
            return []

        # Validate everything first: one bad prompt fails the request before the model runs.
        encoded = []
        for position, prompt in enumerate(items):
            try:
                encoded.append(self._encode(prompt))
            except TokenizerError as exc:
                if single:
                    raise
                raise TokenizerError(f"prompt {position}: {exc}") from exc

        with self._lock:
            predictions, steps = self._run(encoded)

        answers = []
        for i, prompt in enumerate(encoded):
            decoded: Decoded = self.tokenizer.decode(predictions[i], prompt)
            answers.append(
                Answer(
                    output=decoded.text,
                    steps=int(steps[i]),
                    max_steps=self.loaded.halt_max_steps,
                )
            )
        return answers[0] if single else answers

    # -- internals -------------------------------------------------------------

    def _encode(self, prompt: Prompt) -> Encoded:
        encoded = self.tokenizer.encode(
            prompt.input, puzzle_identifier=self._puzzle_identifier(prompt)
        )
        seq_len = self.loaded.seq_len
        if len(encoded) > seq_len:
            raise TokenizerError(
                f"prompt is {len(encoded)} tokens, longer than this checkpoint's seq_len "
                f"({seq_len}). That length is baked into the weights — the rotary table and "
                "the carry are built for it — so a longer prompt cannot be run"
            )
        return encoded

    def _puzzle_identifier(self, prompt: Prompt) -> int:
        """Resolve an ARC task id to the index of its learned embedding.

        That embedding is everything an ARC checkpoint knows about a task, so a request has
        to name one, and the name has to be a task the run was trained on: ``identifiers.json``
        is the list of those, and anything outside it has no embedding to use.

        A checkpoint with no per-task embeddings — every non-ARC dataset, and an ARC run
        trained with ``puzzle_emb_ndim: 0`` — has nothing to select, and answers on 0.
        """
        given = (prompt.puzzle_id or "").strip()

        if not self._has_puzzle_embeddings:
            if given:
                raise TokenizerError(
                    f"puzzle_id {given!r} selects nothing: this checkpoint has no per-task "
                    "puzzle embeddings, so there are no task ids to name"
                )
            return 0

        if self.tokenizer.name != "arc":
            return 0

        if not given:
            raise TokenizerError(
                "this ARC checkpoint needs a puzzle_id — the task's id, as ARC-AGI names it "
                "(e.g. '007bbfb7'). The learned puzzle embedding it selects is where the "
                "checkpoint keeps what it knows about that task"
            )
        if not self._puzzle_ids:
            raise TokenizerError(
                "puzzle_id cannot be resolved: no identifiers.json in the checkpoint "
                f"directory ({self.loaded.checkpoint_dir}). It is the list of task ids this "
                "run was trained on, written by the dataset build"
            )

        index = self._puzzle_ids.get(given)
        if index is None:
            raise TokenizerError(
                f"unknown puzzle_id {given!r}: not one of the {len(self._puzzle_ids)} task "
                "ids this checkpoint was trained on. A task the run never saw has no learned "
                "embedding, so it cannot be answered"
            )
        if not 0 < index < self.loaded.num_puzzle_identifiers:
            # identifiers.json disagreeing with the embedding matrix means the file belongs
            # to a different run; 0 is <blank>, which carries no task.
            raise TokenizerError(
                f"puzzle_id {given!r} maps to index {index}, which this checkpoint's "
                f"embedding table ({self.loaded.num_puzzle_identifiers} rows) cannot use; "
                "identifiers.json looks like it came from another run"
            )
        return index

    def _run(self, encoded: Sequence[Encoded]) -> Tuple[List[np.ndarray], np.ndarray]:
        """One ACT loop over the whole request, shrinking as prompts finish.

        Prompts are padded to the checkpoint's own ``seq_len``, which is what training fed
        the weights; the model itself is used exactly as ``loader.py`` built it.
        """
        model = self.loaded.model
        device = self.loaded.device

        inputs = torch.from_numpy(
            self.tokenizer.pad_batch(encoded, self.loaded.seq_len)
        ).to(device=device, dtype=torch.int32)
        puzzle_ids = torch.tensor(
            [e.puzzle_identifier for e in encoded], device=device, dtype=torch.int32
        )
        # The inner model only reads inputs and puzzle_identifiers; labels exist purely for
        # the loss head, which is not part of the serving path.
        batch = {"inputs": inputs, "puzzle_identifiers": puzzle_ids}

        answers: List[Optional[np.ndarray]] = [None] * len(encoded)
        steps = np.zeros((len(encoded),), dtype=np.int64)
        # Where each row of the current (shrinking) batch belongs in the request.
        active = np.arange(len(encoded))

        with torch.inference_mode():
            # empty_carry/initial_carry allocate with bare torch.empty, so they need the
            # ambient device set, the same way pretrain.py:evaluate does it.
            with torch.device(device):
                carry = model.initial_carry(batch)

            step = 0
            while True:
                carry, preds = model(carry=carry, batch=batch)
                step += 1

                predicted = preds["logits"].argmax(dim=2).cpu().numpy()
                finished = self._act_head_says_stop(preds) | carry.halted.cpu().numpy()
                for row in np.flatnonzero(finished):
                    answers[active[row]] = predicted[row]
                    steps[active[row]] = step

                keep = np.flatnonzero(~finished)
                if not len(keep):
                    break

                # Drop the finished rows. Left in, they would be reset to zero by the next
                # `reset_carry` and re-solve a prompt that is already answered.
                index = torch.from_numpy(keep).to(device=device)
                carry = _select_rows(carry, index)
                batch = {k: v[index] for k, v in batch.items()}
                active = active[keep]

        return answers, steps  # type: ignore[return-value]

    def _act_head_says_stop(self, preds: Dict[str, torch.Tensor]) -> np.ndarray:
        """The ACT head's verdict for each row: has this example thought long enough?

        Every arch computes exactly this and then gates it behind ``self.training``
        (``hrm_act_v2DG.py:780``), so a served model would otherwise halt on step count
        alone. The verdict is taken from the ``q_*`` logits the forward pass already
        returns, using the rule the checkpoint was trained with: ``q_halt > 0`` for
        ``no_ACT_continue`` (TRM, URM), whose head has no continue logit to compare
        against, and ``q_halt > q_continue`` otherwise.

        It is OR-ed with ``carry.halted``, which contributes the ``halt_max_steps`` ceiling
        every arch enforces — and hence the loop's termination — plus dense's
        ``act_inference`` branch, where the model does decide for itself.
        """
        q_halt = preds.get("q_halt_logits")
        if q_halt is None:  # an arch with no halting head; carry.halted decides alone
            return np.zeros((preds["logits"].shape[0],), dtype=bool)

        q_halt = q_halt.to(torch.float32)
        if getattr(self.loaded.model.config, "no_ACT_continue", False):
            return (q_halt > 0).cpu().numpy()
        return (q_halt > preds["q_continue_logits"].to(torch.float32)).cpu().numpy()


def _read_identifiers(checkpoint_dir: str) -> Dict[str, int]:
    """Read ``identifiers.json`` from the checkpoint directory as task id -> index.

    ``build_arc_dataset.convert_dataset`` writes it as a list whose position is the integer
    the puzzle embedding is indexed by and whose value is the task id — ``"007bbfb7"`` for a
    puzzle, ``"007bbfb7_t3_012345678"`` for one of its augmented variants. Only ARC has more
    than ``["<blank>"]`` in it, so a missing file is unremarkable for every other task.
    """
    path = os.path.join(checkpoint_dir, "identifiers.json")
    if not os.path.isfile(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list):
        logger.warning("%s is not a list of task ids, ignoring it", path)
        return {}

    # Later duplicates lose: the first occurrence is the index the builder assigned.
    ids: Dict[str, int] = {}
    for index, name in enumerate(names):
        if isinstance(name, str) and name != "<blank>":
            ids.setdefault(name, index)
    logger.info("Loaded %d task ids from %s", len(ids), path)
    return ids


def _select_rows(value: Any, index: torch.Tensor) -> Any:
    """Keep ``index`` along the batch dimension, anywhere inside a carry.

    The carries are dataclasses of per-row tensors — ``z_H``/``z_L``, ``steps``, ``halted``,
    the ``current_data`` dict — plus dropout-mask fields that are ``None`` under ``eval()``
    (``_maybe_generate_mask`` takes ``self.training`` as its ``enabled`` flag). Walking the
    structure rather than naming fields keeps this working for every arch in the repo.
    """
    if torch.is_tensor(value):
        return value[index]
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{f.name: _select_rows(getattr(value, f.name), index) for f in fields(value)}
        )
    if isinstance(value, dict):
        return {k: _select_rows(v, index) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_select_rows(v, index) for v in value)
    return value
