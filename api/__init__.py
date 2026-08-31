"""HTTP serving layer for trained STARM checkpoints.

The package is deliberately independent of ``pretrain.py``: importing that module
initialises ClearML, flips global torch flags and pulls in the optimizer stack, none
of which an inference server needs. Only ``models.*`` and ``utils.functions`` are used.
"""

__all__ = ["engine", "loader", "tokenizers"]
