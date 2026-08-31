"""Tests for resolving a checkpoint's location — local directory or Hugging Face repo.

These import ``api.loader``, which needs torch and yaml (but not flash-attn: the model code
is only imported when a model is actually built, which none of these do). Nothing here talks
to the network — Hub *detection* is a pure function, and the download itself is not tested.

    python -m pytest api/tests/test_loader.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("torch")
pytest.importorskip("yaml")

from api.loader import (  # noqa: E402
    CheckpointError,
    _read_seq_len,
    looks_like_hf_repo,
    resolve_checkpoint,
)

CONFIG = "arch:\n  name: hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1\ndata_path: data/sudoku\n"


# -- telling a Hub repo from a local path --------------------------------------

@pytest.mark.parametrize("repo", [
    "sapientinc/HRM-checkpoint-sudoku-extreme",
    "hf://sapientinc/HRM-checkpoint-sudoku-extreme",
    "some-org/some.model-v2",
])
def test_hub_repo_ids_are_recognised(repo):
    assert looks_like_hf_repo(repo)


@pytest.mark.parametrize("path", [
    "/workspace/runs/sudoku",      # absolute path
    "checkpoints/sudoku/run-1",    # more than one slash
    "sudoku",                      # no slash at all
    "../runs/sudoku",              # relative, with a dot segment
])
def test_local_paths_are_not_mistaken_for_hub_repos(path):
    assert not looks_like_hf_repo(path)


def test_an_existing_directory_wins_over_a_repo_id(tmp_path, monkeypatch):
    """A local 'org/name' that exists is a path, not a download."""
    (tmp_path / "org").mkdir()
    (tmp_path / "org" / "name").mkdir()
    monkeypatch.chdir(tmp_path)
    assert not looks_like_hf_repo("org/name")


# -- finding the weight file ---------------------------------------------------

def test_step_files_win_and_the_newest_is_taken(tmp_path):
    for name in ("step_10", "step_200", "step_200_ema", "all_config.yaml"):
        (tmp_path / name).write_text("x")
    assert os.path.basename(resolve_checkpoint(str(tmp_path))[0]) == "step_200_ema"
    assert os.path.basename(resolve_checkpoint(str(tmp_path), prefer_ema=False)[0]) == "step_200"


def test_a_hub_repo_weight_file_is_found_without_step_naming(tmp_path):
    """Hub repos name their weights 'checkpoint'; step_N is this repo's own convention."""
    (tmp_path / "checkpoint").write_text("x")
    (tmp_path / "all_config.yaml").write_text(CONFIG)
    weight_file, directory = resolve_checkpoint(str(tmp_path))
    assert os.path.basename(weight_file) == "checkpoint"
    assert directory == str(tmp_path)


def test_a_directory_with_no_weights_says_what_it_looked_for(tmp_path):
    (tmp_path / "all_config.yaml").write_text(CONFIG)
    with pytest.raises(CheckpointError, match="checkpoint"):
        resolve_checkpoint(str(tmp_path))


# -- seq_len -------------------------------------------------------------------

def test_seq_len_comes_from_the_config():
    assert _read_seq_len({"seq_len": 81, "arch": {}}, {}, "/somewhere") == 81
    assert _read_seq_len({"arch": {}}, {"seq_len": 900}, "/somewhere") == 900


def test_an_override_wins_over_the_config():
    assert _read_seq_len({"seq_len": 81, "arch": {}}, {}, "/somewhere", override=243) == 243


def test_a_missing_seq_len_points_at_the_flag():
    """Hub repos carry no seq_len, so the error has to say how to supply it."""
    with pytest.raises(CheckpointError, match="--seq-len"):
        _read_seq_len({"arch": {}}, {}, "/somewhere")


@pytest.mark.parametrize("bad", ["eighty-one", 0, -5])
def test_an_unusable_seq_len_is_rejected(bad):
    with pytest.raises(CheckpointError):
        _read_seq_len({"seq_len": bad, "arch": {}}, {}, "/somewhere")
