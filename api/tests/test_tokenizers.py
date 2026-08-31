"""Round-trip tests for the per-domain tokenizers.

These need only numpy, so they run without torch or flash-attn:

    python -m pytest api/tests/test_tokenizers.py

Each round trip asserts that ``decode`` inverts ``encode``: if the model returned exactly
the prompt's token ids, the caller must get the prompt's text back. That is what catches
a layout drifting away from its dataset builder.

The vocab maps below are written in the shape each ``dataset/build_*_dataset.py``
produces, since the tokenizers consume such a map rather than defining one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.tokenizers.arc import MAX_GRID_SIZE  # noqa: E402
from api.tokenizers import (  # noqa: E402
    TASKS,
    ArcTokenizer,
    ArithmeticTokenizer,
    GameOfLifeTokenizer,
    MazeTokenizer,
    SudokuTokenizer,
    TokenizerError,
    build_tokenizer,
    builtin_vocab_map,
    find_vocab_map,
    load_vocab_map,
    task_from_data_path,
)

# build_sudoku_dataset stores digit + 1, leaving 0 for pad.
SUDOKU_MAP = {"<PAD>": 0, **{str(d): d + 1 for d in range(10)}}

# The same ids as SUDOKU_MAP, spelled the way the training runs write the file: '.' for the
# empty cell (as in the puzzle string itself) and 'p' for pad.
SUDOKU_MAP_DOTTED = {"p": 0, ".": 1, **{str(d): d + 1 for d in range(1, 10)}}

# build_maze_dataset: char2id is CHARSET index + 1.
MAZE_MAP = {"<PAD>": 0, **{c: i + 1 for i, c in enumerate("# SGo")}}

# The same ids as ARC_MAP, spelled the way some runs write the file: lower-case specials.
ARC_MAP_LOWER = {"<pad>": 0, "<eos>": 1, **{str(d): d + 2 for d in range(10)}}

# build_arc_dataset: PAD 0, EOS 1, colour v -> v + 2.
ARC_MAP = {"<PAD>": 0, "<EOS>": 1, **{str(d): d + 2 for d in range(10)}}

# build_arithmetic_dataset.VOCAB, indexed in order; 'p' is the pad token.
ARITHMETIC_VOCAB = ["p", "?", *(str(d) for d in range(10)), "/", "+", "-", "*", "="]
ARITHMETIC_MAP = {t: i for i, t in enumerate(ARITHMETIC_VOCAB)}

# build_gol_dataset.create_vocab(BIN_CHARS): ['<PAD>'] + chars, then <SEP> last.
GOL_CHARS = ["b", "o", "$"] + [str(d) for d in range(10)]
GOL_MAP = {t: i for i, t in enumerate(["<PAD>"] + GOL_CHARS)}
GOL_MAP["<SEP>"] = len(GOL_MAP)

PUZZLE = (
    "003020600"
    "900305001"
    "001806400"
    "008102900"
    "700000008"
    "006708200"
    "002609500"
    "800203009"
    "005010300"
)


# -- base contract -------------------------------------------------------------

def test_vocab_size_comes_from_the_map():
    assert SudokuTokenizer(SUDOKU_MAP).vocab_size == 11
    assert MazeTokenizer(MAZE_MAP).vocab_size == 6
    assert ArcTokenizer(ARC_MAP).vocab_size == 12
    assert ArithmeticTokenizer(ARITHMETIC_MAP).vocab_size == 17
    assert GameOfLifeTokenizer(GOL_MAP).vocab_size == 15


def test_missing_required_token_is_rejected():
    """A map that cannot serve the task fails at construction, not mid-request."""
    with pytest.raises(TokenizerError, match="<EOS>"):
        ArcTokenizer({k: v for k, v in ARC_MAP.items() if k != "<EOS>"})
    with pytest.raises(TokenizerError, match="<SEP>"):
        GameOfLifeTokenizer({k: v for k, v in GOL_MAP.items() if k != "<SEP>"})
    with pytest.raises(TokenizerError, match=r"\?"):
        ArithmeticTokenizer({k: v for k, v in ARITHMETIC_MAP.items() if k != "?"})


@pytest.mark.parametrize("bad", [{}, {"a": "1"}, []])
def test_malformed_map_is_rejected(bad):
    with pytest.raises(TokenizerError):
        SudokuTokenizer(bad)


def test_pad_id_follows_the_map():
    """Arithmetic's pad token is 'p'; the others use the conventional 0 slot."""
    assert ArithmeticTokenizer(ARITHMETIC_MAP).pad_id == ARITHMETIC_MAP["p"]
    assert SudokuTokenizer(SUDOKU_MAP).pad_id == 0


def test_pad_batch_pads_to_the_longest_prompt():
    """Padding exists only to make a batch rectangular; one prompt gets none."""
    tok = ArithmeticTokenizer(ARITHMETIC_MAP)
    short, long = tok.encode("3?5=8"), tok.encode("3?5+2?7=42")
    assert (len(short), len(long)) == (5, 10)

    solo = tok.pad_batch([short], len(short))
    assert solo.shape == (1, 5)
    assert solo[0].tolist() == short.tokens.tolist()

    batch = tok.pad_batch([short, long], max(len(short), len(long)))
    assert batch.shape == (2, 10)
    assert batch[0].tolist() == short.tokens.tolist() + [tok.pad_id] * 5
    assert batch[1].tolist() == long.tokens.tolist()


# -- sudoku --------------------------------------------------------------------

def test_sudoku_ids_follow_the_map():
    encoded = SudokuTokenizer(SUDOKU_MAP).encode(PUZZLE)
    assert encoded.tokens.dtype == np.int32
    assert encoded.tokens.tolist() == [SUDOKU_MAP[c] for c in PUZZLE]


def test_sudoku_round_trip():
    tok = SudokuTokenizer(SUDOKU_MAP)
    encoded = tok.encode(PUZZLE)
    assert tok.decode(encoded.tokens, encoded).text == PUZZLE


def test_sudoku_accepts_dots_and_whitespace():
    tok = SudokuTokenizer(SUDOKU_MAP)
    grid = "\n".join(PUZZLE[r * 9:(r + 1) * 9].replace("0", ".") for r in range(9))
    assert tok.encode(grid).tokens.tolist() == tok.encode(PUZZLE).tokens.tolist()


@pytest.mark.parametrize("bad", ["123", PUZZLE + "0", PUZZLE[:-1] + "x"])
def test_sudoku_rejects_malformed(bad):
    with pytest.raises(TokenizerError):
        SudokuTokenizer(SUDOKU_MAP).encode(bad)


def test_sudoku_accepts_a_dotted_map():
    """A map spelling the empty cell '.' and pad 'p' names the same ids, and must work."""
    tok = SudokuTokenizer(SUDOKU_MAP_DOTTED)
    assert tok.pad_id == 0
    # Both spellings of the empty cell reach the same id, whichever the map uses.
    assert tok.encode(PUZZLE).tokens.tolist() == \
           tok.encode(PUZZLE.replace("0", ".")).tokens.tolist()
    assert tok.encode(PUZZLE).tokens.tolist() == \
           SudokuTokenizer(SUDOKU_MAP).encode(PUZZLE).tokens.tolist()


def test_sudoku_dotted_map_round_trip():
    tok = SudokuTokenizer(SUDOKU_MAP_DOTTED)
    encoded = tok.encode(PUZZLE)
    # decode answers in the map's own spelling, and that text re-encodes identically.
    text = tok.decode(encoded.tokens, encoded).text
    assert text == PUZZLE.replace("0", ".")
    assert tok.encode(text).tokens.tolist() == encoded.tokens.tolist()


def test_sudoku_rejects_a_map_without_any_blank():
    with pytest.raises(TokenizerError):
        SudokuTokenizer({str(d): d for d in range(1, 10)})


# -- maze ----------------------------------------------------------------------

def _maze(side: int) -> str:
    rows = ["".join("#" if (r + c) % 4 == 0 else " " for c in range(side)) for r in range(side)]
    rows[0] = "S" + rows[0][1:]
    rows[-1] = rows[-1][:-1] + "G"
    return "\n".join(rows)


def test_maze_ids_follow_the_map():
    encoded = MazeTokenizer(MAZE_MAP).encode("#S\nGo")
    assert encoded.tokens.tolist() == [MAZE_MAP[c] for c in "#SGo"]


def test_maze_round_trip_grid():
    tok = MazeTokenizer(MAZE_MAP)
    maze = _maze(6)
    encoded = tok.encode(maze)
    decoded = tok.decode(encoded.tokens, encoded)
    assert decoded.text == maze.replace("\n", "")


def test_maze_accepts_flat_line():
    tok = MazeTokenizer(MAZE_MAP)
    maze = _maze(6)
    assert tok.encode(maze.replace("\n", "")).tokens.tolist() == \
        tok.encode(maze).tokens.tolist()


def test_maze_pads_trimmed_trailing_spaces():
    """An all-open row survives an editor that strips trailing whitespace."""
    tok = MazeTokenizer(MAZE_MAP)
    encoded = tok.encode("S###\n\n\n###G")  # rows 1 and 2 entirely open
    assert encoded.tokens[4:12].tolist() == [MAZE_MAP[" "]] * 8


def test_maze_rejects_char_absent_from_map():
    with pytest.raises(TokenizerError):
        MazeTokenizer(MAZE_MAP).encode("#S\nGx")


def test_maze_rejects_row_wider_than_the_grid():
    """The grid is square, so the row count fixes the width too."""
    with pytest.raises(TokenizerError):
        MazeTokenizer(MAZE_MAP).encode("###S\nGo##")  # 2 rows, but 4 wide


def test_maze_rejects_flat_line_that_is_not_square():
    with pytest.raises(TokenizerError):
        MazeTokenizer(MAZE_MAP).encode("#S G o")


def test_maze_length_follows_the_grid():
    tok = MazeTokenizer(MAZE_MAP)
    assert len(tok.encode(_maze(6))) == 36
    assert len(tok.encode(_maze(9))) == 81


# -- arc -----------------------------------------------------------------------

def test_arc_closes_every_row_with_eos():
    """A sequence is the grid row-major, each row closed by EOS — no canvas, no padding."""
    tok = ArcTokenizer(ARC_MAP)
    assert tok.encode("12\n34").tokens.tolist() == [
        ARC_MAP["1"], ARC_MAP["2"], ARC_MAP["<EOS>"],
        ARC_MAP["3"], ARC_MAP["4"], ARC_MAP["<EOS>"],
    ]


def test_arc_accepts_eos_written_out():
    """The wire format spells row breaks '<eos>'; newlines mean the same thing."""
    tok = ArcTokenizer(ARC_MAP)
    assert tok.encode("444000777<eos>404440707").tokens.tolist() == \
        tok.encode("444000777\n404440707").tokens.tolist()


def test_arc_ignores_a_trailing_run_of_eos():
    """Sequences are filled out with EOS, so a tail of them is padding, not empty rows."""
    tok = ArcTokenizer(ARC_MAP)
    padded = "444000777<eos>404440707<eos><eos><eos><eos>"
    assert tok.encode(padded).tokens.tolist() == tok.encode("444000777<eos>404440707").tokens.tolist()


def test_arc_pads_a_batch_with_eos():
    """The filler is EOS, the way the dataset's own sequences are padded."""
    tok = ArcTokenizer(ARC_MAP)
    batch = tok.pad_batch([tok.encode("12\n34")], 10)
    assert batch[0].tolist()[-4:] == [ARC_MAP["<EOS>"]] * 4
    assert ARC_MAP["<PAD>"] not in batch[0].tolist()


def test_arc_round_trip():
    tok = ArcTokenizer(ARC_MAP)
    encoded = tok.encode("0123\n4567\n8900")
    decoded = tok.decode(encoded.tokens, encoded)
    assert decoded.text == "0123<EOS>4567<EOS>8900<EOS>"
    assert tok.encode(decoded.text).tokens.tolist() == encoded.tokens.tolist()


def test_arc_decode_keeps_every_eos():
    """Row breaks and the tail are the model's own tokens, shown rather than trimmed."""
    tok = ArcTokenizer(ARC_MAP)
    prompt = tok.encode("444000777<eos>404440707<eos>400040777<eos>444440000")
    answer = [ARC_MAP[c] for c in "444"] + [ARC_MAP["<EOS>"]] + \
             [ARC_MAP[c] for c in "400"] + [ARC_MAP["<EOS>"]] + \
             [ARC_MAP[c] for c in "700"] + [ARC_MAP["<EOS>"]] * 4
    assert tok.decode(np.array(answer), prompt).text == \
        "444<EOS>400<EOS>700<EOS><EOS><EOS><EOS>"


def test_arc_rejects_grid_taller_than_the_maximum():
    tok = ArcTokenizer(ARC_MAP)
    with pytest.raises(TokenizerError, match="maximum"):
        tok.encode("\n".join("1" * (MAX_GRID_SIZE + 1) for _ in range(2)))


def test_arc_accepts_separated_cells():
    tok = ArcTokenizer(ARC_MAP)
    assert tok.encode("1 2\n3, 4").tokens.tolist() == tok.encode("12\n34").tokens.tolist()


def test_arc_reports_an_all_pad_prediction():
    tok = ArcTokenizer(ARC_MAP)
    prompt = tok.encode("12\n34")
    assert tok.decode(np.full(6, ARC_MAP["<PAD>"]), prompt).text == ""


@pytest.mark.parametrize("bad", ["", "12\n345", "1a\n34", "12 34\n56 78"])
def test_arc_rejects_malformed(bad):
    with pytest.raises(TokenizerError):
        ArcTokenizer(ARC_MAP).encode(bad)


# -- arithmetic ----------------------------------------------------------------

def test_arithmetic_ids_follow_the_map():
    tok = ArithmeticTokenizer(ARITHMETIC_MAP)
    encoded = tok.encode("3?5=8")
    # Natural length: a single prompt is not padded to anything.
    assert encoded.tokens.tolist() == [ARITHMETIC_MAP[c] for c in "3?5=8"]


def test_arithmetic_returns_only_the_predicted_operators():
    """Labels carry operators at '?' positions and pad elsewhere; only those are the answer."""
    tok = ArithmeticTokenizer(ARITHMETIC_MAP)
    encoded = tok.encode("3?5+2?7=42")
    labels = ["p"] * 10
    labels[1] = "*"  # 3 ? 5 + 2 ? 7 = 4 2
    labels[5] = "-"  # 0 1 2 3 4 5 6 7 8 9
    ids = [ARITHMETIC_MAP[c] for c in labels] + [ARITHMETIC_MAP["p"]] * 6

    decoded = tok.decode(np.array(ids), encoded)
    assert decoded.text == "*-"


def test_arithmetic_shows_predictions_away_from_the_masks():
    """Output is the model's tokens, not a filter over the prompt: junk off-mask is shown."""
    tok = ArithmeticTokenizer(ARITHMETIC_MAP)
    encoded = tok.encode("3?5+2?7=42")
    # Junk everywhere the prompt is fixed, real operators at the two '?' positions (1 and 5).
    predicted = ["9"] * 10
    predicted[1], predicted[5] = "+", "*"
    ids = [ARITHMETIC_MAP[c] for c in predicted]

    assert tok.decode(np.array(ids), encoded).text == "9+999*9999"


def test_arithmetic_all_pad_decodes_to_nothing():
    """Pad is the one token dropped, so a model that answered only pad answered nothing."""
    tok = ArithmeticTokenizer(ARITHMETIC_MAP)
    encoded = tok.encode("3?5=8")
    assert tok.decode(np.full(8, ARITHMETIC_MAP["p"]), encoded).text == ""


@pytest.mark.parametrize("bad", ["", "3+5=8", "3?5", "3?5=8=9", "3?5=(8)"])
def test_arithmetic_rejects_malformed(bad):
    with pytest.raises(TokenizerError):
        ArithmeticTokenizer(ARITHMETIC_MAP).encode(bad)


def test_arc_accepts_lower_case_specials():
    """A map writing '<eos>'/'<pad>' serves a task asking for '<EOS>': same tokens, same ids."""
    tok = ArcTokenizer(ARC_MAP_LOWER)
    assert tok.pad_token == "<pad>" and tok.pad_id == 0
    grid = "077\n770\n077"
    encoded = tok.encode(grid)
    assert encoded.tokens.tolist() == ArcTokenizer(ARC_MAP).encode(grid).tokens.tolist()
    # decode answers in the map's own spelling.
    assert tok.decode(encoded.tokens, encoded).text == "077<eos>770<eos>077<eos>"


def test_find_token_prefers_an_exact_match():
    """Case folding is the fallback, never an override of a token the map spells exactly."""
    tok = ArcTokenizer({**ARC_MAP, "<eos>": 99})
    assert tok.find_token("<EOS>") == "<EOS>"
    assert tok.token_id("<EOS>") == ARC_MAP["<EOS>"]


# -- game of life --------------------------------------------------------------

def test_gol_ids_follow_the_map():
    """The input is pattern + <SEP> + the step's digits."""
    tok = GameOfLifeTokenizer(GOL_MAP)
    encoded = tok.encode("bo$ob|12")
    # Natural length: pattern + <SEP> + one token per step digit, and no padding.
    assert encoded.tokens.tolist() == [
        GOL_MAP["b"], GOL_MAP["o"], GOL_MAP["$"], GOL_MAP["o"], GOL_MAP["b"],
        GOL_MAP["<SEP>"], GOL_MAP["1"], GOL_MAP["2"],
    ]
    assert encoded.meta == {"pattern": "bo$ob", "step": 12}


def test_gol_step_as_separate_field():
    tok = GameOfLifeTokenizer(GOL_MAP)
    assert tok.encode("bo$ob", step=12).tokens.tolist() == \
        tok.encode("bo$ob|12").tokens.tolist()


def test_gol_round_trip_pattern():
    tok = GameOfLifeTokenizer(GOL_MAP)
    pattern = "bob$obo$bbo"
    label = [GOL_MAP[c] for c in pattern] + [GOL_MAP["<PAD>"]] * 5
    prompt = tok.encode(f"{pattern}|3")
    decoded = tok.decode(np.array(label), prompt)
    assert decoded.text == pattern


@pytest.mark.parametrize("bad", ["bo$ob", "bo$ob|x", "bo$ob|", "xx|1", "|1"])
def test_gol_rejects_malformed(bad):
    with pytest.raises(TokenizerError):
        GameOfLifeTokenizer(GOL_MAP).encode(bad)


def test_gol_rejects_step_given_twice():
    with pytest.raises(TokenizerError):
        GameOfLifeTokenizer(GOL_MAP).encode("bo$ob|3", step=4)


# -- registry ------------------------------------------------------------------

@pytest.mark.parametrize(
    "data_path,expected",
    [
        ("dataset/data/sudoku-extreme-full", "sudoku"),
        ("dataset/data/maze-30x30-hard-1k", "maze"),
        ("dataset/data/arc-aug-1000", "arc"),
        ("dataset/data/arc-2-aug-1000", "arc"),
        ("dataset/data/arithmetic", "arithmetic"),
        ("dataset/data/game_of_life", "game_of_life"),
        ("/home/x/data/game_of_life/seed2675-p0_2-len17-step50/leq250-bin", "game_of_life"),
        ("data/gol", "game_of_life"),
        ("something/else", None),
    ],
)
def test_task_from_data_path(data_path, expected):
    assert task_from_data_path(data_path) == expected


def test_load_and_find_vocab_map(tmp_path):
    (tmp_path / "vocab_map.json").write_text(json.dumps(GOL_MAP), encoding="utf-8")
    assert find_vocab_map(str(tmp_path)) == str(tmp_path / "vocab_map.json")
    assert load_vocab_map(str(tmp_path / "vocab_map.json")) == GOL_MAP


def test_find_vocab_map_prefers_first_directory(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (second / "vocab_map.json").write_text("{}", encoding="utf-8")
    assert find_vocab_map(str(first), str(second)) == str(second / "vocab_map.json")
    assert find_vocab_map(None, str(first)) is None


def test_build_tokenizer_from_data_path(tmp_path):
    path = tmp_path / "vocab_map.json"
    path.write_text(json.dumps(SUDOKU_MAP), encoding="utf-8")
    tokenizer = build_tokenizer(
        data_path="dataset/data/sudoku-extreme-full",
        vocab_map_file=str(path),
        vocab_size=11,
    )
    assert isinstance(tokenizer, SudokuTokenizer)


def test_build_tokenizer_rejects_vocab_size_mismatch(tmp_path):
    path = tmp_path / "vocab_map.json"
    path.write_text(json.dumps(SUDOKU_MAP), encoding="utf-8")
    with pytest.raises(TokenizerError, match="embedding matrix"):
        build_tokenizer(task="sudoku", vocab_map_file=str(path), vocab_size=12)


def test_build_tokenizer_falls_back_to_the_shipped_map():
    """Without a map of its own, a task is served by the one this repo ships."""
    assert build_tokenizer(task="sudoku").vocab_size == len(SUDOKU_MAP)


def test_build_tokenizer_reports_a_task_it_ships_nothing_for(monkeypatch):
    monkeypatch.setattr("api.tokenizers.BUILTIN_VOCAB_MAPS", "/nonexistent")
    with pytest.raises(TokenizerError, match="vocab_map.json"):
        build_tokenizer(task="sudoku")


def test_build_tokenizer_needs_a_task_hint():
    with pytest.raises(TokenizerError):
        build_tokenizer()
    with pytest.raises(TokenizerError):
        build_tokenizer(data_path="something/else")


def test_build_tokenizer_rejects_unknown_task(tmp_path):
    path = tmp_path / "vocab_map.json"
    path.write_text(json.dumps(SUDOKU_MAP), encoding="utf-8")
    with pytest.raises(TokenizerError, match="unknown task"):
        build_tokenizer(task="chess", vocab_map_file=str(path))


# -- the vocabularies this repo ships -------------------------------------------

# The ids are the builders' constants; how a run spells pad or a blank cell is its own
# business, so the shipped files are checked on ids rather than on spelling.
@pytest.mark.parametrize("task, expected_ids", [
    ("sudoku", {str(d): d + 1 for d in range(1, 10)}),
    ("maze", {c: i + 1 for i, c in enumerate("# SGo")}),
    ("arc", {str(d): d + 2 for d in range(10)}),
    ("arithmetic", {t: i for i, t in enumerate(
        ["p", "?", *(str(d) for d in range(10)), "/", "+", "-", "*", "="])}),
    ("game_of_life", {"b": 1, "o": 2, "$": 3, **{str(d): d + 4 for d in range(10)}}),
])
def test_shipped_vocab_maps_carry_their_builders_ids(task, expected_ids):
    vocab = load_vocab_map(builtin_vocab_map(task))
    assert {k: vocab[k] for k in expected_ids} == expected_ids


@pytest.mark.parametrize("task, vocab_size", [
    ("sudoku", 11), ("maze", 6), ("arc", 12), ("arithmetic", 17), ("game_of_life", 15),
])
def test_shipped_vocab_maps_are_the_size_the_builder_writes(task, vocab_size):
    """The size is what a checkpoint's embedding matrix is checked against at startup."""
    assert len(load_vocab_map(builtin_vocab_map(task))) == vocab_size


def test_shipped_maps_pad_on_zero():
    """Every builder records pad_id 0, whether the token is spelled '<PAD>', '<pad>' or 'p'."""
    for task in TASKS:
        assert build_tokenizer(task=task, vocab_map_file=builtin_vocab_map(task)).pad_id == 0


def test_every_shipped_map_builds_its_tokenizer():
    for task in TASKS:
        path = builtin_vocab_map(task)
        assert path is not None, f"no vocabulary shipped for {task}"
        assert build_tokenizer(task=task, vocab_map_file=path).name == task


def test_a_shipped_map_is_only_the_last_resort(tmp_path):
    """An explicit path wins; the built-in is what a checkpoint without any map falls back to."""
    own = tmp_path / "vocab_map.json"
    own.write_text(json.dumps({"p": 0, ".": 1, **{str(d): d + 1 for d in range(1, 10)}}))
    tok = build_tokenizer(task="sudoku", vocab_map_file=str(own))
    assert tok.blank == "." and tok.pad_token == "p"
