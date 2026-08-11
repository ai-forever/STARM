import json
import os

import numpy as np
import pandas as pd
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata

tqdm.pandas()

RLE_CHARS = ['b', 'o', '$', '!'] + [str(i) for i in range(10)]
BIN_CHARS = ['b', 'o', '$'] + [str(i) for i in range(10)]


class PostProcessConfig(BaseModel):
    input_dir: str = "raw-data/game_of_life"  # where the jsonl files are stored
    format_name: str = "bin"  # "bin" or "rle"
    output_dir: str = "data/game_of_life"


def create_vocab(vocab_chars: list) -> dict:
    tokens = ['<PAD>'] + vocab_chars

    char_to_id = {char: idx for idx, char in enumerate(tokens)}
    char_to_id['<SEP>'] = len(tokens)

    return char_to_id


def tokenize_text(text, vocab_map):
    return [vocab_map[ch] for ch in str(text)]


def tokenize_row(row, vocab_map):
    return tokenize_text(row['pattern'], vocab_map) + [vocab_map['<SEP>']] + tokenize_text(row['step'], vocab_map)


def get_vocab_map(file_dir: str, format_name: str) -> dict:
    file_path = f'{file_dir}/vocab_map.json'

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab_map = json.load(f)

    except FileNotFoundError as e:
        if format_name == 'bin':
            vocab_map = create_vocab(BIN_CHARS)
        elif format_name == 'rle':
            vocab_map = create_vocab(RLE_CHARS)
        else:
            raise ValueError(format_name)

        os.makedirs(file_dir, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_map, f, ensure_ascii=False)

    return vocab_map


def add_padding(ids, max_len):
    return ids + [0] * (max_len - len(ids)) if len(ids) < max_len else ids[:max_len]


def save_groupped_train(inputs, labels, padding_max_len):
    def _seq_to_numpy(seq, max_len):
        arr = np.concatenate(seq, dtype=np.int32).reshape(-1, max_len)
        return arr

    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for inp, out in zip(tqdm(inputs), labels):
        inp, out = np.array(inp, dtype=np.int32), np.array(out, dtype=np.int32)

        # Push puzzle (only single example)
        results["inputs"].append(inp)
        results["labels"].append(out)
        example_id += 1
        puzzle_id += 1

        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)

        # Push group
        results["group_indices"].append(puzzle_id)

    return {
        "inputs": _seq_to_numpy(results["inputs"], padding_max_len),
        "labels": _seq_to_numpy(results["labels"], padding_max_len),

        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }


def save_metadata(save_dir, seq_len, vocab_size, total_groups, sets):
    os.makedirs(save_dir, exist_ok=True)

    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=vocab_size,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=total_groups,
        mean_puzzle_examples=1,
        sets=sets
    )

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    with open(os.path.join(save_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


def prepare_test_set(path, vocab_map, padding_max_len):
    print(path)
    df = pd.read_json(path, lines=True)

    df['ids_input'] = df.apply(tokenize_row, vocab_map=vocab_map, axis=1)
    df['ids_output'] = df['pattern_after_step'].apply(tokenize_text, vocab_map=vocab_map)

    df['length'] = df.apply(lambda x: max(len(x['ids_input']), len(x['ids_output'])), axis=1)
    df = df[df['length'] <= padding_max_len]

    tqdm.pandas(desc='Add padding')
    df['ids_input'] = df['ids_input'].progress_apply(lambda x: add_padding(x, padding_max_len))
    df['ids_output'] = df['ids_output'].progress_apply(lambda x: add_padding(x, padding_max_len))

    return df.sample(10_000)


cli = ArgParser()


@cli.command(singleton=True)
def create_dataset(config: PostProcessConfig):
    assert config.format_name in ("bin", "rle"), "format_name must be 'bin' or 'rle'"

    # Create or load vocabulary
    vocab_map = get_vocab_map(config.input_dir, config.format_name)

    print("TRAIN part...")
    train_path = os.path.join(config.input_dir, "train.jsonl")
    df = pd.read_json(train_path, lines=True)

    df['ids_input'] = df.apply(tokenize_row, vocab_map=vocab_map, axis=1)
    df['ids_output'] = df['pattern_after_step'].apply(tokenize_text, vocab_map=vocab_map)

    df['length'] = df.progress_apply(lambda x: max(len(x['ids_input']), len(x['ids_output'])), axis=1)

    padding_max_len = df.length.max()

    tqdm.pandas(desc='Add padding')
    df['ids_input'] = df['ids_input'].progress_apply(lambda x: add_padding(x, padding_max_len))
    df['ids_output'] = df['ids_output'].progress_apply(lambda x: add_padding(x, padding_max_len))

    results = save_groupped_train(df['ids_input'], df['ids_output'], padding_max_len)

    train_dir = os.path.join(config.output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    for k, v in results.items():
        np.save(os.path.join(train_dir, f"all__{k}.npy"), v)

    save_metadata(train_dir, seq_len=padding_max_len, vocab_size=len(vocab_map),
                  total_groups=len(results["group_indices"]) - 1, sets=["all"])

    print("TEST part...")
    test_sets = ['in_train_in_range', 'out_train'] + [f'in_train_out_range-{step}' for step in [1, 2, 3, 10]]

    test_dir = os.path.join(config.output_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)

    for test_set in test_sets:
        test_set_path = os.path.join(config.input_dir, "test", f"{test_set}.jsonl")

        if os.path.exists(test_set_path):
            df_test = prepare_test_set(test_set_path, vocab_map, padding_max_len)

            results = save_groupped_train(df_test['ids_input'], df_test['ids_output'], padding_max_len)

            for k, v in results.items():
                np.save(os.path.join(test_dir, f"{test_set}__{k}.npy"), v)

    save_metadata(test_dir, seq_len=padding_max_len, vocab_size=len(vocab_map),
                  total_groups=len(results["group_indices"]) - 1, sets=test_sets)

    print("Conversion to npy completed.")


if __name__ == "__main__":
    cli()
