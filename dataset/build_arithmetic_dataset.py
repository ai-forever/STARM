import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

from common import PuzzleDatasetMetadata

VOCAB: List[str] = ['p', '?', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/', '+', '-', '*', '=']
PAD_TOKEN: str = 'p'


class PostProcessConfig(BaseModel):
    input_dir: str = "raw-data/arithmetic"
    output_dir: str = "data/arithmetic"


def tokenize_example(masked_expr: str, mask_ops: str, target: int, token_to_id: Dict[str, int], max_len: int,
                     pad_id: int):
    inp_str = f"{masked_expr}={target}"
    lab_chars = []
    op_idx = 0
    for ch in masked_expr:
        if ch == '?':
            lab_chars.append(mask_ops[op_idx])
            op_idx += 1
        else:
            lab_chars.append('p')
    lab_chars.append('p')  # =
    lab_chars.extend(['p'] * len(str(target)))
    lab_str = ''.join(lab_chars)

    inp = [token_to_id[c] for c in inp_str]
    lab = [token_to_id[c] for c in lab_str]

    inp += [pad_id] * (max_len - len(inp))
    lab += [pad_id] * (max_len - len(lab))

    return np.array(inp, dtype=np.uint8), np.array(lab, dtype=np.uint8)


def key_func(item: Tuple) -> Tuple:
    """Оригинальная функция группировки"""
    masked_expr, mask_ops, target = item
    # Извлекаем цифры из masked_expr
    digits = tuple(int(ch) for ch in masked_expr if ch.isdigit())
    return (digits, target)


def build_grouped_structure(examples: List[Tuple], token_to_id: Dict[str, int], max_len: int, pad_id: int):
    """Максимально близко к оригиналу"""
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    grouped = defaultdict(list)
    for item in examples:
        grouped[key_func(item)].append(item)

    for v in grouped.values():
        for masked_expr, mask_ops, target in v:
            inp, out = tokenize_example(masked_expr, mask_ops, target, token_to_id, max_len, pad_id)

            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        results["group_indices"].append(puzzle_id)

    return results


def save_npy(results: Dict, save_dir: str, prefix: str):
    os.makedirs(save_dir, exist_ok=True)
    for k, v in results.items():
        if isinstance(v, list):
            if k in ["inputs", "labels"]:
                arr = np.concatenate(v).reshape(-1, len(v[0]))
            else:
                arr = np.array(v, dtype=np.int32)
        else:
            arr = v
        np.save(os.path.join(save_dir, f"{prefix}__{k}.npy"), arr)


def save_metadata(save_dir: str, seq_len: int, vocab_size: int, total_groups: int, sets: List[str]):
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
        json.dump(metadata.model_dump(), f, indent=2)

    with open(os.path.join(save_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


cli = ArgParser()


@cli.command(singleton=True)
def postprocess(config: PostProcessConfig):
    token_to_id = {token: idx for idx, token in enumerate(VOCAB)}
    pad_id = token_to_id[PAD_TOKEN]

    print("Loading and processing TRAIN...")
    train_path = os.path.join(config.input_dir, "train.jsonl")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_sample = [json.loads(line) for line in f]

    max_len = max(max(len(x["input"]), len(x["labels"])) for x in train_sample)

    train_examples = [(x["input"].split('=')[0], x["labels"], x["target"])
                      for x in train_sample]  # masked_expr, mask_ops, target

    grouped_results = build_grouped_structure(train_examples, token_to_id, max_len, pad_id)

    train_results = {
        "inputs": np.concatenate(grouped_results["inputs"]).reshape(-1, max_len),
        "labels": np.concatenate(grouped_results["labels"]).reshape(-1, max_len),
        "group_indices": np.array(grouped_results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(grouped_results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(grouped_results["puzzle_identifiers"], dtype=np.int32),
    }

    train_dir = os.path.join(config.output_dir, "train")
    save_npy(train_results, train_dir, "all")
    save_metadata(train_dir, max_len, len(token_to_id),
                  len(train_results["group_indices"]) - 1, ["all"])

    print("\nProcessing TEST...")
    test_path = os.path.join(config.input_dir, "test.jsonl")
    with open(test_path, 'r', encoding='utf-8') as f:
        test_sample = [json.loads(line) for line in f]

    test_intervals = [(0, 101), (102, 201), (202, 301)]
    test_sets = {interval: [] for interval in test_intervals}

    for line in test_sample:
        target = line["target"]
        for interval in test_intervals:
            if interval[0] <= target <= interval[1]:
                test_sets[interval].append((line["input"].split('=')[0], line["labels"], target))
                break

    test_dir = os.path.join(config.output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    for ti in test_intervals:
        print(f"  Converting {ti[0]}-{ti[1]} ({len(test_sets[ti])} examples)")
        subset_results = build_grouped_structure(test_sets[ti], token_to_id, max_len, pad_id)

        results = {
            "inputs": np.concatenate(subset_results["inputs"]).reshape(-1, max_len),
            "labels": np.concatenate(subset_results["labels"]).reshape(-1, max_len),
            "group_indices": np.array(subset_results["group_indices"], dtype=np.int32),
            "puzzle_indices": np.array(subset_results["puzzle_indices"], dtype=np.int32),
            "puzzle_identifiers": np.array(subset_results["puzzle_identifiers"], dtype=np.int32),
        }

        prefix = f"{ti[0]}-{ti[1]}"
        save_npy(results, test_dir, prefix)

    save_metadata(test_dir, max_len, len(token_to_id),
                  len(results["group_indices"]) - 1,
                  [f"{ti[0]}-{ti[1]}" for ti in test_intervals])

    print("Post-processing completed successfully!")


if __name__ == "__main__":
    cli()
