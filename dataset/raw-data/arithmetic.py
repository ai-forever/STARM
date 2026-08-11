import multiprocessing as mp
import os
import random
from collections import defaultdict
from functools import partial
from typing import List, Dict, Tuple, Optional

import numpy as np
from argdantic import ArgParser
from numba import njit
from pydantic import BaseModel
from tqdm import tqdm


class Config(BaseModel):
    seed: int = 42
    num_digit_samples: int = 20
    min_digits: int = 3
    max_digits: int = 8
    digits_range: Tuple[int, int] = (1, 9)
    train_intervals: List[Tuple[int, int]] = [(0, 101)]
    test_intervals: List[Tuple[int, int]] = [(0, 101), (102, 201), (202, 301)]
    train_ratio: float = 0.8
    n_per_bench: int = 2000
    min_masks: int = 0
    output_dir: str = "arithmetic"
    num_processes: int = 10


def apply_op(a: int, b: int, op: str) -> Optional[int]:
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        if b == 0 or a % b != 0:
            return None
        return a // b
    return None


def evaluate_rpn(tokens: str) -> Optional[int]:
    stack = []
    for token in tokens:
        if token.isdigit():
            stack.append(int(token))
        else:
            if len(stack) < 2:
                return None
            b = stack.pop()
            a = stack.pop()
            res = apply_op(a, b, token)
            if res is None:
                return None
            stack.append(res)
    return stack[0] if len(stack) == 1 else None


def dp_solver_full(digits: List[int]) -> Dict[str, Dict[int, List[str]]]:
    """
    Dynamic programming: all expressions and their values for the given digit sequence.
    Returns: {RPN_skeleton (with '#') : {value : [list_of_RPN_expressions]}}
    """
    n = len(digits)
    dp = [[defaultdict(lambda: defaultdict(list)) for _ in range(n)] for _ in range(n)]

    # Single digits
    for i in range(n):
        skeleton = str(digits[i])
        dp[i][i][skeleton][digits[i]].append(str(digits[i]))

    # Increasing subsequences
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                for left_skeleton, left_dict in dp[i][k].items():
                    for right_skeleton, right_dict in dp[k + 1][j].items():
                        new_skeleton = left_skeleton + right_skeleton + '#'
                        for v1, left_exprs in left_dict.items():
                            for v2, right_exprs in right_dict.items():
                                for op in ('+', '-', '*', '/'):
                                    res = apply_op(v1, v2, op)
                                    if res is None:
                                        continue
                                    for expr1 in left_exprs:
                                        for expr2 in right_exprs:
                                            new_expr = expr1 + expr2 + op
                                            dp[i][j][new_skeleton][res].append(new_expr)

    # Remove duplicates
    result = {}
    for skeleton, value_dict in dp[0][n - 1].items():
        result[skeleton] = {
            value: sorted(set(exprs))
            for value, exprs in value_dict.items()
        }
    return result


CHAR_TO_IDX = {'*': 0, '/': 1, '+': 2, '-': 3}
IDX_TO_CHAR = {v: k for k, v in CHAR_TO_IDX.items()}


@njit
def popcount(x: int) -> int:
    x = (x & 0x55555555) + ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x & 0x0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F)
    x = (x & 0x00FF00FF) + ((x >> 8) & 0x00FF00FF)
    x = (x & 0x0000FFFF) + ((x >> 16) & 0x0000FFFF)
    return x


@njit
def find_best_mask(strings_encoded, n_strings, str_len):
    """Return (best_string_index, fixed_positions_mask, masked_count)."""
    if n_strings == 1:
        return 0, 0, str_len

    # Difference matrix
    diff = np.zeros((n_strings, n_strings), dtype=np.int32)
    for i in range(n_strings):
        for j in range(n_strings):
            if i == j:
                continue
            mask = 0
            for pos in range(str_len):
                if strings_encoded[i, pos] != strings_encoded[j, pos]:
                    mask |= (1 << pos)
            diff[i, j] = mask

    full_mask = (1 << str_len) - 1
    best_string_idx = -1
    best_fixed_mask = 0
    max_masked_count = -1

    for i in range(n_strings):
        min_fixed_size = str_len + 1
        best_mask_for_i = full_mask
        for fixed_mask in range(1, full_mask + 1):
            fixed_size = popcount(fixed_mask)
            if fixed_size >= min_fixed_size:
                continue
            valid = True
            for j in range(n_strings):
                if i == j:
                    continue
                if (fixed_mask & diff[i, j]) == 0:
                    valid = False
                    break
            if valid and fixed_size < min_fixed_size:
                min_fixed_size = fixed_size
                best_mask_for_i = fixed_mask
                if min_fixed_size == 1:
                    break
        masked_count = str_len - min_fixed_size
        if masked_count > max_masked_count:
            max_masked_count = masked_count
            best_string_idx = i
            best_fixed_mask = best_mask_for_i

    return best_string_idx, best_fixed_mask, max_masked_count


def solve(operator_strings: List[str]) -> Tuple[str, str, int, int]:
    """
    For a list of operator strings, find one string and a pattern with '?'
    that maximizes the number of masks while preserving uniqueness.
    Returns: (original_string, masked_pattern, masked_count, string_index).
    """
    if not operator_strings:
        return "", "", 0, -1

    n_strings = len(operator_strings)
    str_len = len(operator_strings[0])

    strings_encoded = np.zeros((n_strings, str_len), dtype=np.int8)
    for i, s in enumerate(operator_strings):
        for j, ch in enumerate(s):
            strings_encoded[i, j] = CHAR_TO_IDX[ch]

    str_idx, fixed_mask, masked_count = find_best_mask(strings_encoded, n_strings, str_len)

    best_str = operator_strings[str_idx]
    pattern = [best_str[pos] if fixed_mask & (1 << pos) else '?' for pos in range(str_len)]
    return best_str, ''.join(pattern), masked_count, str_idx


def generate_mlm_benchmark(digits: List[int], max_target: int, min_target: int = 0) -> Dict[int, List[Tuple[str, str]]]:
    dp_results = dp_solver_full(digits)
    benchmark: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    for skeleton, value_map in dp_results.items():
        op_positions = [i for i, ch in enumerate(skeleton) if ch == '#']
        if not op_positions:
            continue

        for target, expressions in value_map.items():
            if target < min_target or target > max_target:
                continue
            if not expressions:
                continue

            op_strings = [''.join(expr[i] for i in op_positions) for expr in expressions]
            best_ops, pattern_ops, masked_count, _ = solve(op_strings)

            if masked_count == 0:
                continue

            # Build full expression string with masks and digits
            masked_full = []
            op_idx = 0
            for ch in skeleton:
                if ch == '#':
                    masked_full.append(pattern_ops[op_idx])
                    op_idx += 1
                else:
                    masked_full.append(ch)
            masked_expr = ''.join(masked_full)

            # Operators that should replace '?' in the pattern
            mask_operators = ''.join(
                best_ops[i] for i, ch in enumerate(pattern_ops) if ch == '?'
            )

            benchmark[target].append((masked_expr, mask_operators))

    return dict(benchmark)


def tokenize_example(masked_expr: str, mask_ops: str, target: int) -> Tuple[str, str]:
    """
    Convert an example into two strings: input (expression=target) and labels
    (symbol 'p' everywhere except '?' positions where the correct operator is placed).
    """
    inp = f"{masked_expr}={target}"
    lab_chars = []
    op_idx = 0
    for ch in masked_expr:
        if ch == '?':
            lab_chars.append(mask_ops[op_idx])
            op_idx += 1
        else:
            lab_chars.append('p')
    lab_chars.append('p')  # for '='
    lab_chars.extend(['p'] * len(str(target)))
    return inp, ''.join(lab_chars)


def sample_patterns_for_target(patterns: List[Tuple[str, str]], min_masks: int = 0) -> List[Tuple[str, str]]:
    by_mask_count = defaultdict(list)
    with_div = []

    for expr, ops in patterns:
        n_masks = expr.count('?')
        if n_masks < min_masks:
            continue
        by_mask_count[n_masks].append((expr, ops))
        if '/' in ops:
            with_div.append((expr, ops))

    sampled = [random.choice(by_mask_count[m]) for m in sorted(by_mask_count)]
    if with_div:
        div_example = random.choice(with_div)
        if div_example not in sampled:
            sampled.append(div_example)
    return sampled


def worker(digits: List[int], max_target: int, min_target: int):
    return generate_mlm_benchmark(digits, max_target, min_target)


def generate_benchmarks_parallel(digit_sets: List[List[int]], max_target: int, min_target: int, num_processes: int):
    with mp.Pool(processes=num_processes) as pool:
        func = partial(worker, max_target=max_target, min_target=min_target)
        return list(tqdm(pool.imap(func, digit_sets, chunksize=100),
                         total=len(digit_sets), desc="Generating benchmarks"))


def collect_examples_from_benchmarks(benches: List[Dict], n_per_bench: int, min_masks: int, seed: int):
    random.seed(seed)
    all_examples = []

    for bench in tqdm(benches, desc="Collecting examples"):
        sampled = []
        for target in bench:
            local_sample = sample_patterns_for_target(bench[target], min_masks)
            sampled.extend([(expr, ops, target) for expr, ops in local_sample])

        other = [(expr, ops, tgt) for tgt, pats in bench.items()
                 for expr, ops in pats if (expr, ops, tgt) not in {tuple(s) for s in sampled}]

        needed = n_per_bench - len(sampled)
        if needed > 0:
            if len(other) <= needed:
                sampled.extend(other)
            else:
                sampled.extend(random.sample(other, needed))

        all_examples.extend(sampled)

    return all_examples


def save_examples(examples: List[Tuple[str, str, int]], filepath: str):
    """Save examples to a JSONL file (fields: input, labels, target)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for masked_expr, mask_ops, target in examples:
            inp, lab = tokenize_example(masked_expr, mask_ops, target)
            f.write(f'{{"input": "{inp}", "labels": "{lab}", "target": {target}}}\n')


def generate_digit_sets(n: int, min_len: int, max_len: int,
                        digit_min: int, digit_max: int, seed: int) -> List[List[int]]:
    """Generate n unique random digit sets."""
    random.seed(seed)
    sample_set = set()
    while len(sample_set) < n:
        length = random.randint(min_len, max_len)
        sample_set.add(tuple(random.randint(digit_min, digit_max) for _ in range(length)))
    tuples = list(sample_set)
    random.shuffle(tuples)
    return [list(t) for t in tuples]


cli = ArgParser()


@cli.command(singleton=True)
def preprocess_data(config: Config):
    print("Generating MLM dataset for arithmetic expressions (RPN)")

    all_digits = generate_digit_sets(
        config.num_digit_samples,
        config.min_digits,
        config.max_digits,
        config.digits_range[0],
        config.digits_range[1],
        config.seed
    )

    # Split into train / test
    split_idx = int(len(all_digits) * config.train_ratio)
    train_digit_sets = all_digits[:split_idx]
    test_digit_sets = all_digits[split_idx:]

    print(f"Train digit sets: {len(train_digit_sets)}")
    print(f"Test digit sets: {len(test_digit_sets)}")

    # Generate benchmarks
    # For train use the first interval (maximum max_target from train_intervals)
    train_max_target = max(e[1] for e in config.train_intervals)
    train_min_target = min(e[0] for e in config.train_intervals)

    train_benches = generate_benchmarks_parallel(
        train_digit_sets, train_max_target, train_min_target, config.num_processes
    )

    # For test combine all intervals (overall max_target and min_target)
    test_max_target = max(e[1] for e in config.test_intervals)
    test_min_target = min(e[0] for e in config.test_intervals)

    test_benches = generate_benchmarks_parallel(
        test_digit_sets, test_max_target, test_min_target, config.num_processes
    )

    # Collect examples
    train_examples = collect_examples_from_benchmarks(
        train_benches, config.n_per_bench, config.min_masks, config.seed
    )
    test_examples = collect_examples_from_benchmarks(
        test_benches, config.n_per_bench, config.min_masks, config.seed + 1
    )

    save_examples(train_examples, os.path.join(config.output_dir, "train.jsonl"))
    save_examples(test_examples, os.path.join(config.output_dir, "test.jsonl"))

    print(f"Done! Generated {len(train_examples)} train and {len(test_examples)} test examples.")


if __name__ == "__main__":
    cli()
