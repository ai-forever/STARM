import multiprocessing as mp
import os
import random
import re
from itertools import groupby
from typing import List, Optional

import lifelib
import numpy as np
import pandas as pd
from argdantic import ArgParser
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

tqdm.pandas()

RLE_TO_BINARY_MAPPER = {'b': '0', 'o': '1', '$': '$'}
BINARY_TO_RLE_MAPPER = dict(zip(RLE_TO_BINARY_MAPPER.values(), RLE_TO_BINARY_MAPPER.keys()))

sess = lifelib.load_rules("b3s23")
LIFE_TREE = sess.lifetree()


class DataProcessConfig(BaseModel):
    seed: int = 267
    n: int = 200_000
    alive_probability: float = 0.4
    max_len: int = 17
    max_step: int = 50
    num_processes: int = 64
    limit: int = 250
    format_name: str = "bin"  # "bin" or "rle"
    output_dir: str = "game_of_life"
    test_split_size: float = 0.15


def get_rle_string(pattern):
    raw_rle = pattern.rle_string()
    lines = raw_rle.strip().split('\n')

    clean_lines = []
    for line in lines:
        if line.startswith('#') or line.strip().startswith('x ='):
            continue
        line = line.replace('.', 'b').replace('A', 'o')
        clean_lines.append(line)

    return ''.join(clean_lines)


def strip_rle_pattern(rle_pattern: str) -> str:
    pattern = LIFE_TREE.pattern(rle_pattern)
    return get_rle_string(pattern[0])


def convert_binary_to_rle(binary_pattern: str) -> str:
    clean_data = re.sub(r'[^01$]', '', binary_pattern.replace('\n', ''))

    if not clean_data:
        return "!"

    rle_parts = []

    for char, group in groupby(clean_data):
        count = sum(1 for _ in group)
        symbol = BINARY_TO_RLE_MAPPER.get(char)

        if symbol:
            if count > 1:
                rle_parts.append(str(count))
            rle_parts.append(symbol)

    rle_parts.append('!')

    return strip_rle_pattern(''.join(rle_parts))


def convert_rle_to_binary(rle_pattern: str) -> str:
    clean_data = re.sub(r'\s', '', rle_pattern.replace('\n', ''))

    if not clean_data:
        return ""

    tokens = re.findall(r'(\d*)([bo$!])', clean_data)

    def expand(count_str: str, symbol: str) -> str:
        count = int(count_str) if count_str else 1
        return RLE_TO_BINARY_MAPPER.get(symbol, '') * count

    return ''.join(expand(count, sym) for count, sym in tokens)


def safe_get_period(pattern) -> Optional[int]:
    try:
        period = pattern.oscar(verbose=False, eventual_oscillator=False, allow_guns=False).get('period')
    except Exception:
        period = None
    return period


def generate_random_binary_pattern(x: int, y: int, alive_probability: float = 0.5, seed: int = 42) -> str:
    random.seed(seed)

    if not 0.0 <= alive_probability <= 1.0:
        raise ValueError("alive_probability must be between 0 and 1")

    if x <= 0 or y <= 0:
        raise ValueError("Dimensions must be positive")

    pattern_rows = []

    for row in range(y):
        row_cells = []
        for col in range(x):
            cell = '1' if random.random() < alive_probability else '0'
            row_cells.append(cell)
        pattern_rows.append(''.join(row_cells))

    binary_pattern = '$'.join(pattern_rows)

    return binary_pattern


def add_test_train_split(df):
    n = len(df)

    # 1. First 10% → train (fixed)
    first_10_percent_idx = int(n * 0.10)
    df.loc[:first_10_percent_idx - 1, 'part'] = 'train'

    # 2. Last quarter (25%) → test-out
    start_test_out_idx = int(n * 0.75)
    df.loc[start_test_out_idx:, 'part'] = [f'test-out-{i}' for i in range(1, len(df) - start_test_out_idx + 1)]

    # 3. Row before test-out → train
    boundary_train_idx = start_test_out_idx - 1
    df.loc[boundary_train_idx, 'part'] = 'train'

    # 4. Remaining rows (between first 10% and row before test-out)
    # Randomly distribute 2/3 train and 1/3 test-in
    middle_start_idx = first_10_percent_idx
    middle_end_idx = boundary_train_idx - 1  # excluding boundary_train_idx

    if middle_end_idx >= middle_start_idx:
        middle_indices = list(range(middle_start_idx, middle_end_idx + 1))
        np.random.shuffle(middle_indices)  # Shuffle indices

        # Calculate number for test-in (1/3 of middle)
        test_in_count = int(len(middle_indices) * (1 / 3))

        # First test_in_count indices after shuffling → test-in
        test_in_indices = middle_indices[:test_in_count]
        df.loc[test_in_indices, 'part'] = 'test-in'

        # Remaining indices → train
        train_indices = middle_indices[test_in_count:]
        df.loc[train_indices, 'part'] = 'train'

    return df


def process_df_parallel(df, max_step, num_processes=4):
    """
    Parallel processing of DataFrame with RLE patterns.

    Parameters:
        df : original DataFrame with column 'rle_pattern'
        max_step : maximum number of evolution steps
        num_processes : number of processes

    Returns:
        df_final : merged DataFrame with all steps
    """
    # Convert DataFrame to list of dicts for easy passing to processes
    data_list = df.to_dict('records')

    manager = mp.Manager()
    results_queue = manager.Queue()  # queue to collect results from processes

    def process_chunk(chunk, process_id):
        """Function executed in each process."""
        print(f"Process {process_id} started, processing {len(chunk)} rows.")

        # Initialize lifelib inside the process (not shared between processes)
        import lifelib
        sess = lifelib.load_rules("b3s23")
        lt = sess.lifetree()

        chunk_results = []  # all records for current chunk

        for row_dict in tqdm(chunk, desc=f"Process {process_id}"):
            pattern = lt.pattern(row_dict['rle_pattern'])
            records_for_pattern = []  # records for a single pattern

            step = 1
            period = None
            start_step = None

            while not pattern[step].empty():
                # Attempt to determine period
                if period is None:
                    current_period = safe_get_period(pattern[step])
                    if current_period is not None:
                        period = current_period
                        start_step = step

                # Stop by period
                if period is not None:
                    if step >= start_step + period:
                        break
                    if period == 1 and step > start_step:
                        break

                # Save step data
                rle = get_rle_string(pattern[step])
                record = {'step': step, 'rle_pattern_after_step': rle}
                record.update(row_dict)  # add all fields from original row
                records_for_pattern.append(record)

                step += 1
                if step > max_step:
                    break

            # Create DataFrame for pattern and apply split
            if records_for_pattern:
                df_temp = pd.DataFrame(records_for_pattern)
                df_temp = add_test_train_split(df_temp)
                chunk_results.extend(df_temp.to_dict('records'))

        # Send chunk results to queue
        results_queue.put(chunk_results)

    # Split data into chunks by number of processes
    chunk_size = len(data_list) // num_processes
    chunks = [data_list[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes - 1)]
    chunks.append(data_list[(num_processes - 1) * chunk_size:])

    # Start processes
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=process_chunk, args=(chunks[i], i))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Collect results from queue (exactly one item from each process)
    all_chunk_results = []
    for _ in range(num_processes):
        all_chunk_results.extend(results_queue.get())

    # Final DataFrame
    if all_chunk_results:
        df_final = pd.DataFrame(all_chunk_results)
    else:
        df_final = pd.DataFrame()

    return df_final


def get_bounding_box(rle_pattern: str) -> List[int]:
    return LIFE_TREE.pattern(rle_pattern)[0].bounding_box


def count_area(bounding_box: List[int]) -> int:
    return bounding_box[2] * bounding_box[3]


cli = ArgParser()


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    assert config.format_name in ("bin", "rle"), "format must be 'bin' or 'rle'"
    format_name = config.format_name

    # Generate random binary patterns
    print("Generating random patterns...")
    df = pd.DataFrame(columns=['bin_pattern', 'w', 'h'])

    for i in trange(config.n, desc="Generating"):
        random.seed(config.seed * i)
        w = random.randint(2, config.max_len)
        h = random.randint(2, config.max_len)
        generated = generate_random_binary_pattern(w, h, config.alive_probability, config.seed * i)
        df.loc[i] = [generated, w, h]

    df = df.drop_duplicates()
    print(f"Unique patterns: {len(df)}")

    # Convert to RLE
    df['rle_pattern'] = df['bin_pattern'].progress_apply(convert_binary_to_rle)

    # Evolve and split
    print("Evolving patterns...")
    df_final = process_df_parallel(df, max_step=config.max_step, num_processes=config.num_processes)
    df_final = df_final.dropna(subset=['rle_pattern'])
    df_final['bin_pattern_after_step'] = df_final['rle_pattern_after_step'].apply(convert_rle_to_binary)
    df_final = df_final[['w', 'h', 'bin_pattern', 'rle_pattern', 'step',
                         'bin_pattern_after_step', 'rle_pattern_after_step', 'part']]

    # Bounding boxes and areas (computed from RLE)
    df_final['input_bounding_box'] = df_final['rle_pattern'].progress_apply(get_bounding_box)
    df_final['output_bounding_box'] = df_final['rle_pattern_after_step'].progress_apply(get_bounding_box)

    df_final['input_area'] = df_final['input_bounding_box'].apply(count_area)
    df_final['output_area'] = df_final['output_bounding_box'].apply(count_area)

    df_final['step'] = df_final['step'].astype(int)

    df_final['bin_pattern'] = df_final['bin_pattern'].apply(lambda pattern: pattern.replace('0', 'b').replace('1', 'o'))
    df_final['bin_pattern_after_step'] = df_final['bin_pattern_after_step'].apply(
        lambda pattern: pattern.replace('0', 'b').replace('1', 'o'))

    # Keep necessary format
    df_final = df_final[
        ['w', 'h', f'{format_name}_pattern', 'step',
         f'{format_name}_pattern_after_step', 'input_bounding_box',
         'output_bounding_box', 'input_area', 'output_area', 'part'
         ]]

    df_final = df_final.rename(columns={
        f'{format_name}_pattern': 'pattern',
        f'{format_name}_pattern_after_step': 'pattern_after_step',
    })

    # Length of patterns in each format
    df_final['max_len_pattern'] = df_final.apply(
        lambda x: max(len(x['pattern']), len(x['pattern_after_step'])), axis=1
    )

    # Split into train / test
    df_train = df_final[df_final['part'] == 'train'].copy()
    df_train, df_not_train = train_test_split(df_train, test_size=config.test_split_size, random_state=42)

    # Train
    df_train_clean = df_train[(
            (df_train['output_area'] <= config.limit) &
            (df_train['input_area'] <= config.limit) &
            (df_train['max_len_pattern'] <= config.limit)
    )]

    print("Saving TRAIN part...")
    train_path = os.path.join(config.output_dir, "train.jsonl")
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    df_train_clean.to_json(train_path, orient="records", lines=True, force_ascii=False)

    # Test in_train-in_range
    df_test_in_train_in_range = df_final[df_final['part'] == 'test-in_train-in_range']
    df_test_in_train_in_range = df_test_in_train_in_range[
        df_test_in_train_in_range['pattern'].isin(df_train_clean['pattern'].unique())
    ]

    if len(df_test_in_train_in_range) > 0:
        print("Saving TEST in_train-in_range part...")
        test_in_train_in_range_path = os.path.join(config.output_dir, "test", "in_train_in_range.jsonl")
        os.makedirs(os.path.dirname(test_in_train_in_range_path), exist_ok=True)
        df_test_in_train_in_range.to_json(test_in_train_in_range_path, orient="records", lines=True, force_ascii=False)

    # Test in_train-out_range
    for step in [1, 2, 3, 10]:
        df_in_train_out_range = df_final[df_final['part'] == f'test-in_train-out_range-{step}']
        df_in_train_out_range = df_in_train_out_range[
            df_in_train_out_range['pattern'].isin(df_train_clean['pattern'].unique())
        ]

        if len(df_in_train_out_range) > 0:
            print(f"Saving TEST in_train_out_range-{step} part...")
            test_in_train_out_range_path = os.path.join(config.output_dir, "test", f"in_train_out_range-{step}.jsonl")
            os.makedirs(os.path.dirname(test_in_train_out_range_path), exist_ok=True)
            df_in_train_out_range.to_json(test_in_train_out_range_path, orient="records", lines=True, force_ascii=False)

    # Test out_train
    df_not_train_clean = df_not_train[(
            (df_not_train['output_area'] <= config.limit) &
            (df_not_train['input_area'] <= config.limit) &
            (df_not_train['max_len_pattern'] <= config.limit)
    )]

    print("Saving TEST out_train part...")
    test_out_train_path = os.path.join(config.output_dir, "test", "out_train.jsonl")
    os.makedirs(os.path.dirname(test_out_train_path), exist_ok=True)
    df_not_train_clean.to_json(test_out_train_path, orient="records", lines=True, force_ascii=False)

    print("Data preparation complete.")


if __name__ == "__main__":
    cli()
