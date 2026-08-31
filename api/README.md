# HTTP inference API

An HTTP service for serving a checkpoint trained in this repository, or a checkpoint from the
Hugging Face Hub in [this format](https://huggingface.co/sapientinc/HRM-checkpoint-sudoku-extreme).

🇷🇺 [README in Russian](./README_rus.md) | 🇨🇳 [README in Chinese](./README_zh.md)

## Quick start

```bash
pip install -r api/requirements.txt
./host_model --path_directory checkpoints/<project>/<run> --port 8080

curl -s localhost:8080/generate -d '{"task": "sudoku", "input": "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"}'
```

`./host_model` is a wrapper around [`api/host_model.py`](./host_model.py); both take the same flags:

| Flag               | Purpose                                                                          |
|--------------------|----------------------------------------------------------------------------------|
| `--path_directory` | Run directory, a checkpoint file, or a Hugging Face repo id                       |
| `--port`, `--host` | Where to listen; `8080` on `0.0.0.0` by default                                   |
| `--device`         | `cuda:1`, or just the index                                                      |
| `--seq-len`        | Sequence length, when `all_config.yaml` does not have one                         |
| `--vocab-map`      | Vocabulary, when the run's differs from those in [`api/vocab_maps/`](./vocab_maps) |
| `--task`           | Override the task detected from `data_path` in `all_config.yaml`                  |
| `--no-ema`         | Load the base weights even when an EMA shadow is present                          |
| `--revision`       | Hugging Face branch, tag or commit                                                |

Given a run directory, the newest `step_N` is used, preferring EMA weights when present.

## Endpoints

| Method | Path        | Purpose                                       |
|--------|-------------|-----------------------------------------------|
| POST   | `/generate` | One prompt, or a batch                        |
| GET    | `/info`     | Checkpoint, task, vocabulary, sequence length |
| GET    | `/health`   | Liveness probe                                |
| GET    | `/docs`     | Interactive OpenAPI docs                      |

A prompt is `{"task": ..., "input": ...}`, plus `puzzle_id` for ARC; the response mirrors the shape you sent:

- `output` — what the model emitted, with padding removed;
- `steps` — the recursion depth this prompt took before its ACT head decided the answer was
ready;
- `max_steps` — the maximum number of recursion steps.

## Input format per task

| Task           | Input                                                          |
|----------------|----------------------------------------------------------------|
| `sudoku`       | 81 cells row-major, `.` or `0` for blanks; whitespace ignored   |
| `maze`         | Square grid of maze characters: rows, or one flat line          |
| `arc`          | Grid of colour digits, rows separated by `<eos>` or newlines    |
| `arithmetic`   | `3?5+2?7=42` — `?` marks each operator to recover               |
| `game_of_life` | `bbo$obb\|3` — pattern, then the number of generations           |

ARC additionally requires `puzzle_id` — the task id in ARC-AGI notation (`"007bbfb7"`) — which
selects the learned puzzle embedding. Valid ids come from `identifiers.json`, which has to sit
next to `all_config.yaml`.

Request examples for every task:

```bash
# sudoku
curl -s localhost:8080/generate -d '{"task": "sudoku", "input": "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"}'

# maze
curl -s localhost:8080/generate -d '{"task": "maze", "input": "S#   \n  ## \n# #  \n  ###\n   #G"}'

# arc
curl -s localhost:8080/generate -d '{"task": "arc", "input": "077<eos>777<eos>077", "puzzle_id": "007bbfb7"}'

# arithmetic
curl -s localhost:8080/generate -d '{"task": "arithmetic", "input": "3?5+2?7=42"}'

# game_of_life
curl -s localhost:8080/generate -d '{"task": "game_of_life", "input": "bbo$obb$bob|3"}'
```

## Sequence length

Every prompt is padded to the length the run was trained at, so the value has to be right: a
wrong one does not raise an error, it turns answers into noise. It is read from
`all_config.yaml` or passed with `--seq-len`, and it lives in the training dataset's
`dataset.json`.

Example values for the datasets this repository builds:

| Task           | `seq_len` |
|----------------|-----------|
| `sudoku`       | 81        |
| `maze`         | 900       |
| `arc`          | 900       |
| `arithmetic`   | 19        |
| `game_of_life` | 243       |

## Vocabularies

Token ids are constants of `dataset/build_*_dataset.py`, so the repository ships a vocabulary
for every task in [`api/vocab_maps/`](./vocab_maps). It can be overridden with `--vocab-map`.

## Checkpoints from the Hugging Face Hub

`--path_directory` also accepts a repo id: it is downloaded once and then treated like a local
directory:

```bash
./host_model --path_directory sapientinc/HRM-checkpoint-sudoku-extreme --seq-len 81 --port 8080
```

`org/name` is read as a repo id only when no such path exists locally; `hf://org/name` forces
it. Such a repo carries only `all_config.yaml` and the weights, so `--seq-len` is required, and
the architecture code comes from this repository's `models/` rather than from a copy of the
checkpoint's own.

## Serving several models at once

One process serves one checkpoint on one GPU, so a machine with several cards runs one server
per model:

```bash
./host_model --path_directory checkpoints/sudoku/<run>     --device cuda:0 --port 8080 &
./host_model --path_directory checkpoints/arithmetic/<run> --device cuda:1 --port 8081 &
```

`CUDA_VISIBLE_DEVICES` works too — then the process sees its card as `cuda:0`.
