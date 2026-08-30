# HTTP inference API

用于部署本仓库训练出的检查点，或 Hugging Face Hub 上采用[该格式](https://huggingface.co/sapientinc/HRM-checkpoint-sudoku-extreme)
的检查点的 HTTP 服务。

🇬🇧 [英文 README](./README.md) | 🇷🇺 [俄文 README](./README_rus.md)

## 快速开始

```bash
pip install -r api/requirements.txt
./host_model --path_directory checkpoints/<project>/<run> --port 8080

curl -s localhost:8080/generate -d '{"task": "sudoku", "input": "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"}'
```

`./host_model` 是 [`api/host_model.py`](./host_model.py) 的封装，二者参数相同：

| 参数               | 用途                                                            |
|--------------------|-----------------------------------------------------------------|
| `--path_directory` | 训练目录、检查点文件，或 Hugging Face 仓库 id                    |
| `--port`、`--host` | 监听位置；默认为 `0.0.0.0` 的 `8080`                             |
| `--device`         | `cuda:1`，或只写下标                                           |
| `--seq-len`        | 序列长度，用于 `all_config.yaml` 中没有该值时                    |
| `--vocab-map`      | 词表，用于该次训练的词表与 [`api/vocab_maps/`](./vocab_maps) 中的不同时 |
| `--task`           | 覆盖依据 `all_config.yaml` 中 `data_path` 判定出的任务           |
| `--no-ema`         | 即使存在 EMA 影子，也加载基础权重                                |
| `--revision`       | Hugging Face 的分支、标签或提交                                  |

若传入的是训练目录，则取最新的 `step_N`，存在 EMA 权重时优先使用。

## 接口

| 方法  | 路径        | 用途                          |
|-------|-------------|-------------------------------|
| POST  | `/generate` | 单条请求或批量                 |
| GET   | `/info`     | 检查点、任务、词表、序列长度    |
| GET   | `/health`   | 存活探针                       |
| GET   | `/docs`     | 交互式 OpenAPI 文档            |

一条请求是 `{"task": ..., "input": ...}`，ARC 还需加上 `puzzle_id`；响应会保持你发送的形态：

- `output`——模型输出的内容，去掉填充之后的结果；
- `steps`——该请求在其 ACT 头判定答案已就绪之前所经历的递归深度；
- `max_steps`——递归步数的上限。

## 各任务的输入格式

| 任务           | 输入                                                    |
|----------------|---------------------------------------------------------|
| `sudoku`       | 按行排列的 81 个单元格，空格用 `.` 或 `0`；空白字符被忽略 |
| `maze`         | 由迷宫字符组成的方形网格：分行书写，或写成一整行           |
| `arc`          | 颜色数字组成的网格，各行以 `<eos>` 或换行分隔              |
| `arithmetic`   | `3?5+2?7=42`——`?` 标出每个待恢复的运算符                  |
| `game_of_life` | `bbo$obb\|3`——先是图案，然后是代数                        |

ARC 还需要 `puzzle_id`——采用 ARC-AGI 记法的任务 id（`"007bbfb7"`），用于选中学到的 puzzle
嵌入。合法 id 来自 `identifiers.json`，该文件必须与 `all_config.yaml` 放在一起。

各任务的请求示例：

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

## 序列长度

每条请求都会补齐到该次训练所用的长度，因此这个值必须正确：传错不会报错，只会让答案变成噪声。
它从 `all_config.yaml` 读取，或通过 `--seq-len` 传入，其来源是训练数据集的 `dataset.json`。

本仓库所构建数据集的示例取值：

| 任务           | `seq_len` |
|----------------|-----------|
| `sudoku`       | 81        |
| `maze`         | 900       |
| `arc`          | 900       |
| `arithmetic`   | 19        |
| `game_of_life` | 243       |

## 词表

token id 是 `dataset/build_*_dataset.py` 中的常量，因此本仓库在
[`api/vocab_maps/`](./vocab_maps) 中为每个任务提供了词表。如有需要，可通过 `--vocab-map`
覆盖它。

## 来自 Hugging Face Hub 的检查点

`--path_directory` 同样接受仓库 id：下载一次之后即按本地目录对待：

```bash
./host_model --path_directory sapientinc/HRM-checkpoint-sudoku-extreme --seq-len 81 --port 8080
```

只有当本地不存在同名路径时，`org/name` 才会被当作仓库 id；`hf://org/name` 则强制如此。这类
仓库只包含 `all_config.yaml` 与权重，因此必须传入 `--seq-len`，而架构代码取自本仓库的
`models/`，而非检查点自带的副本。

## 同时部署多个模型

一个进程在一块 GPU 上服务一个检查点，因此多卡机器上按模型各起一个服务：

```bash
./host_model --path_directory checkpoints/sudoku/<run>     --device cuda:0 --port 8080 &
./host_model --path_directory checkpoints/arithmetic/<run> --device cuda:1 --port 8081 &
```

`CUDA_VISIBLE_DEVICES` 同样可用——此时进程会把自己的卡看作 `cuda:0`。
