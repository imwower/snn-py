# snn-py

The package uses the standard library logging module. Call `snn_py.logging_config.setup()` early in your program (CLI entry point, test bootstrap, or interactive session) to configure output formatting and align logger levels. By default the log level is `INFO`, but you can override it via the `SNN_PY_LOGLEVEL` environment variable.

Typical usage during development:

```bash
SNN_PY_LOGLEVEL=DEBUG python -m unittest -v
```

## 项目概览

snn-py 是一个面向 SNN（Spiking Neural Network）工作流的实验/演示平台，提供：

- **统一日志与运行清单**：`logging_config` 输出 JSON 行日志，`manifest` 记录 run_id、Git SHA、策略哈希、环境变量与随机种子，便于回溯。
- **可复现的随机流**：`SeedManager` 通过 base seed + 名字稳定派生 RNG，保障多线程/多模块改动后仍可重放。
- **通用指标库**：`core.metrics` 提供群体发放率、Fano 因子、稳定性与可靠性统计，可在 demo/runner/replay 等场景复用。
- **多条 CLI 链路**：`cli.demo`、`pipeline.runner`、`cli.replay`、`cli.sweep` 覆盖从生成、并发推理到结果回放的完整闭环。
- **可扩展策略/插件体系**：策略 JSON 定义可用工具，`plugins.registry` 允许通过 `module:Class` 动态加载 scorer/gate。

以下章节介绍如何测试、运行以及调试这些组件。

## 测试与演示

在提交前可以使用 INFO 级别日志跑完整测试：

```bash
SNN_PY_LOGLEVEL=INFO python -m unittest -v
```

若要体验端到端流程（LIF → 群体率 → Up/Down → 意向门 → 记忆 → 审计），可运行演示 CLI：

```bash
python -m snn_py.cli.demo \
  --policy examples/policy.demo.json \
  --jsonl-dir ./episodes \
  --loglevel INFO
```

执行过程中会打印 JSON 日志，最终输出 `run_complete` 事件以及生成的提案数（`episodes/` 下会保存 `*.jsonl.gz` 片段）。

需要完整的并发训练/推理流水线时，使用 Pipeline Runner：

```bash
python -m snn_py.pipeline.runner \
  --policy examples/policy.demo.json \
  --jsonl-dir ./episodes \
  --max-events 50 \
  --timeout-s 1.0 \
  --loglevel INFO
```

该命令会启动多线程生产/意向/审计环节，最后汇报 `pipeline_complete` 并保证提案写入同一 `episodes/` 目录。

## 安装与快速开始

```bash
python -m pip install .
python -m snn_py.cli.doctor --policy policy.example.json --check-dir .
python -m snn_py.cli.demo --help
```

## 工具与日志事件速览

### 回放（Replay）

```bash
snnpy-replay --jsonl-dir ./episodes --since 10
```

日志事件：

- `replay_loaded`：统计读入文件与集数。
- `replay_timeline`：逐个 `run_id` 输出时间线与片段。
- `replay_summary`：汇总运行与提案计数。

### 参数扫（Sweep）

```bash
snnpy-sweep --seeds 1,2 --theta 0.7:1.0:0.1 --decay 0.5,0.7 --T 5
```

生成 `results.json`，日志事件：

- `sweep_case_done`：记录单个参数组合及触发次数。
- `sweep_complete`：汇报总运行案例数。

### 并发 Runner

```bash
python -m snn_py.pipeline.runner --policy policy.example.json --max-events 50 --timeout-s 1.0
```

或使用安装后的脚本执行同等命令。关键日志：

- `pipeline_start`：管线启动。
- `segment_produced`/`intent_fired`/`audit_decision`：阶段性内部步骤（INFO 级别）。
- `pipeline_complete`：提供生产与消费计数摘要。

### 快照（Checkpoint）

在 LIF 或 ClusterWLC 模型中：

```python
lif = LIF(cfg, seed=7)
lif.save_json("lif.json")
restored = LIF.load_json("lif.json")
```

日志事件：

- `checkpoint_saved`：保存快照成功。
- `checkpoint_loaded`：从快照恢复成功。

## 文本叙述流水线（训练 → 生成 → 评测）

这一套组件允许你持续吞入新的自然语料、训练 n-gram 模型、实时跟随 proposals，并离线评估 perplexity/OOV。仓库已经提供了基于常用词库随机生成的中文对话语料（`corpus/dialog_cn_common.txt`、`corpus/dialog_cn_common.txt.gz`），方便开箱即用；也建议在 `models/`、`corpus/`、`runs/` 等目录下安排你自己的输入输出。

### 1. 一次性训练或增量更新

`learn_watch` 会扫描 `corpus/` 目录下的 `.txt/.txt.gz` 文件生成 fingerprint，并在文件发生变化时重新训练或更新模型。重复运行即可自动增量。

```bash
python -m snn_py.cli.learn_watch \
  --corpus corpus/ \
  --state runs/corpus.fp.json \
  --out models/ngram.json
```

### 2. 持续运行：Proposals → Narrations

先确保 `runs/proposals.jsonl` 持续追加 `proposal` 事件。若暂时没有 Runner，可用内置工具快速生成：

```bash
python -m snn_py.cli.mock_proposals \
  --out runs/proposals.jsonl \
  --count 20 \
  --interval-ms 200
```

然后让 `scribe_loop` tail 该文件（或参考 `tests/test_end2end_longrun.py` 用线程写入模拟 proposals），它会在发现新 proposal 时调用训练好的模型生成 narration。

```bash
python -m snn_py.cli.scribe_loop \
  --in runs/proposals.jsonl \
  --out runs/narrations.jsonl \
  --state runs/scribe.state.json \
  --model models/ngram.json \
  --poll-ms 200
```

### 3. 验证训练成果（离线）

用 held-out 语料评测 perplexity 与 OOV 覆盖率，期望随着语料增多指标逐步下降。

```bash
python -m snn_py.cli.text_eval \
  --model models/ngram.json \
  --corpus corpus_val/
```

`text_eval_complete` 日志会包含 `{"perplexity": ..., "oov_ratio": ...}`。

### 4. 端到端一键回归

跑一遍关键测试确保持续集成：scribe loop、corpus watcher、text eval 以及 proposal→narration 长跑。

```bash
python -m unittest -v \
  tests.test_scribe_loop \
  tests.test_corpus_watch \
  tests.test_text_eval \
  tests.test_end2end_longrun
```

### 一键示例：中文语料→训练→解码

若使用仓库自带的中文对话语料，可按以下顺序执行命令，完成从语料训练、生成提案到输出解码文本的完整流程：

```bash
# 1. 训练或增量更新中文 n-gram 模型
python -m snn_py.cli.learn_watch \
  --corpus corpus \
  --state runs/corpus_cn.fp.json \
  --out models/ngram_cn.json

# 2. （可选）快速生成提案事件以驱动 scribe loop
python -m snn_py.cli.mock_proposals \
  --out runs/proposals_cn.demo.jsonl \
  --count 20 \
  --interval-ms 200

# 3. 启动 scribe loop，实时生成并记录中文叙述
python -m snn_py.cli.scribe_loop \
  --in runs/proposals_cn.demo.jsonl \
  --out runs/narrations_cn.demo.jsonl \
  --state runs/scribe_cn.state.json \
  --model models/ngram_cn.json \
  --poll-ms 200

# 4. 观察解码结果（日志与 JSONL 均为 UTF-8 输出）
tail -f runs/narrations_cn.demo.jsonl
```

`scribe_loop` 默认使用 `ensure_ascii=False` 打印 `narration_text`，因此中文内容无需额外解码就能直接在终端与 JSONL 中查看。若希望完全移除英文提示词，可自定义 `context` 或 fork CLI 实现。
