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
  --poll-ms 200 \
  --context 叙述者

# 4. 观察解码结果（日志与 JSONL 均为 UTF-8 输出）
tail -f runs/narrations_cn.demo.jsonl
```

`scribe_loop` 默认使用 `ensure_ascii=False` 打印 `narration_text`，因此中文内容无需额外解码就能直接在终端与 JSONL 中查看。若希望完全移除英文提示词，可通过 `--context` 传入任意中文提示词（默认已使用 `叙述者`），或根据需要 fork CLI 实现。

# 5. 一键串联训练、提案与解码
```bash
python -m snn_py.cli.learn_watch \
  --corpus corpus \
  --state runs/corpus_cn.fp.json \
  --out models/ngram_cn.json && \
python -m snn_py.cli.mock_proposals \
  --out runs/proposals_cn.showcase.jsonl \
  --count 5 \
  --interval-ms 0 \
  --seed 13 && \
python -m snn_py.cli.scribe_loop \
  --in runs/proposals_cn.showcase.jsonl \
  --out runs/narrations_cn.showcase.jsonl \
  --state runs/scribe_cn.showcase.state.json \
  --model models/ngram_cn.json \
  --poll-ms 50 \
  --max-events 5
```

示例解码（节选自 `runs/narrations_cn.showcase.jsonl`）：

```text
the agent 火 常 糖 轮 正 斑 水 化 阳
the agent 零 奶 于 高 沙 形 两 生 邮 甜 森 活 算
the agent 腿 树 自 声 在 肉 语 林 作 视 伴 八 冰
the agent 比 一 用 果 林 能 通 力 处 储 甜 看 漠 鼻
the agent 蜘 闪 声 北 雨 常 氧 铁 据
```

# 6. 持续学习 + 主动输出

如果希望一条命令搞定“监控语料→必要时重训→定期输出中文叙述”，可以使用自带的 `auto_narrate`：

```bash
python -m snn_py.cli.auto_narrate \
  --corpus corpus \
  --state runs/corpus_cn.fp.json \
  --model models/ngram_cn.json \
  --out runs/narrations_cn.auto.jsonl \
  --emit-ms 1500 \
  --context 叙述者
```

该进程会持续运行：当 `corpus/` 下的 `.txt/.txt.gz` 发生变化时自动增量训练模型，并按 `--emit-ms` 节奏生成 `auto_narration` 事件写入 JSONL 文件（日志同步打印 `narration_text`）。无需额外的 proposal 驱动即可获得稳定中文输出，如需停止可 `Ctrl+C` 或指定 `--max-events`。

也可以直接执行脚本一键启动（可用环境变量覆盖默认参数，括号内为默认值）：

```bash
EMIT_MS=1500 \
POLL_MS=500 \
Q_MIN=0.2 \
Q_MAX=0.8 \
AUTO_CONTEXT=叙述者 \
./scripts/run_cn_auto.sh
```

脚本还接受可选 CLI 参数（例如 `--max-events 100`），并允许用 `PYTHON_BIN` 指定 Python 解释器；若不设置上述变量则使用括号内的默认值。
