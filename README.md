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
