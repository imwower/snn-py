# snn-py

The package uses the standard library logging module. Call `snn_py.logging_config.setup()` early in your program (CLI entry point, test bootstrap, or interactive session) to configure output formatting and align logger levels. By default the log level is `INFO`, but you can override it via the `SNN_PY_LOGLEVEL` environment variable.

Typical usage during development:

```bash
SNN_PY_LOGLEVEL=DEBUG python -m unittest -v
```

## 测试与演示

在提交前可以使用 INFO 级别日志跑完整测试：

```bash
SNN_PY_LOGLEVEL=INFO python -m unittest -v
```

若要体验端到端流程（LIF → 群体率 → Up/Down → 意向门 → 记忆 → 审计），可运行演示 CLI：

```bash
python -m snn_py.cli.demo --T 5 --policy policy.example.json --loglevel INFO
```

执行过程中会打印 JSON 日志，最终输出 `run_complete` 事件以及生成的提案数。
