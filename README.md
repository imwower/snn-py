# snn-py

The package uses the standard library logging module. Call `snn_py.logging_config.setup()` early in your program (CLI entry point, test bootstrap, or interactive session) to configure output formatting and align logger levels. By default the log level is `INFO`, but you can override it via the `SNN_PY_LOGLEVEL` environment variable.

Typical usage during development:

```bash
SNN_PY_LOGLEVEL=DEBUG python -m unittest -v
```

That command enables debug-level logs while the unit tests execute, letting you inspect the JSON setup event and any additional diagnostics emitted by `snn_py`.
