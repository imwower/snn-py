import json

from snn_py import logging_config


def record_payload(record):
    formatter = logging_config.JSONLineFormatter(getattr(record, "run", {}))
    return json.loads(formatter.format(record))

