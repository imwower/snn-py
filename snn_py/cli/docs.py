"""Generate Markdown API docs from in-line docstrings."""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from snn_py import logging_config

logger = logging_config.get_logger("snn_py.cli.docs")


def _parse_args(argv: Optional[Iterable[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Markdown docs for snn_py.")
    parser.add_argument("--out", default="docs.md", help="Output Markdown file.")
    parser.add_argument("--package", default="snn_py", help="Root package to document.")
    return parser.parse_args(argv)


def _iter_module_names(package_name: str) -> Iterable[str]:
    module = importlib.import_module(package_name)
    yield module.__name__
    if not hasattr(module, "__path__"):
        return
    for finder in pkgutil.walk_packages(module.__path__, module.__name__ + "."):
        yield finder.name


def _safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:  # pragma: no cover - logged diagnostics
        logger.warning(
            "unable to import module",
            extra={"event": "docs_skip_module", "meta": {"module": module_name}},
        )
        return None


def _format_signature(obj) -> str:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "()"
    return str(sig)


def _collect_members(module) -> List[Tuple[str, object]]:
    members: List[Tuple[str, object]] = []
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            if getattr(obj, "__module__", None) == module.__name__:
                members.append((name, obj))
    return members


def _render_module(module) -> Tuple[str, int]:
    members = _collect_members(module)
    module_doc = inspect.getdoc(module)
    if not module_doc and not members:
        return "", 0
    lines: List[str] = [f"## Module `{module.__name__}`"]
    if module_doc:
        lines.append("")
        lines.append(textwrap.dedent(module_doc).strip())
    for name, obj in members:
        kind = "class" if inspect.isclass(obj) else "function"
        signature = _format_signature(obj)
        doc = inspect.getdoc(obj) or "No docstring provided."
        lines.append("")
        lines.append(f"### {kind.title()} `{name}`")
        lines.append("")
        lines.append(f"Signature: `{name}{signature}`")
        lines.append("")
        lines.append(textwrap.dedent(doc).strip())
    lines.append("")
    return "\n".join(lines), len(members)


def _generate_docs(package_name: str) -> Tuple[str, int]:
    lines: List[str] = [f"# {package_name} API Reference", ""]
    total_symbols = 0
    for module_name in sorted(set(_iter_module_names(package_name))):
        module = _safe_import(module_name)
        if module is None:
            continue
        block, count = _render_module(module)
        if not block:
            continue
        lines.append(block)
        total_symbols += count
    content = "\n".join(lines).strip() + "\n"
    return content, total_symbols


def main(argv: Optional[Iterable[str]] = None) -> int:
    parsed = _parse_args(argv)
    logging_config.setup()
    content, symbol_count = _generate_docs(parsed.package)
    out_path = Path(parsed.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    logger.info(
        "",
        extra={"event": "docs_generated", "meta": {"out": str(out_path), "symbols": symbol_count}},
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
