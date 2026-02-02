# ai_orchestrator/logging_utils.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RunLogContext:
    run_id: str
    run_name: str
    dry_run: bool
    branch: Optional[str] = None


class ContextAdapter(logging.LoggerAdapter):
    """
    Injects contextual fields into log records so every module can log with run/node/phase info.
    """
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        merged = {**self.extra, **extra}
        kwargs["extra"] = merged
        return msg, kwargs


def _fmt() -> logging.Formatter:
    # Keep readable + greppable
    return logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s | run=%(run_id)s node=%(node_id)s phase=%(phase)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _ensure_logrecord_defaults_installed() -> None:
    """
    Ensure formatter-required fields always exist, even for plain logging.getLogger(__name__)
    calls without adapters. This prevents KeyError in the formatter.
    """
    if getattr(logging, "_ai_orchestrator_record_factory", None) is not None:
        return

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        for k, v in {
            "run_id": "-",
            "run_name": "-",
            "dry_run": False,
            "branch": "",
            "node_id": "-",
            "phase": "-",
        }.items():
            if not hasattr(record, k):
                setattr(record, k, v)
        return record

    logging.setLogRecordFactory(record_factory)
    setattr(logging, "_ai_orchestrator_record_factory", record_factory)


def init_run_logging(
    *,
    run_dir: Path,
    level: str = "INFO",
) -> None:
    """
    Process-wide logging setup.
    - Console handler once
    - A run-scoped file handler (run.log) for the run_dir
    """
    _ensure_logrecord_defaults_installed()

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = _fmt()

    # Console (install once)
    if not getattr(root, "_ai_orchestrator_console", False):
        sh = logging.StreamHandler()
        sh.setLevel(root.level)
        sh.setFormatter(formatter)
        root.addHandler(sh)
        setattr(root, "_ai_orchestrator_console", True)

    # Run file handler (install per run_dir if not already present)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = (run_dir / "run.log").resolve()

    existing = getattr(root, "_ai_orchestrator_run_logs", set())
    if str(run_log_path) not in existing:
        fh = logging.FileHandler(run_log_path, encoding="utf-8")
        fh.setLevel(root.level)
        fh.setFormatter(formatter)
        root.addHandler(fh)
        existing.add(str(run_log_path))
        setattr(root, "_ai_orchestrator_run_logs", existing)


def make_logger(
    name: str,
    *,
    run: RunLogContext,
    node_id: str = "-",
    phase: str = "-",
) -> ContextAdapter:
    base = logging.getLogger(name)
    return ContextAdapter(
        base,
        {
            "run_id": run.run_id,
            "run_name": run.run_name,
            "dry_run": run.dry_run,
            "branch": run.branch or "",
            "node_id": node_id,
            "phase": phase,
        },
    )


def add_node_file_handler(
    *,
    node_dir: Path,
    level: Optional[int] = None,
) -> logging.Handler:
    """
    Adds a file handler that captures everything (all modules) into node.log.

    Returns the handler so the caller can remove it when the node finishes.
    """
    _ensure_logrecord_defaults_installed()

    root = logging.getLogger()
    formatter = _fmt()

    node_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(node_dir / "node.log", encoding="utf-8")
    fh.setLevel(level if level is not None else root.level)
    fh.setFormatter(formatter)
    root.addHandler(fh)
    return fh


def remove_handler(handler: logging.Handler) -> None:
    root = logging.getLogger()
    try:
        root.removeHandler(handler)
    except Exception:
        pass
    try:
        handler.close()
    except Exception:
        pass
