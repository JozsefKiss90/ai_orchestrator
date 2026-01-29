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


def init_run_logging(
    *,
    run_dir: Path,
    level: str = "INFO",
) -> None:
    """
    One-time setup for the process.
    Writes to console + run.log.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicated handlers if CLI calls multiple times in-process (tests, etc.)
    if getattr(root, "_ai_orchestrator_configured", False):
        return
    setattr(root, "_ai_orchestrator_configured", True)

    formatter = _fmt()

    # Console
    sh = logging.StreamHandler()
    sh.setLevel(root.level)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # Run file
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setLevel(root.level)
    fh.setFormatter(formatter)
    root.addHandler(fh)


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
) -> None:
    """
    Adds a file handler that captures everything (all modules) into node.log,
    so you can open one file and see what happened for that node.
    """
    root = logging.getLogger()
    formatter = _fmt()

    node_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(node_dir / "node.log", encoding="utf-8")
    fh.setLevel(level if level is not None else root.level)
    fh.setFormatter(formatter)
    root.addHandler(fh)
