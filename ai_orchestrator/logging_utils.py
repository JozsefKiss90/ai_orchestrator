# ai_orchestrator/logging_utils.py
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """
    Minimal JSON formatter. Keeps logs machine-parsable without external deps.
    """

    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Attach structured fields if provided via logger extra={"fields": {...}}
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            base.update(fields)

        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(base, ensure_ascii=False)


def setup_logging(
    *,
    level: Optional[str] = None,
    json_logs: Optional[bool] = None,
) -> None:
    """
    Central logging setup.

    Env overrides:
      - AO_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
      - AO_LOG_JSON=1|0
    """
    lvl = (level or os.getenv("AO_LOG_LEVEL") or "INFO").upper().strip()
    use_json_env = os.getenv("AO_LOG_JSON")
    if json_logs is None and use_json_env is not None:
        json_logs = use_json_env.strip() in ("1", "true", "TRUE", "yes", "YES")

    if json_logs is None:
        json_logs = False

    try:
        numeric = getattr(logging, lvl)
        if not isinstance(numeric, int):
            numeric = logging.INFO
    except Exception:
        numeric = logging.INFO

    root = logging.getLogger()
    root.setLevel(numeric)

    # Replace handlers (avoid duplicate logs in repeated runs / unit tests)
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root.addHandler(handler)

    # Quieten noisy libs if needed
    logging.getLogger("openai").setLevel(max(numeric, logging.WARNING))
