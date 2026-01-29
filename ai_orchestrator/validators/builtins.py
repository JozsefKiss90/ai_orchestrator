# ai_orchestrator/validators/builtins.py
from __future__ import annotations

from typing import Optional

from .types import ValidatorSpec


def default_validators(test_command: Optional[str]) -> list[ValidatorSpec]:
    """
    Backward-compatible default: if no validators are specified in config,
    we run only tests (if configured).
    """
    if not test_command:
        # If test_command is absent, preserve prior behavior: effectively "skip tests"
        # and allow commit. You may choose to tighten this later.
        return [ValidatorSpec(name="tests", command="", shell=True)]
    return [ValidatorSpec(name="tests", command=test_command, shell=True)]
