# ai_orchestrator/patching.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class FileContentPatch:
    """
    Backward-compatible patch: replace full file contents.
    """
    path: Path
    new_content: str


@dataclass(frozen=True)
class UnifiedDiffPatch:
    """
    Preferred patch: unified diff to apply using git apply.

    diff_text should include the standard unified diff headers, e.g.
      --- a/path
      +++ b/path
    """
    diff_text: str


Patch = Union[FileContentPatch, UnifiedDiffPatch]
