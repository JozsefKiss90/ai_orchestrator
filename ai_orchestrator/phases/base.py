# ai_orchestrator/phases/base.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..llm import LLMClient
from ..repo import Repo, CommandResult
from ..patching import FileContentPatch, Patch, UnifiedDiffPatch

# Backward compatibility alias:
FilePatch = FileContentPatch


class PhaseContext:
    """
    Per-run/per-node context container.
    Use ctx.state for all extensibility (context packs, prior failures, etc.).
    """

    def __init__(self, phase_name: str, extra_state: Optional[Dict] = None):
        self.phase_name = phase_name
        self.state: Dict = extra_state or {}


class Phase:
    name: str = "base"

    def select_files(
        self,
        repo: Repo,
        files: List[Path],
        llm: LLMClient,
        ctx: PhaseContext,
        max_files: int,
    ) -> List[Path]:
        return files[:max_files]

    def generate_patches(
        self,
        repo: Repo,
        files: List[Path],
        llm: LLMClient,
        ctx: PhaseContext,
    ) -> List[Patch]:
        raise NotImplementedError

    def evaluate_run(
        self,
        repo: Repo,
        test_result: CommandResult,
        patches: List[Patch],
        llm: LLMClient,
        ctx: PhaseContext,
    ) -> None:
        pass
