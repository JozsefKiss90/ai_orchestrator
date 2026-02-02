# ai_orchestrator/phases/scaffold.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .base import Phase, PhaseContext
from ..llm import LLMClient
from ..patching import Patch, FileContentPatch


SCHEMA_SCAFFOLD_V1: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["files", "notes"],
    "additionalProperties": False,
}


class ScaffoldPhase(Phase):
    """
    Pure scaffolding: create new files and minimal supporting edits if needed.
    Prefer FileContentPatch (not unified diff).
    """
    name = "scaffold"

    def generate_patches(
        self,
        repo,
        files: List[Path],
        llm: LLMClient,
        ctx: PhaseContext,
    ) -> List[Patch]:
        # Ensure the goal doc is visible if present
        goal_rel = "docs/ORCHESTRATOR_GOAL.md"
        goal_path = repo.root / goal_rel

        existing_goal_excerpt = ""
        if goal_path.exists():
            existing_goal_excerpt = goal_path.read_text(encoding="utf-8", errors="ignore")[:6000]

        system_prompt = (
            "You are a scaffolding tool.\n"
            "You create new files when required by the goal.\n"
            "Return ONLY JSON.\n"
        )

        user_prompt = (
            "Create required new files for the repo based on the goal.\n\n"
            "Hard requirement:\n"
            "- Ensure docs/goals/ORCHESTRATOR_GOAL_FILE_CREATION.md exists.\n"
            "- Keep it a short markdown (10-30 lines).\n"
            "- Do not change runtime behavior (hello.py prints exactly 'hello world').\n\n"
            "Existing goal excerpt:\n"
            f"{existing_goal_excerpt}\n\n"
            "Return JSON with:\n"
            "{ files: [ {path, content} ], notes: string }\n"
        )

        data = llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=SCHEMA_SCAFFOLD_V1,
            schema_name="scaffold_v1",
        )

        patches: List[Patch] = []
        for item in data.get("files", []):
            rp = (item.get("path") or "").strip().replace("\\", "/")
            content = item.get("content")
            if not rp or content is None:
                continue
            patches.append(FileContentPatch(path=repo.root / rp, new_content=str(content)))

        return patches
