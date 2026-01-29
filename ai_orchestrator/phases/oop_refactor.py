# ai_orchestrator/phases/oop_refactor.py
from __future__ import annotations

from pathlib import Path
from typing import List

from .base import Phase, PhaseContext
from ..llm import LLMClient, SCHEMA_UNIFIED_DIFF_V1
from ..patching import Patch, UnifiedDiffPatch


class OopRefactorPhase(Phase):
    name = "oop_refactor"

    def generate_patches(
        self,
        repo,
        files: List[Path],
        llm: LLMClient,
        ctx: PhaseContext,
    ) -> List[Patch]:
        system_prompt = (
            "You are a senior software engineer refactoring code into a clean, "
            "object-oriented and modular architecture. "
            "You must not change public APIs or break obvious external contracts. "
            "Preserve behavior; improve structure."
        )

        context_pack = ctx.state.get("context_pack")

        constraints_blob = ""
        module_summaries_blob = ""
        if isinstance(context_pack, dict):
            constraints = context_pack.get("constraints", {}) or {}
            if constraints:
                constraints_blob = "CONSTRAINTS / ARCHITECTURE DOCS (excerpts):\n" + "\n\n".join(
                    f"## {k}\n{v}" for k, v in constraints.items() if isinstance(v, str) and v.strip()
                )

            ms = context_pack.get("module_summaries", []) or []
            if ms:
                module_summaries_blob = "MODULE SUMMARIES (cached, deterministic):\n" + "\n\n".join(
                    f"## Module: {m.get('module')}\n{m.get('summary','')}"
                    for m in ms
                    if isinstance(m, dict)
                )

        file_blobs = []
        for path in files:
            rel = path.relative_to(repo.root)
            content = path.read_text(encoding="utf-8", errors="ignore")
            file_blobs.append(
                f"// FILE: {rel}\n"
                "```code\n"
                f"{content}\n"
                "```"
            )

        user_prompt = (
            "Refactor the following files to improve OOP and modular design.\n"
            "Constraints:\n"
            "- Prefer minimal, surgical changes.\n"
            "- Preserve formatting where possible.\n"
            "- Avoid unrelated edits.\n"
            "- Do NOT rewrite whole files unless necessary.\n\n"
            "OUTPUT FORMAT (CRITICAL):\n"
            "- Return JSON strictly matching the schema.\n"
            "- The 'diff' field MUST contain ONLY a raw unified diff (no prose, no markdown fences).\n"
            "- The diff MUST be directly applyable with `git apply`.\n\n"
            "Unified diff contract (MUST satisfy ALL):\n"
            "1) For each changed file, include:\n"
            "   - diff --git a/<path> b/<path>\n"
            "   - --- a/<path>\n"
            "   - +++ b/<path>\n"
            "2) Every hunk header MUST include ranges, e.g.:\n"
            "   @@ -1,6 +1,11 @@\n"
            "   (Do NOT output '@@' without '-l,s +l,s'.)\n"
            "3) Use file paths exactly as shown in the FILE headers above.\n"
            "4) Include sufficient context lines so the patch applies cleanly.\n"
            "5) Keep edits minimal; do not introduce unrelated changes.\n\n"
        )

        if constraints_blob:
            user_prompt += "\n" + constraints_blob + "\n\n"
        if module_summaries_blob:
            user_prompt += "\n" + module_summaries_blob + "\n\n"

        user_prompt += "FILES:\n" + "\n\n".join(file_blobs)

        data = llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=SCHEMA_UNIFIED_DIFF_V1,
            schema_name="unified_diff_v1",
        )

        diff_text = (data.get("diff") or "").strip()
        return [UnifiedDiffPatch(diff_text=diff_text)]

    def select_files(
        self,
        repo,
        files: List[Path],
        llm: LLMClient,
        ctx: PhaseContext,
        max_files: int,
    ) -> List[Path]:
        # Prefer Python files for OOP refactor; fall back to anything.
        py = [p for p in files if p.suffix == ".py"]
        chosen = py if py else list(files)
        return chosen[: max(1, int(max_files))]
