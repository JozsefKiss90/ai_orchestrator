# ai_orchestrator/phases/oop_refactor.py
from __future__ import annotations

from pathlib import Path
from typing import List

from .base import Phase, PhaseContext
from ..llm import LLMClient, SCHEMA_UNIFIED_DIFF_V1, SCHEMA_FILE_CONTENT_V1
from ..patching import Patch, UnifiedDiffPatch
from ..patching import FileContentPatch


class OopRefactorPhase(Phase):
    name = "oop_refactor"

    def generate_patches(
        self,
        repo,
        files: List[Path],
        llm: LLMClient,
        ctx: PhaseContext,
    ) -> List[Patch]:
        """
        Robust mode: return full file contents only (no unified diffs inside JSON).
        This avoids invalid JSON due to raw newlines in diff strings.

        Behavior:
          - If node objective says "validators only", emit no patches.
          - Otherwise, update existing files by returning full new contents.
          - May create new files by returning full contents as well.
        """
        system_prompt = (
            "You are a senior software engineer.\n"
            "Return ONLY JSON matching the schema.\n"
            "Do not include markdown fences.\n"
        )

        objective = (ctx.state.get("node_objective") or "").strip().lower()
        if "validators only" in objective or "run validators only" in objective or "tests-only" in objective:
            return []

        # Build context blobs
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
                module_summaries_blob = "MODULE SUMMARIES:\n" + "\n\n".join(
                    f"## Module: {m.get('module')}\n{m.get('summary','')}"
                    for m in ms
                    if isinstance(m, dict)
                )

        # Files given to the phase (selected)
        file_blobs = []
        selected_rel_paths: List[str] = []
        for path in files:
            rel = path.relative_to(repo.root).as_posix()
            selected_rel_paths.append(rel)
            content = path.read_text(encoding="utf-8", errors="ignore")
            file_blobs.append(
                f"// FILE: {rel}\n"
                "```code\n"
                f"{content}\n"
                "```"
            )

        user_prompt = (
            "TASK:\n"
            "- Apply minimal OOP refactoring if required by the goal.\n"
            "- Implement the current NODE OBJECTIVE precisely.\n"
            "- Preserve behavior: running `python hello.py` must print exactly `hello world`.\n\n"
            "OUTPUT FORMAT:\n"
            "Return JSON: { \"files\": [ {\"path\": string, \"new_content\": string}, ... ] }\n\n"
            "Rules:\n"
            "- You may edit ONLY files listed in SELECTED FILES unless you are explicitly creating a new file required by the goal.\n"
            "- For any file you edit, return its COMPLETE new content.\n"
            "- For any new file you create, return its COMPLETE content.\n"
            "- Paths must be repo-relative (use forward slashes).\n\n"
        )

        if objective:
            user_prompt += f"NODE OBJECTIVE:\n{objective}\n\n"

        if constraints_blob:
            user_prompt += constraints_blob + "\n\n"
        if module_summaries_blob:
            user_prompt += module_summaries_blob + "\n\n"

        user_prompt += "SELECTED FILES:\n" + "\n".join(f"- {p}" for p in selected_rel_paths) + "\n\n"
        user_prompt += "FILE CONTENTS:\n" + "\n\n".join(file_blobs)

        data = llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=SCHEMA_FILE_CONTENT_V1,
            schema_name="file_content_v1",
        )

        patches: List[Patch] = []
        for item in data.get("files", []):
            rp = (item.get("path") or "").strip().replace("\\", "/")
            new_content = item.get("new_content")
            if not rp or new_content is None:
                continue
            patches.append(FileContentPatch(path=repo.root / rp, new_content=str(new_content)))

        return patches



    def select_files(
            self,
            repo,
            files: List[Path],
            llm: LLMClient,
            ctx: PhaseContext,
            max_files: int,
        ) -> List[Path]:
            """
            Deterministic selection for the demo goal:
            Always include the goal doc and app/config.py if present, then fill with .py files.
            """
            goal = repo.root / "docs" / "ORCHESTRATOR_GOAL.md"
            app_init = repo.root / "app" / "__init__.py"
            app_cfg = repo.root / "app" / "config.py"

            selected: List[Path] = []
            for p in [goal, app_init, app_cfg]:
                if p.exists() and p.is_file():
                    selected.append(p)

            py = [p for p in files if p.suffix == ".py"]
            selected.extend(py)

            # Deduplicate, preserve order
            seen = set()
            out: List[Path] = []
            for p in selected:
                rp = p.relative_to(repo.root).as_posix()
                if rp not in seen:
                    seen.add(rp)
                    out.append(p)

            return out[: max(1, int(max_files))]

