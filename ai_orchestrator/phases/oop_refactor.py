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
        Dynamic selection (no hard-coded filenames):

        - Extract repo-relative paths mentioned in ctx.state["node_objective"].
        - Prepend ctx.state["touched_paths"] (carried forward by the runner).
        - Fill remaining budget using a capped relevance score across repo files.
        """
        import re

        max_files = max(1, int(max_files))

        # ---------- helpers ----------

        def _normalize_relpath(s: str) -> str:
            s = s.strip().strip("\"'`")
            s = s.replace("\\", "/")
            # Trim trailing punctuation often attached in prose.
            s = s.rstrip(").,;:")
            # Remove leading ./ for consistency.
            if s.startswith("./"):
                s = s[2:]
            return s

        def _safe_repo_relpath(s: str) -> str | None:
            """
            Reject absolute paths, drive letters, parent traversal.
            Return normalized posix relpath if safe.
            """
            s = _normalize_relpath(s)
            if not s:
                return None
            # reject URLs
            if "://" in s:
                return None
            p = Path(s)
            if p.is_absolute():
                return None
            if len(p.parts) > 0 and p.parts[0].endswith(":"):
                return None
            if ".." in p.parts:
                return None
            return Path(p.as_posix()).as_posix()

        def _extract_paths_from_objective(text: str) -> List[str]:
            """
            Extract likely file paths from free text.
            Examples captured:
            - app/config.py
            - docs/goals/ORCHESTRATOR_GOAL_FILE_CREATION.md
            - tests/test_greeter.py
            """
            if not text:
                return []

            # Match tokens that contain at least one slash/backslash and some filename chars.
            # Avoid eating whole sentences; keep it conservative.
            candidates = re.findall(r"(?<!\w)([A-Za-z0-9_.\-]+(?:[\\/][A-Za-z0-9_.\-]+)+)(?!\w)", text)
            out: List[str] = []
            seen = set()
            for c in candidates:
                rp = _safe_repo_relpath(c)
                if not rp:
                    continue
                if rp not in seen:
                    seen.add(rp)
                    out.append(rp)
            return out

        def _score_candidate(p: Path, *, obj_tokens: set[str], obj_paths: set[str], touched: set[str]) -> int:
            """
            Capped relevance heuristic:
            - objective-mentioned paths highest
            - touched paths next
            - python files preferred
            - token overlap bonus
            - directory proximity bonus
            """
            rel = p.relative_to(repo.root).as_posix()
            name = p.name.lower()
            parts = {seg.lower() for seg in Path(rel).parts}

            score = 0
            if rel in obj_paths:
                score += 1000
            if rel in touched:
                score += 800

            # Prefer code, but don't exclude docs/tests/etc.
            if p.suffix == ".py":
                score += 50
            elif p.suffix in {".md", ".json", ".yaml", ".yml"}:
                score += 10

            # Token overlap (cheap “semantic” signal)
            score += 5 * len((parts | {name}) & obj_tokens)

            # Directory proximity: if candidate shares a directory segment with objective paths
            for op in obj_paths:
                op_parts = {seg.lower() for seg in Path(op).parts[:-1]}  # directory parts only
                if op_parts and (op_parts & parts):
                    score += 15
                    break

            # Small bump for top-level entrypoints (still not hard-coded names)
            if len(Path(rel).parts) == 1 and p.suffix == ".py":
                score += 10

            return score

        # ---------- build priority sets ----------

        objective = str(ctx.state.get("node_objective") or "")
        obj_paths_list = _extract_paths_from_objective(objective)

        # Touched paths from previous nodes (injected by runner)
        touched_list = ctx.state.get("touched_paths") or []
        if not isinstance(touched_list, list):
            touched_list = []

        # Normalize touched paths
        touched_norm: List[str] = []
        seen = set()
        for t in touched_list:
            if not isinstance(t, str):
                continue
            rp = _safe_repo_relpath(t)
            if not rp:
                continue
            if rp not in seen:
                seen.add(rp)
                touched_norm.append(rp)

        obj_paths = set(obj_paths_list)
        touched = set(touched_norm)

        # Tokenize objective for relevance fill
        obj_tokens = {tok.lower() for tok in re.findall(r"[A-Za-z0-9_.\-]+", objective) if tok}

        # ---------- select ----------
        selected: List[Path] = []
        selected_rel: set[str] = set()

        def _try_add_rel(rp: str) -> None:
            if len(selected) >= max_files:
                return
            p = repo.root / rp
            if not p.exists() or not p.is_file():
                return
            if rp in selected_rel:
                return
            selected.append(p)
            selected_rel.add(rp)

        # 1) Always include objective paths (if they exist)
        for rp in obj_paths_list:
            _try_add_rel(rp)

        # 2) Then include touched paths from upstream nodes
        for rp in touched_norm:
            _try_add_rel(rp)

        if len(selected) >= max_files:
            return selected[:max_files]

        # 3) Capped relevance fill across repo files
        # Score all files once; then take highest-scoring not already selected
        scored = []
        for p in files:
            if not p.is_file():
                continue
            rel = p.relative_to(repo.root).as_posix()
            if rel in selected_rel:
                continue
            scored.append(( _score_candidate(p, obj_tokens=obj_tokens, obj_paths=obj_paths, touched=touched), rel, p))

        scored.sort(key=lambda x: (-x[0], x[1]))  # score desc, then path asc for determinism
        for _, rel, p in scored:
            if len(selected) >= max_files:
                break
            selected.append(p)
            selected_rel.add(rel)

        return selected[:max_files]


